"""Paired audit of independent QC specialists on their shared held-out studies.

Reads immutable saved predictions. No image is inferred and no model is retrained.
The train-only hybrid is the primary comparator; the newer typed OOF decision is
reported separately because it used a different five-fold training protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dxaqc.model import VIOLATION_LABEL, group_of
from dxaqc.specialist_qc import digest
from evaluate_organizer_dataset import binary_metrics
from experiments.cnn_quality import valid_labels
from report_specialists import load_run
from train_specialist import split_indices

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "spine_convnext": "convnextv2-spine-finetune-v3",
    "hip_convnext": "convnextv2-hip-finetune-v3",
    "general_efficientnet": "efficientnet-b4-bce-v2",
}
CRITERIA = ("spine_coverage", "spine_axis", "spine_artifact",
            "hip_position_rotation", "hip_roi_coverage")


def paired_ci(y: np.ndarray, a_score: np.ndarray, a_pred: np.ndarray,
              b_score: np.ndarray, b_pred: np.ndarray, groups: np.ndarray,
              repeats: int = 2000) -> dict:
    """Cluster by study and resample both predictions with identical indices."""
    rng = np.random.default_rng(17)
    units = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    values: dict[str, list[float]] = {"delta_f1": [], "delta_roc_auc": [],
                                      "delta_sensitivity": [], "delta_specificity": []}
    for _ in range(repeats):
        sample = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
        first = binary_metrics(y[sample], a_pred[sample], a_score[sample])
        second = binary_metrics(y[sample], b_pred[sample], b_score[sample])
        for name, field in (("delta_f1", "f1"), ("delta_roc_auc", "roc_auc"),
                            ("delta_sensitivity", "sensitivity"),
                            ("delta_specificity", "specificity")):
            if first[field] is not None and second[field] is not None:
                values[name].append(first[field] - second[field])
    return {name: {"low": float(np.percentile(items, 2.5)),
                   "high": float(np.percentile(items, 97.5)),
                   "defined_resamples": len(items)}
            for name, items in values.items() if items}


def disagreements(indices: np.ndarray, truth: np.ndarray, candidate: np.ndarray,
                  baseline: np.ndarray, regions: np.ndarray, studies: np.ndarray) -> dict:
    """Keep review references without copying source paths or patient identifiers."""
    corrected, regressed = [], []
    for i in indices:
        if candidate[i] == baseline[i]:
            continue
        item = {"label_row": int(i), "region": str(regions[i]), "quality_true": int(truth[i]),
                "study_sha256": hashlib.sha256(str(studies[i]).encode()).hexdigest()}
        if candidate[i] == truth[i]:
            corrected.append(item)
        else:
            regressed.append(item)
    return {"candidate_corrected_baseline": corrected,
            "candidate_regressed_from_baseline": regressed,
            "corrected_count": len(corrected), "regressed_count": len(regressed)}


def evaluate(labels_path: Path, runs: Path, benchmark: Path, typed_oof: Path,
             repeats: int = 2000) -> dict:
    labels = valid_labels(labels_path)
    _, _, test = split_indices(labels)
    y = labels.quality_class.to_numpy(dtype=int)
    studies = labels.study_key.to_numpy(dtype=str)
    region = np.array([group_of(value) for value in labels.region])
    saved = json.loads((benchmark / "predictions.json").read_text())
    if len(saved) != len(labels) or [row["label_row"] for row in saved] != list(range(len(labels))):
        raise ValueError("Benchmark rows do not match labels")
    if any(saved[i]["quality_true"] != y[i] or saved[i]["true_region"] != labels.region[i]
           for i in range(len(labels))):
        raise ValueError("Benchmark truth or region mismatch")
    if set(np.flatnonzero([row["split"] == "test" for row in saved])) != set(test):
        raise ValueError("Benchmark test split mismatch")
    metrics = json.loads((benchmark / "metrics.json").read_text())
    if metrics["labels_sha256"] != digest(labels_path):
        raise ValueError("Benchmark labels checksum mismatch")
    base_score = np.array([row["hybrid_refit_score"] for row in saved], dtype=float)
    base_pred = np.array([row["hybrid_refit_pred"] for row in saved], dtype=int)
    if not np.isfinite(base_score).all() or ((base_score < 0) | (base_score > 1)).any():
        raise ValueError("Invalid hybrid benchmark scores")
    with typed_oof.open(newline="") as stream:
        typed_rows = list(csv.DictReader(stream))
    typed = {row["source_path"]: row for row in typed_rows}
    if len(typed) != len(labels) or len(typed_rows) != len(typed):
        raise ValueError("Typed OOF does not cover labels exactly once")
    ordered = []
    for row in labels.itertuples():
        item = typed.get(row.first_source_path)
        if (item is None or item["study"] != row.study_key or
                item["true_region"] != row.region or
                item["quality_true"] != str(int(row.quality_class)) or item["repeat"] != "0"):
            raise ValueError("Typed OOF identity mismatch")
        ordered.append(item)
    typed_score = np.array([float(row["quality_score"]) for row in ordered])
    typed_pred = np.array([int(row["quality_pred"]) for row in ordered])
    if not np.isfinite(typed_score).all() or ((typed_score < 0) | (typed_score > 1)).any():
        raise ValueError("Invalid typed OOF scores")
    output = {
        "protocol": "Fixed test by study; saved train-only hybrid and specialist weights; known anatomical region; validation-selected thresholds; no retraining or test-time threshold selection.",
        "limitations": ["Internal test was previously inspected during development; no independent clinical validation.",
                        "Only 46 test images and 10 positive quality labels; no patient-disjoint guarantee.",
                        "Typed v4 is a separate five-fold OOF reference, not the train-only paired benchmark.",
                        "Preprocessing changed since original label pixel hashes; verify expert labels before promotion."],
        "labels_sha256": digest(labels_path),
        "benchmark_predictions_sha256": digest(benchmark / "predictions.json"),
        "typed_oof_sha256": digest(typed_oof),
        "test": {"images": len(test), "studies": len(set(studies[test])),
                 "positives": int(y[test].sum())},
        "primary_baseline": binary_metrics(y[test], base_pred[test], base_score[test]),
        "typed_v4_secondary_reference": binary_metrics(y[test], typed_pred[test], typed_score[test]),
        "models": {}, "criterion_comparison": {}, "deployment_changed": False,
    }
    regional_score = np.full(len(labels), np.nan)
    regional_pred = np.full(len(labels), -1)
    for name, directory in MODELS.items():
        meta, _, scores, splits, groups = load_run(runs / directory, labels_path)
        if any(splits[i] != "test" for i in test) or not np.array_equal(groups, np.array([row["study_hash"] for row in saved])):
            raise ValueError(f"{name} study split or identity mismatch")
        scope = meta.get("region_scope", "all")
        mask = np.array([scope == "all" or scope == value for value in region])
        idx = test[mask[test]]
        score = scores[:, 0]
        cutoff = meta["thresholds"]["quality"]
        pred = (score >= cutoff).astype(int)
        if scope != "all":
            regional_score[idx] = score[idx]
            regional_pred[idx] = pred[idx]
        output["models"][name] = {
            "weight_sha256": meta["model_sha256"], "scope": scope,
            "predictions_sha256": digest(runs / directory / "predictions.json"),
            "quality_threshold": cutoff,
            "quality": binary_metrics(y[idx], pred[idx], score[idx]),
            "paired_vs_train_only_hybrid": paired_ci(y[idx], score[idx], pred[idx],
                                                     base_score[idx], base_pred[idx],
                                                     studies[idx], repeats),
            "train_only_hybrid_same_scope": binary_metrics(y[idx], base_pred[idx], base_score[idx]),
            "typed_v4_same_scope": binary_metrics(y[idx], typed_pred[idx], typed_score[idx]),
            "disagreements_vs_train_only_hybrid": disagreements(
                idx, y, pred, base_pred, region, studies),
        }
        if scope == "all":
            for anatomical in ("spine", "hip"):
                local = idx[region[idx] == anatomical]
                output["models"][name][f"quality_{anatomical}"] = binary_metrics(
                    y[local], pred[local], score[local])
        for j, criterion in enumerate(CRITERIA, 1):
            if scope != "all" and group_of("spine" if criterion.startswith("spine") else "hip_left") != scope:
                continue
            known = np.array([str(value) in ("0.0", "1.0", "0", "1")
                              for value in labels[criterion]])
            selected = idx[known[idx]]
            truth = labels[criterion].iloc[selected].to_numpy(dtype=int)
            threshold = meta["thresholds"][criterion]
            local_score = scores[selected, j]
            specialist_pred = (local_score >= threshold).astype(int) if threshold is not None else None
            official = VIOLATION_LABEL[criterion]
            core_pred = np.array([int(typed_pred[i] == 1 and official in ordered[i]["violation_type"].split(";"))
                                  for i in selected])
            output["criterion_comparison"].setdefault(criterion, {})[name] = {
                "n": len(selected), "positives": int(truth.sum()),
                "official_type": official, "threshold": threshold,
                "specialist": binary_metrics(truth, specialist_pred, local_score)
                if specialist_pred is not None and len(selected) else None,
                "typed_v4_official_decision": binary_metrics(truth, core_pred)
                if len(selected) else None,
            }
    if not np.isfinite(regional_score[test]).all() or (regional_pred[test] < 0).any():
        raise ValueError("Regional specialists do not cover the full test")
    if any(not np.isclose(regional_score[i], saved[i]["regional_finetuned_cnn_score"])
           or regional_pred[i] != saved[i]["regional_finetuned_cnn_pred"] for i in test):
        raise ValueError("Regional saved benchmark differs from independent members")
    output["regional_portfolio_routed"] = {
        "quality": binary_metrics(y[test], regional_pred[test], regional_score[test]),
        "paired_vs_train_only_hybrid": paired_ci(y[test], regional_score[test], regional_pred[test],
                                                  base_score[test], base_pred[test], studies[test], repeats),
        "disagreements_vs_train_only_hybrid": disagreements(
            test, y, regional_pred, base_pred, region, studies),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "competition/labels/image_labels.csv")
    parser.add_argument("--runs", type=Path, default=ROOT / "data/specialists/runs")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/specialists/benchmarks/paired-v3")
    parser.add_argument("--typed-oof", type=Path, default=ROOT / "competition/reports/typed_v4_pipeline_oof.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "competition/reports/independent_specialists_paired_2026_09_24.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    result = evaluate(args.labels, args.runs, args.benchmark, args.typed_oof, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "test": result["test"],
                      "models": {k: v["quality"]["f1"] for k, v in result["models"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
