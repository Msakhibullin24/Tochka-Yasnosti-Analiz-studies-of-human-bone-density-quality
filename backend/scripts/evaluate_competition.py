"""Evaluate held-out per-image predictions, preserving failures in coverage counts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

KEY = ("path_to_study", "study_uid", "image_uid")
OUTPUT_COLUMNS = (*KEY, "anatomical_region", "quality_class", "violation_type",
                  "processing_status", "time_of_processing", "quality_probability")
TRUTH_COLUMNS = (*KEY, "anatomical_region", "quality_class", "group_id")


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        if set(required) - set(reader.fieldnames):
            raise ValueError(f"Missing columns: {sorted(set(required) - set(reader.fieldnames))}")
        return list(reader)


def indexed(rows):
    result = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in KEY)
        if not all(key) or key in result:
            raise ValueError("Missing identifiers or duplicate input row")
        result[key] = row
    return result


def binary_metrics(rows):
    if not rows:
        return {}
    y = np.array([r["y"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    score = np.array([r["score"] for r in rows])
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(rows), "positive": int(y.sum()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2 if len(set(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, score)) if np.any(y) else None,
        "brier": float(np.mean((y - score) ** 2)),
    }


def summarize(rows, repeats, seed):
    valid = [r for r in rows if r["status"] == "Success" and r["y"] is not None]
    labeled = [r for r in rows if r["y"] is not None]
    result = {
        "expected_images": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
        "labeled_images": len(labeled), "scored_images": len(valid),
        "success_fraction": sum(r["status"] == "Success" for r in rows) / len(rows),
        "labeled_coverage": len(valid) / len(labeled) if labeled else None,
        "unscored_positive": sum(r["y"] == 1 and r["status"] != "Success" for r in rows),
        "unscored_negative": sum(r["y"] == 0 and r["status"] != "Success" for r in rows),
        "metrics_on_successful_labeled_images": binary_metrics(valid),
    }
    groups = defaultdict(list)
    for row in valid:
        groups[row["group"]].append(row)
    result["scored_groups"] = len(groups)
    intervals = {}
    if len(groups) >= 2 and repeats:
        rng = np.random.default_rng(seed)
        keys = sorted(groups)
        samples = defaultdict(list)
        for _ in range(repeats):
            batch = [row for i in rng.integers(len(keys), size=len(keys)) for row in groups[keys[i]]]
            metrics = binary_metrics(batch)
            for name in ("f1", "roc_auc", "sensitivity", "specificity", "balanced_accuracy", "average_precision"):
                if metrics[name] is not None:
                    samples[name].append(metrics[name])
        for name, values in samples.items():
            intervals[name] = {"lower": float(np.quantile(values, 0.025)),
                               "upper": float(np.quantile(values, 0.975)), "valid_resamples": len(values)}
    result["group_bootstrap_95_ci"] = intervals
    return result


def evaluate(truth, predictions, repeats=1000, seed=17):
    actual, submitted = indexed(truth), indexed(predictions)
    if not actual:
        raise ValueError("Empty reference set")
    if submitted.keys() - actual.keys():
        raise ValueError("Predictions contain rows outside the reference set")
    records, times = [], []
    region_correct = region_scored = 0
    group_by_study = {}
    for key, target in actual.items():
        if target.get("quality_class") not in ("", "0", "1"):
            raise ValueError("Reference quality_class must be 0, 1, or empty (unknown)")
        group, region = target.get("group_id"), target.get("anatomical_region")
        if not group or not region:
            raise ValueError("Reference group_id and anatomical_region are required")
        previous = group_by_study.setdefault(key[1], group)
        if previous != group:
            raise ValueError("One study cannot belong to multiple bootstrap groups")
        pred = submitted.get(key)
        status = pred.get("processing_status") if pred else "Missing"
        if pred and status not in ("Success", "Failure"):
            raise ValueError("processing_status must be Success or Failure")
        row = {"y": int(target["quality_class"]) if target["quality_class"] else None,
               "group": group, "region": region, "status": status}
        if pred:
            seconds = float(pred["time_of_processing"])
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError("Processing time must be finite and nonnegative")
            times.append(seconds)
        if status == "Success":
            score = float(pred["quality_probability"])
            if pred["quality_class"] not in ("0", "1") or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Success requires binary class and finite probability in [0, 1]")
            if not pred.get("anatomical_region"):
                raise ValueError("Success requires anatomical_region")
            row.update(pred=int(pred["quality_class"]), score=score)
            region_scored += 1
            region_correct += pred["anatomical_region"] == region
        records.append(row)
    return {
        "scope": "development evaluation, not the organizer scoring formula",
        "notes": ["Binary metrics condition on successful labeled images; coverage and failures are separate.",
                  "Groups must preserve patients/related studies; this tool cannot verify training split leakage.",
                  "Bootstrap describes fixed predictions, not model selection uncertainty.",
                  "Average precision is AP, not trapezoidal PR AUC. Undefined metrics are null.",
                  "Classes are used as submitted; probabilities are not rethresholded or tuned here.",
                  "Per-image CSV durations do not prove the 180-second full-study latency requirement."],
        "bootstrap": {"repeats": repeats, "seed": seed},
        "overall": summarize(records, repeats, seed),
        "by_reference_region": {region: summarize([r for r in records if r["region"] == region], repeats, seed)
                                for region in sorted({r["region"] for r in records})},
        "region_accuracy_on_success": region_correct / region_scored if region_scored else None,
        "reported_image_seconds": dict(zip(("p50", "p95", "max"), map(float, np.quantile(times, [0.5, 0.95, 1])))) if times else {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    result = evaluate(read_csv(args.truth, TRUTH_COLUMNS), read_csv(args.predictions, OUTPUT_COLUMNS), args.bootstrap)
    result["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("truth", args.truth), ("predictions", args.predictions))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}; coverage={result['overall']['labeled_coverage']}")


if __name__ == "__main__":
    main()
