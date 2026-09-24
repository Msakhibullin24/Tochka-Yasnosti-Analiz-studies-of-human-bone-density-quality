"""Study-held-out comparison of frozen DAX features with the saved QC baseline.

Uses known anatomical regions and inner-fold thresholds. The resulting OOF is
research evidence, not an independent clinical test or a deployable QC model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_dxa_candidate import candidate_model, features, load_image

ROOT = Path(__file__).resolve().parents[1]

MODELS = ("dax_resnet18_a", "dax_vit_t16_a", "dax_vit_t8_a")


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.r_[score.min(), (np.sort(np.unique(score))[:-1] +
                                       np.sort(np.unique(score))[1:]) / 2]
    return float(max(candidates, key=lambda t: (f1(y, score >= t), t)))


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.01, class_weight="balanced", max_iter=3000, random_state=17))


def paired_interval(y, candidate_score, candidate_pred, baseline_score, baseline_pred,
                    studies, repeats=1000):
    rng = np.random.default_rng(17)
    indices = [np.flatnonzero(studies == study) for study in np.unique(studies)]
    values = {"candidate_f1": [], "baseline_f1": [], "delta_f1": [],
              "candidate_auc": [], "baseline_auc": [], "delta_auc": []}
    for _ in range(repeats):
        sampled = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        truth = y[sampled]
        first_f1 = f1(truth, candidate_pred[sampled])
        second_f1 = f1(truth, baseline_pred[sampled])
        values["candidate_f1"].append(first_f1)
        values["baseline_f1"].append(second_f1)
        values["delta_f1"].append(first_f1 - second_f1)
        if len(np.unique(truth)) == 2:
            first_auc = roc_auc_score(truth, candidate_score[sampled])
            second_auc = roc_auc_score(truth, baseline_score[sampled])
            values["candidate_auc"].append(first_auc)
            values["baseline_auc"].append(second_auc)
            values["delta_auc"].append(first_auc - second_auc)
    return {name: [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]
            for name, sample in values.items() if sample}


def checked_labels(path: Path, baseline_path: Path):
    with path.open(newline="") as stream:
        labels = [row for row in csv.DictReader(stream) if row["quality_class"] in ("0", "1")]
    with baseline_path.open(newline="") as stream:
        baseline = list(csv.DictReader(stream))
    if len(labels) != len(baseline) or len({row["first_source_path"] for row in labels}) != len(labels):
        raise ValueError("Labels and baseline must contain one row for each image")
    lookup = {row["first_source_path"]: row for row in labels}
    folds = {}
    for row in baseline:
        label = lookup.get(row["source_path"])
        if label is None or row["study"] != label["study_key"] or row["quality_true"] != label["quality_class"]:
            raise ValueError("Baseline OOF identity or label does not match source")
        if row["repeat"] != "0":
            raise ValueError("Expected one saved OOF repeat")
        if row["study"] in folds and folds[row["study"]] != int(row["fold"]):
            raise ValueError("Study split across folds")
        folds[row["study"]] = int(row["fold"])
    ordered = [lookup[row["source_path"]] for row in baseline]
    return ordered, baseline


def image_lookup(image_dir: Path) -> dict[str, Path]:
    lookup = {}
    for path in image_dir.glob("*-IMG-*.png"):
        key = path.stem.rsplit("-IMG-", 1)[-1]
        if key in lookup:
            raise ValueError(f"Duplicate preview key {key}")
        lookup[key] = path
    return lookup


def extract(model_name: str, labels: list[dict], image_dir: Path, assets: Path,
            cache_dir: Path, catalog: dict, label_sha: str) -> np.ndarray:
    import torch

    weight = catalog["weights"][model_name]
    cache = cache_dir / f"{model_name}-{label_sha[:12]}-{weight['sha256'][:12]}.npz"
    identities = np.array([row["pixel_sha256"] for row in labels])
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if np.array_equal(saved["identities"], identities):
                return saved["features"]
        raise ValueError(f"Feature cache identities changed: {cache}")
    torch.set_num_threads(4)
    model = candidate_model(model_name, assets, catalog)
    paths = image_lookup(image_dir)
    vectors = []
    for i, row in enumerate(labels):
        key = row["pixel_sha256"][:20]
        path = paths.get(key)
        if path is None:
            raise FileNotFoundError(f"Missing DXA preview for pixel hash {key}")
        image = load_image(path)
        if row["region"] == "hip_left":
            image = np.ascontiguousarray(image[:, ::-1])
        vectors.append(features(model, image))
        if (i + 1) % 50 == 0:
            print(f"{model_name}: {i + 1}/{len(labels)}", flush=True)
    result = np.stack(vectors)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, identities=identities, features=result)
    return result


def evaluate(model_name: str, x: np.ndarray, labels: list[dict], baseline: list[dict]) -> tuple[dict, list[dict]]:
    y = np.array([int(row["quality_class"]) for row in labels])
    groups = np.array([row["study_key"] for row in labels])
    region = np.array(["spine" if row["region"] == "spine" else "hip" for row in labels])
    fold = np.array([int(row["fold"]) for row in baseline])
    score = np.full(len(labels), np.nan)
    pred = np.full(len(labels), -1)
    thresholds = {}
    for outer in sorted(set(fold)):
        for anatomical in ("spine", "hip"):
            train = np.flatnonzero((fold != outer) & (region == anatomical))
            test = np.flatnonzero((fold == outer) & (region == anatomical))
            if not len(test):
                continue
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=117)
            inner_scores = np.full(len(train), np.nan)
            for inner_train, inner_val in inner.split(x[train], y[train], groups[train]):
                if len(np.unique(y[train[inner_train]])) < 2:
                    raise ValueError("Inner fold lacks a quality class")
                fitted = classifier().fit(x[train[inner_train]], y[train[inner_train]])
                inner_scores[inner_val] = fitted.predict_proba(x[train[inner_val]])[:, 1]
            if not np.isfinite(inner_scores).all():
                raise ValueError("Incomplete inner OOF scores")
            cutoff = threshold(y[train], inner_scores)
            fitted = classifier().fit(x[train], y[train])
            score[test] = fitted.predict_proba(x[test])[:, 1]
            pred[test] = (score[test] >= cutoff).astype(int)
            thresholds[f"fold_{outer}_{anatomical}"] = cutoff
    if not np.isfinite(score).all() or (pred < 0).any():
        raise ValueError("Incomplete candidate OOF")
    baseline_score = np.array([float(row["quality_score"]) for row in baseline])
    baseline_pred = np.array([int(row["quality_pred"]) for row in baseline])
    metrics = {"n": len(y), "f1": f1(y, pred), "roc_auc": float(roc_auc_score(y, score)),
               "baseline_f1": f1(y, baseline_pred),
               "baseline_roc_auc": float(roc_auc_score(y, baseline_score)),
               "by_region": {name: {"n": int((region == name).sum()),
                                    "f1": f1(y[region == name], pred[region == name]),
                                    "roc_auc": float(roc_auc_score(y[region == name], score[region == name]))}
                             for name in ("spine", "hip")}}
    metrics["paired_study_bootstrap_95"] = paired_interval(
        y, score, pred, baseline_score, baseline_pred, groups)
    rows = [{"source_path": label["first_source_path"], "study": label["study_key"],
             "region": label["region"], "fold": int(fold[i]), "quality_true": int(y[i]),
             "quality_score": float(score[i]), "quality_pred": int(pred[i])}
            for i, label in enumerate(labels)]
    return {"model": model_name, "metrics": metrics, "thresholds": thresholds,
            "protocol": "same outer study folds as saved baseline; 3-fold inner study OOF thresholds; known region; frozen features",
            "clinical_validation": False, "preprocessing_matches_upstream": False}, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "competition/labels/image_labels.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "competition/reports/scope_gate_1_9_pipeline_oof.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/competition-v1/images")
    parser.add_argument("--assets", type=Path, default=ROOT / "data/specialists")
    parser.add_argument("--output", type=Path, default=ROOT / "data/specialists/reports/dax-oof-v1")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "docs/competition/dxa_candidate_sources.json").read_text())
    label_sha = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    labels, baseline = checked_labels(args.labels, args.baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in MODELS:
        started = perf_counter()
        vector = extract(name, labels, args.images, args.assets, args.output / "features", catalog, label_sha)
        report, rows = evaluate(name, vector, labels, baseline)
        report["elapsed_seconds"] = round(perf_counter() - started, 2)
        report["labels_sha256"] = label_sha
        report["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
        (args.output / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        results.append(report)
        print(json.dumps({"model": name, **report["metrics"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
