"""Exploratory, study-held-out ablation for the hip positioning criterion.

Thresholds are selected on inner training folds. This previously inspected
organiser set is diagnostic and must not be called an independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.model import CRITERIA, SEED, _cnn_lr, _lr, _matrix, _rf, best_f1_threshold  # noqa: E402
from dxaqc.embedding import WEIGHTS_SHA256  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402

TARGET = "hip_position_rotation"
VARIANTS = {
    "current_rf_lr_cnn": ("rf", "lr", "cnn"),
    "rf_cnn": ("rf", "cnn"),
    "rf_only": ("rf",),
    "cnn_only": ("cnn",),
}


def component_scores(feats: list[dict], embeddings: np.ndarray, y: np.ndarray,
                     train: np.ndarray, test: np.ndarray) -> dict[str, np.ndarray]:
    names = CRITERIA["hip"][TARGET]
    x_train = _matrix([feats[i] for i in train], names)
    x_test = _matrix([feats[i] for i in test], names)
    models = {
        "rf": (_rf().fit(x_train, y[train]), x_test),
        "lr": (_lr().fit(x_train, y[train]), x_test),
        "cnn": (_cnn_lr().fit(embeddings[train], y[train]), embeddings[test]),
    }
    return {name: model.predict_proba(data)[:, 1] for name, (model, data) in models.items()}


def f1_counts(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "tn_fp_fn_tp": [tn, fp, fn, tp],
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)}


def run(labels: pd.DataFrame, feats: list[dict], embeddings: np.ndarray) -> tuple[dict, pd.DataFrame]:
    selected = np.flatnonzero(labels.region.isin(["hip_left", "hip_right"]).values)
    y = labels.loc[selected, TARGET].to_numpy(dtype=int)
    y_all = labels[TARGET].to_numpy(dtype=float)
    studies = labels.loc[selected, "study_key"].to_numpy()
    rows = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer.split(selected, y, studies)):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 100)
        inner_scores = {name: np.empty(len(tr)) for name in VARIANTS}
        for itr, iva in inner.split(tr, y[tr], studies[tr]):
            parts = component_scores(feats, embeddings, y_all, selected[tr[itr]], selected[tr[iva]])
            for name, keys in VARIANTS.items():
                inner_scores[name][iva] = np.mean([parts[key] for key in keys], axis=0)
        thresholds = {name: best_f1_threshold(y[tr], score) for name, score in inner_scores.items()}
        parts = component_scores(feats, embeddings, y_all, selected[tr], selected[te])
        for j, pos in enumerate(te):
            row = {"index": int(selected[pos]), "fold": fold, "study": str(studies[pos]),
                   "region": str(labels.region.iloc[selected[pos]]), "target": int(y[pos])}
            for name, keys in VARIANTS.items():
                score = float(np.mean([parts[key][j] for key in keys]))
                row[name + "_score"] = score
                row[name + "_pred"] = int(score >= thresholds[name])
            rows.append(row)
    table = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    truth = table.target.to_numpy()
    report = {"protocol": "Exploratory 5-fold study-held-out OOF; 4-fold inner threshold selection",
              "criterion": TARGET, "images": len(table), "studies": int(table.study.nunique()),
              "positives": int(truth.sum()), "variants": {},
              "limitations": ["Organiser data and candidate models have been inspected before; this is not an independent test.",
                              "Criterion-stratified folds differ from typed_v4 quality-stratified folds.",
                              "Known region is supplied; router errors and final binary quality are not evaluated.",
                              "No patient-disjoint guarantee or expert landmark reference."]}
    for name in VARIANTS:
        score = table[name + "_score"].to_numpy()
        pred = table[name + "_pred"].to_numpy()
        report["variants"][name] = {**f1_counts(truth, pred),
                                    "roc_auc": float(roc_auc_score(truth, score)),
                                    "average_precision": float(average_precision_score(truth, score))}
    rng = np.random.default_rng(SEED)
    group_indices = [np.flatnonzero(table.study.to_numpy() == study) for study in table.study.unique()]
    baseline = table["current_rf_lr_cnn_pred"].to_numpy()
    for name in VARIANTS:
        if name == "current_rf_lr_cnn":
            continue
        candidate = table[name + "_pred"].to_numpy()
        deltas = []
        for _ in range(2000):
            sample = np.concatenate([group_indices[i] for i in rng.integers(len(group_indices), size=len(group_indices))])
            deltas.append(f1_counts(truth[sample], candidate[sample])["f1"] -
                          f1_counts(truth[sample], baseline[sample])["f1"])
        report["variants"][name]["paired_delta_f1_vs_current"] = (
            report["variants"][name]["f1"] - report["variants"]["current_rf_lr_cnn"]["f1"])
        report["variants"][name]["paired_delta_f1_ci95"] = [float(v) for v in np.percentile(deltas, [2.5, 97.5])]
    return report, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "reports" / "hip_rotation_ablation.json")
    args = parser.parse_args()
    labels_path = HERE / "labels" / "image_labels.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    integrity = inspect_sources([("organiser", labels_path, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(entry["path"], entry["pixel_sha256_current"]) for entry in integrity["entries"]],
        ensure_ascii=False).encode()).hexdigest()
    feats, embeddings, _ = build_table(args.dataset, labels, HERE / ".cache", fingerprint)
    report, table = run(labels, feats, embeddings)
    report["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    report["source_fingerprint"] = fingerprint
    report["encoder_sha256"] = WEIGHTS_SHA256
    report["experiment_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    table.to_csv(args.output.with_name(args.output.stem + "_oof.csv"), index=False)
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()
