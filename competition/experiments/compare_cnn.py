"""Pair held-out CNN scores with the existing project model on identical DXA images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from dxaqc.model import group_of


def scores(y: np.ndarray, values: np.ndarray) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, values)), 4),
            "average_precision": round(float(average_precision_score(y, values)), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cnn = pd.read_csv(args.cnn)
    base = pd.read_csv(args.baseline)
    keys = ["pixel_sha256", "study_key", "region", "quality_class"]
    for name, table in (("cnn", cnn), ("baseline", base)):
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate labelled images in {name}")
    paired = cnn.merge(base[keys + ["oof_score", "oof_pred"]], on=keys, validate="one_to_one")
    if len(paired) != len(cnn) or len(paired) != len(base):
        raise ValueError("CNN and baseline must score exactly the same labelled images")
    report = {"scope": "paired OOF scores on the same organiser images",
              "caveat": "baseline is the existing repeated-CV hybrid; CNN uses one 5-fold CV. "
                        "This is exploratory model selection, not a locked independent test.",
              "images": len(paired), "studies": paired.study_key.nunique(), "by_region": {}}
    paired["group"] = paired.region.map(group_of)
    paired["cnn_pred_05"] = (paired.cnn_score >= .5).astype(int)
    paired["cnn_error"] = paired.cnn_pred_05 != paired.quality_class
    paired["hybrid_error"] = paired.oof_pred != paired.quality_class
    for group, table in paired.groupby("group"):
        y = table.quality_class.to_numpy(dtype=int)
        report["by_region"][group] = {
            "n": len(table), "positives": int(y.sum()),
            "cnn": scores(y, table.cnn_score.to_numpy()),
            "hybrid": scores(y, table.oof_score.to_numpy()),
            "cnn_errors_hybrid_correct": int((table.cnn_error & ~table.hybrid_error).sum()),
            "hybrid_errors_cnn_correct": int((table.hybrid_error & ~table.cnn_error).sum()),
            "both_wrong": int((table.cnn_error & table.hybrid_error).sum()),
        }
    labels = pd.read_csv(args.labels)
    criteria = ["spine_coverage", "spine_axis", "spine_artifact",
                "hip_position_rotation", "hip_roi_coverage"]
    enriched = paired.merge(labels[["pixel_sha256"] + criteria], on="pixel_sha256", validate="one_to_one")
    if len(enriched) != len(paired):
        raise ValueError("Criterion labels must cover the entire paired cohort")
    report["positive_criterion_misses"] = {}
    for criterion in criteria:
        positive = enriched[enriched[criterion] == 1]
        report["positive_criterion_misses"][criterion] = {
            "positives": len(positive),
            "cnn_quality_missed": int((positive.cnn_pred_05 == 0).sum()),
            "hybrid_quality_missed": int((positive.oof_pred == 0).sum()),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output / "paired_predictions.csv", index=False)
    paired[paired.cnn_error | paired.hybrid_error].to_csv(args.output / "error_cases.csv", index=False)
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
