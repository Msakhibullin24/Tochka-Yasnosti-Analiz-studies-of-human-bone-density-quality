"""List the labels most worth a second expert look (confident out-of-fold disagreements).

    python review_candidates.py --dataset "/path/Исследования" [--top 30]

With ~250 images every wrong or borderline label costs measurable AUC. The list is built from honest
out-of-fold predictions (reports/oof_predictions.csv): the model never saw the image it disagrees on.
Output: reports/label_review.csv + reports/label_review_sheet.png. Labels are NOT changed automatically -
the expert decides; corrected labels go into labels/image_labels.csv with a note in the `comment` column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dxaqc.dicom_io import read_any

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    oof = pd.read_csv(HERE / "reports" / "oof_predictions.csv").dropna(subset=["oof_score"])
    lab = pd.read_csv(HERE / "labels" / "image_labels.csv")
    d = oof.merge(lab, on=["pixel_sha256", "study_key", "region", "quality_class"])
    # Scores of spine and hip models live on different scales: rank inside each group.
    d["group"] = np.where(d.region == "spine", "spine", "hip")
    d["rank_pct"] = d.groupby("group").oof_score.rank(pct=True)
    d["disagreement"] = np.where(d.quality_class == 1, 1 - d.rank_pct, d.rank_pct)
    d["kind"] = np.where(d.quality_class == 1, "эксперт: нарушение, модель: норма", "эксперт: норма, модель: нарушение")
    top = d.sort_values("disagreement", ascending=False).head(args.top).reset_index(drop=True)
    cols = ["excel_row", "study_key", "region", "quality_class", "oof_score", "kind", "spine_coverage", "spine_axis",
            "spine_artifact", "hip_position_rotation", "hip_roi_coverage", "comment", "first_source_path"]
    top[cols].to_csv(HERE / "reports" / "label_review.csv", index=False)
    tiles = []
    for i, r in top.iterrows():
        px = cv2.resize(read_any(args.dataset / r.first_source_path).pixels, (200, 230))
        cv2.putText(px, f"{i} r{int(r.excel_row)} y={int(r.quality_class)} p={r.oof_score:.2f}", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, 255, 1, cv2.LINE_AA)
        tiles.append(px)
    tiles += [np.zeros((230, 200), np.uint8)] * (-len(tiles) % 6)
    cv2.imwrite(str(HERE / "reports" / "label_review_sheet.png"),
                np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    print(top[["excel_row", "region", "quality_class", "oof_score", "kind", "comment"]].to_string())


if __name__ == "__main__":
    main()
