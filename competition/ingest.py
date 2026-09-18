"""Prepare a newly downloaded dataset for training: inventory -> de-duplication -> label template.

    python ingest.py --name mydata --root /path/to/downloaded/folder [--pixel-mm 0.6]

What it does (no network, nothing is modified in --root):
  1. walks the folder, reads every DICOM / PNG / JPG, skips unreadable files (listed in the report);
  2. drops exact pixel copies and images already present in labels/image_labels.csv (no train/test leakage);
  3. routes each image (spine / hip_right / hip_left) and runs the current model to produce SUGGESTIONS;
  4. writes labels/candidates_<name>.csv with EMPTY label columns + contact sheets for visual review.

Labels are never invented: the suggested_* columns are hints for the human reviewer only. Fill
region / quality_class / criteria columns (0 or 1), save as labels/<name>.csv and train with
    python train.py --dataset <organiser root> --extra <name>=labels/<name>.csv=/path/to/downloaded/folder
Reported metrics stay organiser-only, so you immediately see whether the new data really helped.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dxaqc.dicom_io import RASTER_SUFFIXES, DicomReadError, looks_like_dicom, read_any
from dxaqc.pipeline import LOW_REGION_CONFIDENCE, Analyzer

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
LABEL_COLUMNS = ["region", "quality_class", "spine_coverage", "spine_axis", "spine_artifact",
                 "hip_position_rotation", "hip_roi_coverage"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--pixel-mm", type=float, default=None, help="pixel size from the dataset documentation, if known")
    args = ap.parse_args()
    known = set(pd.read_csv(HERE / "labels" / "image_labels.csv").pixel_sha256)
    analyzer = Analyzer()
    rows, skipped, seen, thumbs = [], [], set(), []
    files = [p for p in sorted(args.root.rglob("*")) if p.is_file()
             and (p.suffix.lower() in RASTER_SUFFIXES or looks_like_dicom(p))]
    for n, path in enumerate(files):
        rel = path.relative_to(args.root).as_posix()
        try:
            img = read_any(path, args.pixel_mm)
        except DicomReadError as exc:
            skipped.append({"file": rel, "reason": exc.code})
            continue
        if img.pixel_sha256 in seen or img.pixel_sha256 in known:
            skipped.append({"file": rel, "reason": "DUPLICATE_PIXELS" if img.pixel_sha256 in seen else "ALREADY_IN_TRAINING_LABELS"})
            continue
        seen.add(img.pixel_sha256)
        res = analyzer.analyze(img)
        rows.append({"pixel_sha256": img.pixel_sha256, "study_key": img.study_uid or Path(rel).parent.as_posix() or rel,
                     "first_source_path": rel, "rows": img.pixels.shape[0], "columns": img.pixels.shape[1],
                     "pixel_mm": round(img.pixel_mm, 4), "pixel_mm_source": img.pixel_mm_source,
                     **{c: "" for c in LABEL_COLUMNS}, "comment": "",
                     "suggested_region": res["region"], "suggested_region_confidence": round(res["region_confidence"], 3),
                     "suggested_quality_prob": round(res["score"], 3),
                     "suggested_violations": ";".join(res["violations"]),
                     "needs_attention": ";".join(img.warnings + (["LOW_REGION_CONFIDENCE"] if res["region_confidence"] < LOW_REGION_CONFIDENCE else []))})
        thumb = cv2.resize(img.pixels, (180, 210))
        cv2.putText(thumb, str(len(rows) - 1), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        thumbs.append(thumb)
        if n % 50 == 0:
            print(f"{n}/{len(files)}", flush=True)
    out_csv = HERE / "labels" / f"candidates_{args.name}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    sheets = HERE / "labels" / f"candidates_{args.name}_sheets"
    sheets.mkdir(exist_ok=True)
    for k in range(0, len(thumbs), 48):
        chunk = thumbs[k:k + 48] + [np.zeros((210, 180), np.uint8)] * (-len(thumbs[k:k + 48]) % 8)
        grid = np.vstack([np.hstack(chunk[i:i + 8]) for i in range(0, len(chunk), 8)])
        cv2.imwrite(str(sheets / f"sheet_{k // 48:03d}.png"), grid)
    report = {"name": args.name, "files_seen": len(files), "unique_new_images": len(rows), "skipped": len(skipped),
              "skipped_by_reason": pd.Series([s["reason"] for s in skipped]).value_counts().to_dict() if skipped else {},
              "suggested_regions": pd.Series([r["suggested_region"] for r in rows]).value_counts().to_dict() if rows else {},
              "low_confidence_images": int(sum("LOW_REGION_CONFIDENCE" in r["needs_attention"] for r in rows)),
              "label_template": str(out_csv), "contact_sheets": str(sheets)}
    (HERE / "labels" / f"candidates_{args.name}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
