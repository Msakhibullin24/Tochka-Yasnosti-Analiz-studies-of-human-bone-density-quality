"""Measure geometry repeatability on organiser DXA without fitting or editing labels.

The rotation check tests numeric equivariance, not clinical correctness. The hip
check uses the same brightness perturbation as the displayed landmark audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.anatomy import detect_landmarks  # noqa: E402
from dxaqc.decision import AXIS_LIMIT_DEG  # noqa: E402
from dxaqc.dicom_io import read_any  # noqa: E402
from dxaqc.geometry import measure_image  # noqa: E402


def rotated_physical(pixels: np.ndarray, degrees: float, mm_y: float, mm_x: float) -> np.ndarray:
    h, w = pixels.shape
    centre = np.array([(w - 1) / 2, (h - 1) / 2])
    scale = np.diag([mm_x, mm_y])
    rotation = cv2.getRotationMatrix2D((0, 0), degrees, 1.0)[:, :2]
    linear = np.linalg.solve(scale, rotation @ scale)
    affine = np.column_stack([linear, centre - linear @ centre])
    return cv2.warpAffine(pixels, affine, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def body_candidate_angle(pixels: np.ndarray, overlay: dict, mm_y: float, mm_x: float) -> tuple[int, float | None]:
    anatomy = detect_landmarks(pixels, "spine", overlay, mm_y, mm_x)
    points = next(item["points"] for item in anatomy["landmarks"]
                  if item["name"] == "vertebral_body_candidates")
    if len(points) < 3:
        return len(points), None
    xy = np.asarray(points)
    angle = float(np.degrees(np.arctan(np.polyfit(xy[:, 1] * mm_y, xy[:, 0] * mm_x, 1)[0])))
    return len(points), angle


def audit(dataset: Path, labels_path: Path) -> tuple[dict, pd.DataFrame]:
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    rows = []
    source_hash = hashlib.sha256()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Invalid value for VR UI")
        for i, label in labels.iterrows():
            img = read_any(dataset / label.first_source_path)
            source_hash.update(img.pixel_sha256.encode())
            mm_y, mm_x = img.pixel_mm, img.pixel_mm_x or img.pixel_mm
            pixels = np.ascontiguousarray(img.pixels[:, ::-1]) if label.region == "hip_left" else img.pixels
            base = measure_image(pixels, label.region, mm_y, mm_x)
            row = {"index": i, "study": label.study_key, "region": label.region,
                   "pixel_sha256": img.pixel_sha256}
            if label.region == "spine":
                angle = base.features["spine_angle_deg"]
                count, body_angle = body_candidate_angle(pixels, base.overlay, mm_y, mm_x)
                row.update({"axis_label": label.spine_axis,
                            "global_abs_angle_deg": abs(angle),
                            "body_candidate_abs_angle_deg": abs(body_angle) if body_angle is not None else np.nan,
                            "body_candidate_count": count})
                for degrees in (-7, 7):
                    changed = rotated_physical(pixels, degrees, mm_y, mm_x)
                    measured = measure_image(changed, "spine", mm_y, mm_x).features["spine_angle_deg"]
                    row[f"rotation_{degrees:+d}_error_deg"] = measured - angle - degrees
            else:
                changed = np.clip(pixels.astype(np.float32) * 1.1 - 3, 0, 255).astype(np.uint8)
                shifted = measure_image(changed, label.region, mm_y, mm_x).features
                row.update({"hip_position_label": label.hip_position_rotation,
                            "hip_roi_label": label.hip_roi_coverage,
                            "source_roi_status": img.source_roi.get("status"),
                            "shaft_angle_shift_deg": abs(base.features["shaft_abs_angle_deg"] -
                                                         shifted["shaft_abs_angle_deg"]),
                            "lesser_troch_shift_mm": abs(base.features["lesser_troch_protrusion_mm"] -
                                                        shifted["lesser_troch_protrusion_mm"])})
            rows.append(row)
    table = pd.DataFrame(rows)
    spine = table[table.region == "spine"]
    hip = table[table.region != "spine"]
    rotation = np.concatenate([spine[f"rotation_{degrees:+d}_error_deg"].to_numpy() for degrees in (-7, 7)])
    positive = spine[spine.axis_label == 1]
    body_available = positive.body_candidate_abs_angle_deg.notna()
    hip_shaft = hip.shaft_angle_shift_deg.to_numpy()
    hip_troch = hip.lesser_troch_shift_mm.to_numpy()
    result = {
        "protocol": "Organiser images; deterministic physical ±7° rotation and 1.1x−3 brightness perturbation",
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "decoded_pixel_hashes_sha256": source_hash.hexdigest(),
        "images": len(table), "studies": int(table.study.nunique()),
        "spine": {"n": len(spine), "axis_positive_labels": len(positive),
                  "positive_with_global_angle_at_most_5deg": int((positive.global_abs_angle_deg <= AXIS_LIMIT_DEG).sum()),
                  "positive_with_body_angle_at_most_5deg": int((positive.body_candidate_abs_angle_deg <= AXIS_LIMIT_DEG).sum()),
                  "positive_with_body_angle_available": int(body_available.sum()),
                  "rotation_trials": len(rotation),
                  "rotation_abs_error_median_deg": float(np.median(np.abs(rotation))),
                  "rotation_abs_error_max_deg": float(np.max(np.abs(rotation))),
                  "rotation_abs_error_over_1deg": int((np.abs(rotation) > 1).sum())},
        "hip": {"n": len(hip), "shaft_angle_shift_over_1deg": int((hip_shaft > 1).sum()),
                "lesser_troch_shift_over_5mm": int((hip_troch > 5).sum()),
                "lesser_troch_shift_p95_mm": float(np.percentile(hip_troch, 95)),
                "source_roi_absent": int((hip.source_roi_status == "absent").sum()),
                "positive_roi_labels": int((hip.hip_roi_label == 1).sum()),
                "positive_roi_labels_without_source_roi": int(((hip.hip_roi_label == 1) &
                                                                 (hip.source_roi_status == "absent")).sum())},
        "limitations": ["Synthetic equivariance does not validate anatomical localisation or labels.",
                        "Body candidates and the global axis share the same centreline tracker.",
                        "Brightness shift is one repeatability stress, not a clinical accuracy test.",
                        "Study groups are not proven patient-disjoint."],
    }
    return result, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=HERE / "labels" / "image_labels.csv")
    parser.add_argument("--output", type=Path, default=HERE / "reports" / "geometry_stress_2026_09_25.json")
    parser.add_argument("--private-details", type=Path, help="Per-image CSV; may contain study identifiers")
    args = parser.parse_args()
    result, table = audit(args.dataset, args.labels)
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.private_details:
        args.private_details.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.private_details, index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
