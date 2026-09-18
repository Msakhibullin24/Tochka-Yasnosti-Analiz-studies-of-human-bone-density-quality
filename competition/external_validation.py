"""External check on a different vendor: Ramathibodi public DXA set (Hologic, expert quality labels).

    python external_validation.py --root ../data/external/ramathibodi/extracted/Osteoporosis/BMD

Nothing here is used for training. The images are report-style exports (inverted polarity, burned-in ROI),
so this measures how far the *measurements* and the *router* transfer to another device - not the
organiser-domain classifier. n = 20 patients: numbers are indicative only and are reported as such.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import measure_hip, measure_spine
from dxaqc.pipeline import Analyzer
from dxaqc.xlsx_read import read_first_sheet

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PAIRS = {  # expert label -> (our measurement, direction: +1 larger = violation, -1 smaller = violation)
    "lsd": ("spine_abs_angle_deg", +1), "lssco": ("spine_curve_max_mm", +1),
    "lso": ("spine_abs_centre_offset_mm", +1), "lsc": ("crest_min_frac", -1),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    rows = read_first_sheet(args.root / "DataTable.xlsx")
    head, table = rows[0], [dict(zip(rows[0], r)) for r in rows[1:] if r and r[0]]
    analyzer = Analyzer()
    spine, routed, inverted = [], {"spine_ok": 0, "hip_ok": 0, "n": 0}, 0
    for rec in table:
        folder = next((args.root / sub / rec["No"] / "images" for sub in ("Verification", "Annotation")
                       if (args.root / sub / rec["No"] / "images").exists()), None)
        if folder is None:
            continue
        s_img, h_img = read_dxa(folder / "spine_image.dcm"), read_dxa(folder / "hip_image.dcm")
        inverted += ("POLARITY_INVERTED" in s_img.warnings) + ("POLARITY_INVERTED" in h_img.warnings)
        routed["n"] += 1
        routed["spine_ok"] += analyzer.analyze(s_img)["region"] == "spine"
        routed["hip_ok"] += analyzer.analyze(h_img)["region"] != "spine"
        spine.append({**measure_spine(s_img.pixels, s_img.pixel_mm).features,
                      **{k: int(float(rec[k] or 0) > 0) for k in PAIRS}})
    out = {"dataset": "Ramathibodi BMD (Hologic Horizon A / Discovery A)", "patients": routed["n"],
           "images_with_polarity_auto_corrected": int(inverted),
           "router_accuracy": {"spine": routed["spine_ok"] / routed["n"], "hip": routed["hip_ok"] / routed["n"]},
           "measurement_vs_expert_label": {}}
    for label, (feat, sign) in PAIRS.items():
        y = np.array([r[label] for r in spine])
        x = np.array([r[feat] for r in spine]) * sign
        entry = {"our_measurement": feat, "positives": int(y.sum()), "n": int(len(y))}
        if 0 < y.sum() < len(y):
            entry["roc_auc"] = float(roc_auc_score(y, x))
        out["measurement_vs_expert_label"][label] = entry
    out["note"] = ("Indicative only (n=20, 2-7 positives per label). Images carry burned-in ROI lines and a different "
                   "presentation than the organiser GE export; the organiser-domain classifier is NOT validated here.")
    (HERE / "reports" / "external_ramathibodi.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
