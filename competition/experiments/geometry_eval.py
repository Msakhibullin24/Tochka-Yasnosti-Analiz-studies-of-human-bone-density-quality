"""Compute geometric features for all reviewed images and print univariate AUCs."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "competition"))
from dxaqc.geometry import measure_spine, measure_hip
df = pd.read_csv(ROOT / "competition/labels/image_labels.csv")
rows = []
for r in df.itertuples():
    p = next((ROOT / "data/competition-v1/images").glob(f"*-IMG-{r.pixel_sha256[:20]}.png"))
    a = np.asarray(Image.open(p).convert("L"))
    if r.region == "hip_left": a = np.ascontiguousarray(a[:, ::-1])
    m = measure_spine(a) if r.region == "spine" else measure_hip(a)
    rows.append(m.features)
F = pd.DataFrame(rows); F.to_csv(ROOT / "data/runs/features/geometry.csv", index=False)
for grp, targets in (("spine", ["quality_class","spine_coverage","spine_axis","spine_artifact"]), ("hip", ["quality_class","hip_position_rotation","hip_roi_coverage"])):
    mask = (df.region == "spine") if grp == "spine" else (df.region != "spine")
    for t in targets:
        m = mask & df[t].notna(); y = df.loc[m, t].astype(int)
        res = []
        for c in F.columns:
            v = F.loc[m, c]
            if v.notna().sum() < len(v) * 0.5 or v.nunique() < 3: continue
            auc = roc_auc_score(y, v.fillna(v.median()))
            res.append((abs(auc - 0.5), c, auc))
        print(grp, t, " | ".join(f"{c}={a:.2f}" for _, c, a in sorted(res, reverse=True)[:7]))
