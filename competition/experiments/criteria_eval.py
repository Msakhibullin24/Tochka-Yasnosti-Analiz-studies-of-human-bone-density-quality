"""Criterion-specific compact models and composition into quality_class (repeated grouped CV)."""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "competition/labels/image_labels.csv")
G = pd.read_csv(ROOT / "data/runs/features/geometry.csv")
E = np.load(ROOT / "data/runs/features/resnet18_320.npy")
SETS = {
    "spine_coverage": ["crest_left_frac", "crest_right_frac", "crest_min_frac", "crest_height_mm", "image_height_mm", "rib_signal"],
    "spine_axis": ["spine_abs_angle_deg", "spine_seg_max_abs_angle_deg", "spine_curve_rms_mm", "spine_curve_max_mm", "spine_abs_centre_offset_mm", "spine_seg_angle_range_deg"],
    "spine_artifact": ["tophat_lat_p999", "tophat_lat_p99", "tophat_lat_strong_frac", "tophat_col_p999", "sat_pixels", "sat_max_blob", "rib_signal", "lateral_mean"],
    "hip_position_rotation": ["shaft_abs_angle_deg", "shaft_angle_deg", "lesser_troch_protrusion_mm", "lesser_troch_area_mm2", "lesser_troch_notch_mm", "medial_concavity_mm", "neck_start_height_mm", "shaft_width_mm", "lateral_margin_mm", "troch_top_margin_mm", "bone_frac"],
    "hip_roi_coverage": ["image_height_mm", "ischium_bottom_margin_mm", "troch_top_margin_mm", "lateral_margin_mm", "bone_top_rows_frac", "ischium_touches_bottom"],
}
def rf(): return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0, n_jobs=8))
def lr(C=0.3): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
def cnn(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000))
def bestf1(y, s): return max(f1_score(y, s >= t) for t in np.unique(s))
def rep(name, y, s):
    aucs = [roc_auc_score(y, r) for r in s]
    print(f"   {name:24s} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} AP={average_precision_score(y, s.mean(0)):.3f} bestF1={bestf1(y, s.mean(0)):.3f}", flush=True)

R = 5
for grp, crits in (("spine", ["spine_coverage", "spine_axis", "spine_artifact"]), ("hip", ["hip_position_rotation", "hip_roi_coverage"])):
    mask = ((df.region == "spine") if grp == "spine" else (df.region != "spine")) & df.quality_class.notna()
    idx = np.where(mask.values)[0]
    yq = df.loc[mask, "quality_class"].astype(int).values
    groups = df.loc[mask, "study_key"].values
    allcols = [c for c in G.columns if G.loc[mask, c].notna().mean() > 0.5 and G.loc[mask, c].nunique() > 2]
    P = {c: {k: np.zeros((R, len(idx))) for k in ("rf", "lr", "cnn")} for c in crits}
    Q = {k: np.zeros((R, len(idx))) for k in ("direct_rf", "direct_cnn")}
    for r in range(R):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(idx, yq, groups):
            for c in crits:
                yc = df.loc[mask, c].astype(int).values
                X = G.loc[mask, SETS[c]].values
                P[c]["rf"][r, te] = rf().fit(X[tr], yc[tr]).predict_proba(X[te])[:, 1]
                P[c]["lr"][r, te] = lr().fit(X[tr], yc[tr]).predict_proba(X[te])[:, 1]
                P[c]["cnn"][r, te] = cnn().fit(E[idx][tr], yc[tr]).predict_proba(E[idx][te])[:, 1]
            Xa = G.loc[mask, allcols].values
            Q["direct_rf"][r, te] = rf().fit(Xa[tr], yq[tr]).predict_proba(Xa[te])[:, 1]
            Q["direct_cnn"][r, te] = cnn().fit(E[idx][tr], yq[tr]).predict_proba(E[idx][te])[:, 1]
    print("==", grp)
    for c in crits:
        yc = df.loc[mask, c].astype(int).values
        for k in ("rf", "lr", "cnn"): rep(f"{c[:14]} {k}", yc, P[c][k])
        rep(f"{c[:14]} rf+lr", yc, (P[c]["rf"] + P[c]["lr"]) / 2)
        rep(f"{c[:14]} rf+lr+cnn", yc, (P[c]["rf"] + P[c]["lr"] + P[c]["cnn"]) / 3)
    for combo in ("rf", "rf+lr", "rf+lr+cnn"):
        parts = combo.split("+")
        crit_p = [sum(P[c][k] for k in parts) / len(parts) for c in crits]
        nor = 1 - np.prod([1 - p for p in crit_p], axis=0)
        mx = np.max(crit_p, axis=0)
        rep(f"quality noisyOR[{combo}]", yq, nor); rep(f"quality max[{combo}]", yq, mx)
        rep(f"quality nOR+direct[{combo}]", yq, (nor + Q["direct_rf"] + Q["direct_cnn"]) / 3)
    rep("quality direct_rf", yq, Q["direct_rf"]); rep("quality direct_cnn", yq, Q["direct_cnn"])
    rep("quality direct_rf+cnn", yq, (Q["direct_rf"] + Q["direct_cnn"]) / 2)
