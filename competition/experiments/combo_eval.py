"""Compare geometry-only, CNN-only and stacked models under repeated grouped CV."""
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
REPEATS = 5

def models():
    return {
        "geo_lr": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)),
        "geo_rf": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=0, n_jobs=8)),
        "geo_et": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=0, n_jobs=8)),
        "cnn_lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000)),
    }

def run(Xg, Xe, y, groups):
    out = {k: np.zeros((REPEATS, len(y))) for k in list(models()) + ["avg_rf_cnn", "avg_all"]}
    for rep in range(REPEATS):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Xg, y, groups):
            p = {}
            for name, mk in models().items():
                X = Xe if name == "cnn_lr" else Xg
                p[name] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                out[name][rep, te] = p[name]
            out["avg_rf_cnn"][rep, te] = (p["geo_rf"] + p["cnn_lr"]) / 2
            out["avg_all"][rep, te] = (p["geo_rf"] + p["geo_lr"] + p["geo_et"] + p["cnn_lr"]) / 4
    return out

def bestf1(y, s): return max(f1_score(y, s >= t) for t in np.unique(s))

for grp, targets in (("spine", ["quality_class","spine_coverage","spine_axis","spine_artifact"]), ("hip", ["quality_class","hip_position_rotation","hip_roi_coverage"])):
    mask = (df.region == "spine") if grp == "spine" else (df.region != "spine")
    cols = [c for c in G.columns if G.loc[mask, c].notna().mean() > 0.5 and G.loc[mask, c].nunique() > 2]
    for t in targets:
        m = (mask & df[t].notna()).values
        y = df.loc[m, t].astype(int).values
        res = run(G.loc[m, cols].values, E[m], y, df.loc[m, "study_key"].values)
        print(f"== {grp} {t} n={len(y)} pos={y.sum()}")
        for k, s in res.items():
            aucs = [roc_auc_score(y, r) for r in s]
            print(f"   {k:11s} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} AP={average_precision_score(y, s.mean(0)):.3f} bestF1={bestf1(y, s.mean(0)):.3f}", flush=True)
