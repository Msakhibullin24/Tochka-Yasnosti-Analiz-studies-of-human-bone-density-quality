"""Grouped repeated CV of linear probes on frozen embeddings.

Usage: python cv_eval.py <features.npy> [<features2.npy> ...]
Groups = study folder. Reports pooled out-of-fold ROC-AUC / AP / best-F1 per region group and target.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "competition" / "labels" / "image_labels.csv"

TARGETS = {
    "spine": ["quality_class", "spine_coverage", "spine_axis", "spine_artifact"],
    "hip": ["quality_class", "hip_position_rotation", "hip_roi_coverage"],
}


def oof_scores(X, y, groups, C, repeats=5, pca=None):
    out = np.zeros((repeats, len(y)))
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for tr, te in cv.split(X, y, groups):
            steps = [StandardScaler()]
            if pca:
                steps.append(PCA(n_components=min(pca, len(tr) - 1), random_state=0))
            steps.append(LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
            m = make_pipeline(*steps).fit(X[tr], y[tr])
            out[rep, te] = m.predict_proba(X[te])[:, 1]
    return out


def best_f1(y, s):
    ths = np.unique(s)
    return max(f1_score(y, s >= t) for t in ths)


def main() -> None:
    df = pd.read_csv(LABELS)
    X = np.concatenate([np.load(p) for p in sys.argv[1:]], axis=1)
    print("features", X.shape)
    for group, targets in TARGETS.items():
        mask = (df.region == "spine") if group == "spine" else (df.region != "spine")
        for t in targets:
            m = (mask & df[t].notna()).values
            y = df.loc[m, t].astype(int).values
            g = df.loc[m, "study_key"].values
            for C in (0.001, 0.01, 0.1):
                s = oof_scores(X[m], y, g, C)
                aucs = [roc_auc_score(y, r) for r in s]
                mean = s.mean(0)
                print(f"{group:5s} {t:22s} n={len(y):3d} pos={y.sum():2d} C={C:<6} "
                      f"AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} AP={average_precision_score(y, mean):.3f} "
                      f"bestF1={best_f1(y, mean):.3f}", flush=True)


if __name__ == "__main__":
    main()
