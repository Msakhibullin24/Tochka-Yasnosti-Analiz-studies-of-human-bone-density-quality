"""Negative control against leakage / overfitting of the validation protocol.

Labels are permuted BETWEEN STUDIES: the whole label block of a study (all its regions and criteria)
moves to another study that has the same set of regions, so the within-study label correlation - the
very thing study-grouped CV must protect against - is preserved. Then the same grouped CV is run.
An honest protocol must give ROC-AUC ~ 0.5 on permuted labels. Anything clearly above means leakage.

    python leakage_check.py --dataset "/path/Исследования" [--permutations 5]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dxaqc.model import CRITERIA, SEED, group_of
from train import HERE, build_table, oof

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--permutations", type=int, default=5)
    args = ap.parse_args()
    labels = pd.read_csv(HERE / "labels" / "image_labels.csv")
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    feats, emb, _ = build_table(args.dataset, labels, HERE / ".cache")
    studies = labels.study_key.values
    out = {}
    for group in ("spine", "hip"):
        idx = np.where(np.array([group_of(r) for r in labels.region]) == group)[0]
        real_q = labels.quality_class.values.astype(float)
        crit = {k: labels[k].values.astype(float) for k in CRITERIA[group]}
        s, _ = oof(group, feats, emb, real_q, crit, studies, idx, 5, SEED)
        real_auc = float(roc_auc_score(real_q[idx].astype(int), s))
        aucs = []
        for p in range(args.permutations):
            rng = np.random.default_rng(1000 + p)
            q = real_q.copy()
            c = {k: v.copy() for k, v in crit.items()}
            regions = labels.region.values
            by_signature: dict[tuple, list[str]] = {}
            for st in np.unique(studies[idx]):
                sig = tuple(sorted(regions[i] for i in idx if studies[i] == st))
                by_signature.setdefault(sig, []).append(st)
            for sig, members in by_signature.items():
                donors = list(rng.permutation(members))
                for target, donor in zip(members, donors):
                    for region in sig:
                        ti = next(i for i in idx if studies[i] == target and regions[i] == region)
                        di = next(i for i in idx if studies[i] == donor and regions[i] == region)
                        q[ti] = real_q[di]
                        for k in c:
                            c[k][ti] = crit[k][di]
            s, _ = oof(group, feats, emb, q, c, studies, idx, 5, SEED)
            aucs.append(float(roc_auc_score(q[idx].astype(int), s)))
            print(group, "permutation", p, "AUC", round(aucs[-1], 3), flush=True)
        out[group] = {"real_labels_auc": real_auc, "permuted_auc_mean": float(np.mean(aucs)),
                      "permuted_auc_values": aucs}
    (HERE / "reports" / "leakage_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
