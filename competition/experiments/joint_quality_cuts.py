"""Select coupled spine/hip QC cutoffs using training-only inner OOF scores.

The objective is F1 over both anatomical groups, plus an explicit research
regularizer for mean type F1. No outer-fold scores or truths are read here.
"""
from __future__ import annotations
import numpy as np
from experiments.quality_gated_decoder import gated_decisions


def candidate_counts(state, spec, quality, preserve_independent):
    codes = state['codes']
    raw = np.column_stack([state['inner'][c] for c in codes])
    cuts = np.array([spec['thresholds'][c] for c in codes])
    truth = state['quality_truth'].astype(bool)
    thresholds = np.r_[np.unique(quality), np.nextafter(1., np.inf)]
    counts, type_f1 = [], []
    for cutoff in thresholds:
        selected, _ = gated_decisions(raw, cuts, codes, state['angles'], quality, cutoff, preserve_independent)
        predicted = selected.any(1)
        tp, fp, fn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())
        counts.append((tp, fp, fn))
        values = []
        for j in range(len(codes)):
            known = np.isfinite(state['criterion_truth'][:, j])
            y, p = state['criterion_truth'][known, j].astype(bool), selected[known, j]
            a, b, c = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
            values.append(2 * a / max(1, 2 * a + b + c))
        type_f1.append(sum(values))
    return thresholds, np.asarray(counts), np.asarray(type_f1)


def fit_joint_quality_cuts(groups, type_weight=.1, preserve_independent=False):
    if set(groups) != {'spine', 'hip'}:
        raise ValueError('Coupled cutoffs require both anatomical groups from the same outer fold')
    a, b = [candidate_counts(*groups[g], preserve_independent) for g in ('spine', 'hip')]
    counts = a[1][:, None, :] + b[1][None, :, :]
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    ntypes = sum(len(groups[g][0]['codes']) for g in groups)
    macro = (a[2][:, None] + b[2][None, :]) / ntypes
    objective = f1 + type_weight * macro
    # Same documented tie policy as the independent gate: overall F1,
    # fewer false negatives, then a more conservative cutoff.
    ranks = np.lexsort(((a[0][:, None] + b[0][None, :]).ravel(), -fn.ravel(), f1.ravel(), objective.ravel()))
    i, j = np.unravel_index(ranks[-1], objective.shape)
    return {'spine': float(a[0][i]), 'hip': float(b[0][j])}
