"""DXA-only study-held-out probe for the organiser's spine-axis label.

The experiment compares fixed feature families and training-only thresholds.
It does not claim a verified anatomical angle or independent clinical test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.model import CRITERIA, SEED, _cnn_lr, _lr, _matrix, _rf, best_f1_threshold
from source_integrity import inspect_sources
from train import build_table

HERE = Path(__file__).resolve().parents[1]
TARGET = 'spine_axis'
COMPACT = ('spine_abs_angle_deg', 'spine_abs_centre_offset_mm')
VARIANTS = ('current_rf_lr_cnn', 'geometry_rf_lr', 'compact_lr')


def fit_predict(geometry, embeddings, y, training, testing):
    x = _matrix([geometry[i] for i in training], CRITERIA['spine'][TARGET])
    z = _matrix([geometry[i] for i in testing], CRITERIA['spine'][TARGET])
    scores = {
        'rf': _rf().fit(x, y[training]).predict_proba(z)[:, 1],
        'lr': _lr().fit(x, y[training]).predict_proba(z)[:, 1],
        'cnn': _cnn_lr().fit(embeddings[training], y[training]).predict_proba(embeddings[testing])[:, 1],
        'compact': _lr(c=.3).fit(_matrix([geometry[i] for i in training], COMPACT),
                              y[training]).predict_proba(_matrix([geometry[i] for i in testing], COMPACT))[:, 1],
    }
    return {'current_rf_lr_cnn': (scores['rf'] + scores['lr'] + scores['cnn']) / 3,
            'geometry_rf_lr': (scores['rf'] + scores['lr']) / 2,
            'compact_lr': scores['compact']}


def metrics(y, score, predicted):
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tp = int(((y == 1) & (predicted == 1)).sum())
    return {'tn_fp_fn_tp': [tn, fp, fn, tp], 'f1': 2 * tp / max(1, 2 * tp + fp + fn),
            'sensitivity': tp / max(1, tp + fn), 'specificity': tn / max(1, tn + fp),
            'roc_auc': float(roc_auc_score(y, score)),
            'average_precision': float(average_precision_score(y, score))}


def evaluate(labels, geometry, embeddings):
    selected = np.flatnonzero(labels.region.eq('spine').to_numpy())
    y = labels.loc[selected, TARGET].to_numpy(dtype=int)
    groups = labels.loc[selected, 'study_key'].to_numpy()
    y_all = pd.to_numeric(labels[TARGET], errors='coerce').to_numpy()
    if len(selected) != 99 or int(y.sum()) != 10 or len(set(groups)) != 99:
        raise ValueError('Unexpected organizer spine DXA cohort')
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(selected, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise ValueError('Outer study leakage')
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED + 100)
        inner_scores = {name: np.full(len(tr), np.nan) for name in VARIANTS}
        for itr, iva in inner.split(tr, y[tr], groups[tr]):
            if set(groups[tr[itr]]) & set(groups[tr[iva]]):
                raise ValueError('Inner study leakage')
            scores = fit_predict(geometry, embeddings, y_all, selected[tr[itr]], selected[tr[iva]])
            for name in VARIANTS:
                inner_scores[name][iva] = scores[name]
        if any(not np.isfinite(score).all() for score in inner_scores.values()):
            raise ValueError('Incomplete inner predictions')
        thresholds = {name: best_f1_threshold(y[tr], inner_scores[name]) for name in VARIANTS}
        scores = fit_predict(geometry, embeddings, y_all, selected[tr], selected[te])
        for j, position in enumerate(te):
            row = {'index': int(selected[position]), 'fold': fold,
                   'study_sha256': hashlib.sha256(str(groups[position]).encode()).hexdigest(),
                   'target': int(y[position])}
            for name in VARIANTS:
                row[name + '_score'] = float(scores[name][j])
                row[name + '_pred'] = int(scores[name][j] >= thresholds[name])
            rows.append(row)
    table = pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    truth = table.target.to_numpy()
    report = {'scope': 'DXA-only organizer spine-axis label; research only',
              'protocol': 'Fixed 5-fold study-held-out outer CV, 3-fold inner threshold selection; seed17; no test-fold tuning',
              'images': len(table), 'studies': int(table.study_sha256.nunique()),
              'positives': int(truth.sum()), 'variants': {},
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Only ten positive DXA labels; confidence intervals are wide.',
                              'The organiser axis label has no independent per-vertebra angle reference.',
                              'Development cohort inspected previously; exploratory, not independent validation.',
                              'Known spine region is supplied; routing and final quality are not evaluated.']}
    for name in VARIANTS:
        report['variants'][name] = metrics(truth, table[name + '_score'].to_numpy(),
                                          table[name + '_pred'].to_numpy())
    rng = np.random.default_rng(SEED)
    studies = [np.flatnonzero(table.study_sha256.to_numpy() == study)
               for study in table.study_sha256.unique()]
    current = table.current_rf_lr_cnn_pred.to_numpy()
    for name in VARIANTS[1:]:
        candidate = table[name + '_pred'].to_numpy()
        differences = []
        for _ in range(2000):
            sampled = np.concatenate([studies[i] for i in rng.integers(len(studies), size=len(studies))])
            def f1(prediction):
                positives = truth[sampled] == 1
                tp = int((positives & (prediction[sampled] == 1)).sum())
                return 2 * tp / max(1, int((prediction[sampled] == 1).sum()) + int(positives.sum()))
            differences.append(f1(candidate) - f1(current))
        report['variants'][name]['paired_delta_f1_vs_current'] = (
            report['variants'][name]['f1'] - report['variants']['current_rf_lr_cnn']['f1'])
        report['variants'][name]['paired_delta_f1_ci95'] = [float(z) for z in np.percentile(differences, [2.5, 97.5])]
    return report, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new report path')
    labels_path = HERE / 'labels/image_labels.csv'
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organizer', labels_path, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(item['path'], item['pixel_sha256_current']) for item in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    geometry, embeddings, _ = build_table(args.dataset, labels, HERE / '.cache', fingerprint)
    report, table = evaluate(labels, geometry, embeddings)
    report.update(labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                  source_fingerprint=fingerprint, encoder_sha256=WEIGHTS_SHA256,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table.to_csv(args.output.with_name(args.output.stem + '_oof.csv'), index=False)
    print(json.dumps(report['variants'], indent=2))


if __name__ == '__main__':
    main()
