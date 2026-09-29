"""Research-only local TabPFN v2 probes on organiser DXA geometric criteria.

The pinned transformer sees only training-fold geometry and never raw DICOM.
Threshold selection uses inner study-held-out folds. Existing OOF rows fix the
outer comparison to exactly the same studies as the current criterion model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tabpfn import TabPFNClassifier

from dxaqc.model import CRITERIA, SEED, _matrix, best_f1_threshold
from source_integrity import inspect_sources
from train import build_table

HERE = Path(__file__).resolve().parents[1]
TASKS = {
    'hip_position_rotation': ('hip', 150, 36, 4, 'current_score', 'current_pred'),
    'spine_axis': ('spine', 99, 10, 3, 'current_rf_lr_cnn_score', 'current_rf_lr_cnn_pred'),
}
CHECKPOINT_SHA256 = 'f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_predict(x, y, training, testing, checkpoint):
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(x[training])
    x_test = imputer.transform(x[testing])
    model = TabPFNClassifier(model_path=str(checkpoint), device='cpu', n_estimators=4,
                             n_jobs=4, random_state=SEED)
    model.fit(x_train, y[training])
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Unexpected TabPFN classes')
    scores = model.predict_proba(x_test)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite TabPFN probabilities')
    return scores


def counts(y, predicted):
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tp = int(((y == 1) & (predicted == 1)).sum())
    return {'tn_fp_fn_tp': [tn, fp, fn, tp],
            'f1': 2 * tp / max(1, 2 * tp + fp + fn),
            'sensitivity': tp / max(1, tp + fn),
            'specificity': tn / max(1, tn + fp)}


def evaluate(task, labels, geometry, baseline, checkpoint):
    region, expected_n, expected_pos, inner_folds, base_score, base_pred = TASKS[task]
    selected = np.flatnonzero(labels.region.eq('spine').to_numpy() if region == 'spine'
                             else labels.region.isin(['hip_left', 'hip_right']).to_numpy())
    y = labels.loc[selected, task].to_numpy(dtype=int)
    groups = labels.loc[selected, 'study_key'].to_numpy()
    if len(selected) != expected_n or int(y.sum()) != expected_pos:
        raise ValueError('Unexpected DXA cohort')
    names = CRITERIA[region][task]
    x = _matrix([geometry[i] for i in selected], names)
    if np.isnan(x).all(axis=0).any():
        raise ValueError('An entire geometry feature is missing')
    baseline = baseline.sort_values('index').reset_index(drop=True)
    expected = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(expected.split(selected, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise ValueError('Outer study leakage')
        inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=SEED + 100)
        training_scores = np.full(len(tr), np.nan)
        for itr, iva in inner.split(tr, y[tr], groups[tr]):
            if set(groups[tr[itr]]) & set(groups[tr[iva]]):
                raise ValueError('Inner study leakage')
            training_scores[iva] = fit_predict(x, y, tr[itr], tr[iva], checkpoint)
        if not np.isfinite(training_scores).all():
            raise ValueError('Incomplete inner predictions')
        threshold = best_f1_threshold(y[tr], training_scores)
        scores = fit_predict(x, y, tr, te, checkpoint)
        for j, position in enumerate(te):
            rows.append({'index': int(selected[position]), 'fold': fold,
                         'study_sha256': hashlib.sha256(str(groups[position]).encode()).hexdigest(),
                         'target': int(y[position]), 'tabpfn_score': float(scores[j]),
                         'tabpfn_pred': int(scores[j] >= threshold)})
    table = pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    if not np.array_equal(table['index'], baseline['index']) or not np.array_equal(table['fold'], baseline['fold']):
        raise ValueError('Outer folds differ from baseline')
    if not np.array_equal(table['target'], baseline['target']):
        raise ValueError('Targets differ from baseline')
    if 'study_sha256' in baseline:
        if not np.array_equal(table['study_sha256'], baseline['study_sha256']):
            raise ValueError('Study groups differ from baseline')
    elif 'study' in baseline:
        if not np.array_equal(table['study_sha256'],
                              baseline['study'].astype(str).map(lambda s: hashlib.sha256(s.encode()).hexdigest())):
            raise ValueError('Study groups differ from baseline')
    table['baseline_score'] = baseline[base_score]
    table['baseline_pred'] = baseline[base_pred]
    truth = table.target.to_numpy()
    metrics = {}
    for variant in ('baseline', 'tabpfn'):
        score = table[variant + '_score'].to_numpy()
        predicted = table[variant + '_pred'].to_numpy()
        metrics[variant] = {**counts(truth, predicted),
                            'roc_auc': float(roc_auc_score(truth, score)),
                            'average_precision': float(average_precision_score(truth, score))}
    rng = np.random.default_rng(SEED)
    study_rows = [np.flatnonzero(table.study_sha256.to_numpy() == study)
                  for study in table.study_sha256.unique()]
    differences = []
    for _ in range(2000):
        sample = np.concatenate([study_rows[i] for i in rng.integers(len(study_rows), size=len(study_rows))])
        differences.append(counts(truth[sample], table.tabpfn_pred.to_numpy()[sample])['f1'] -
                           counts(truth[sample], table.baseline_pred.to_numpy()[sample])['f1'])
    metrics['tabpfn']['paired_delta_f1_vs_baseline'] = metrics['tabpfn']['f1'] - metrics['baseline']['f1']
    metrics['tabpfn']['paired_delta_f1_ci95'] = [float(z) for z in np.percentile(differences, [2.5, 97.5])]
    return metrics, table, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-oof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new report path')
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('TabPFN v2 checkpoint checksum mismatch')
    labels_path = HERE / 'labels/image_labels.csv'
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organizer', labels_path, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(item['path'], item['pixel_sha256_current']) for item in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    geometry, _, _ = build_table(args.dataset, labels, HERE / '.cache', fingerprint)
    baseline = pd.read_csv(args.baseline_oof)
    metrics, table, names = evaluate(args.task, labels, geometry, baseline, args.checkpoint)
    report = {'scope': 'Research-only DXA criterion ablation', 'task': args.task,
              'protocol': 'Fixed 5-fold study-held-out outer CV matching baseline, training-only inner F1 threshold selection; pinned local TabPFN v2 with four estimators, geometry only, seed17',
              'features': names, 'images': len(table), 'studies': int(table.study_sha256.nunique()),
              'positives': int(table.target.sum()), 'variants': metrics,
              'labels_sha256': sha256(labels_path), 'source_fingerprint': fingerprint,
              'baseline_oof_sha256': sha256(args.baseline_oof),
              'checkpoint_sha256': CHECKPOINT_SHA256, 'tabpfn_version': '2.0.9',
              'code_sha256': sha256(Path(__file__)),
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Previously inspected organiser development DXA; no independent test.',
                              'Combined hip positioning/rotation label has no subtype ground truth.' if args.task.startswith('hip') else 'Only ten positive spine-axis labels; no independently reviewed vertebral angle.',
                              'Known anatomical region supplied; full routing and final quality not evaluated.',
                              'The transformer acts on geometry only and cannot correct missing landmarks.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table.to_csv(args.output.with_name(args.output.stem + '_oof.csv'), index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
