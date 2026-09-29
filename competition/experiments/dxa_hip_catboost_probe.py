"""Fixed CPU CatBoost probe for the organiser's combined hip positioning label.

Research only. Uses the same study-held-out folds and inner threshold protocol
as hip_rotation_ablation, with no tuning on outer test folds. The label does not
separate insufficient/excessive rotation or establish anatomical accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.model import CRITERIA, SEED, _matrix, best_f1_threshold
from hip_rotation_ablation import f1_counts
from source_integrity import inspect_sources
from train import build_table

HERE = Path(__file__).resolve().parents[1]
TARGET = 'hip_position_rotation'
FEATURES = CRITERIA['hip'][TARGET]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03,
                               l2_leaf_reg=5., loss_function='Logloss',
                               auto_class_weights='Balanced', random_seed=SEED,
                               thread_count=4, allow_writing_files=False,
                               verbose=False)
    model.fit(x[train], y[train])
    scores = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('CatBoost returned invalid probabilities')
    return scores


def evaluate(labels: pd.DataFrame, features: list[dict], baseline: pd.DataFrame):
    selected = np.flatnonzero(labels.region.isin(('hip_left', 'hip_right')).to_numpy())
    y = labels.iloc[selected][TARGET].to_numpy(dtype=int)
    groups = labels.iloc[selected].study_key.astype(str).to_numpy()
    x = _matrix([features[int(i)] for i in selected], FEATURES)
    if len(selected) != 150 or int(y.sum()) != 36 or x.shape != (150, len(FEATURES)):
        raise ValueError('Unexpected organiser hip cohort or feature schema')
    baseline = baseline.sort_values('index').reset_index(drop=True)
    if len(baseline) != len(selected) or not {'index', 'fold', 'study', 'target',
                                              'current_rf_lr_cnn_pred'} <= set(baseline):
        raise ValueError('Baseline OOF schema differs')
    rows = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(outer.split(selected, y, groups)):
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Outer study leakage')
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 100)
        inner_scores = np.full(len(train), np.nan)
        for inner_train, inner_valid in inner.split(train, y[train], groups[train]):
            if set(groups[train[inner_train]]) & set(groups[train[inner_valid]]):
                raise ValueError('Inner study leakage')
            inner_scores[inner_valid] = fit_predict(x, y, train[inner_train], train[inner_valid])
        if not np.isfinite(inner_scores).all():
            raise ValueError('Inner OOF predictions incomplete')
        threshold = best_f1_threshold(y[train], inner_scores)
        scores = fit_predict(x, y, train, test)
        for position, score in zip(test, scores):
            rows.append({'index': int(selected[position]), 'fold': fold,
                         'study': groups[position], 'target': int(y[position]),
                         'catboost_score': float(score), 'catboost_pred': int(score >= threshold)})
    table = pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    identity = ('index', 'fold', 'study', 'target')
    if not table[list(identity)].equals(baseline[list(identity)]):
        raise ValueError('Candidate and baseline OOF identities or folds differ')
    table['current_score'] = baseline.current_rf_lr_cnn_score
    table['current_pred'] = baseline.current_rf_lr_cnn_pred
    truth = table.target.to_numpy()
    variants = {}
    for name in ('current', 'catboost'):
        prediction = table[name + '_pred'].to_numpy()
        score = table[name + '_score'].to_numpy()
        variants[name] = {**f1_counts(truth, prediction),
                          'roc_auc': float(roc_auc_score(truth, score)),
                          'average_precision': float(average_precision_score(truth, score))}
    rng = np.random.default_rng(SEED)
    study_rows = [np.flatnonzero(groups == study) for study in np.unique(groups)]
    differences = []
    for _ in range(2000):
        sample = np.concatenate([study_rows[i] for i in rng.integers(len(study_rows), size=len(study_rows))])
        differences.append(f1_counts(truth[sample], table.catboost_pred.to_numpy()[sample])['f1'] -
                           f1_counts(truth[sample], table.current_pred.to_numpy()[sample])['f1'])
    variants['catboost']['paired_delta_f1_vs_current'] = variants['catboost']['f1'] - variants['current']['f1']
    variants['catboost']['paired_delta_f1_ci95'] = [float(v) for v in np.percentile(differences, (2.5, 97.5))]
    return variants, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline-oof', type=Path, default=HERE/'reports/hip_rotation_ablation_oof.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path')
    labels_path = HERE/'labels/image_labels.csv'
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    source_audit = inspect_sources([('organizer', labels_path, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(item['path'], item['pixel_sha256_current']) for item in source_audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    geometry, _, _ = build_table(args.dataset, labels, HERE/'.cache', fingerprint)
    baseline = pd.read_csv(args.baseline_oof)
    variants, table = evaluate(labels, geometry, baseline)
    report = {'scope': 'Research-only DXA hip positioning/rotation; released QC unchanged',
              'protocol': 'Fixed 5-fold study-held-out outer CV, 4-fold study-held-out inner F1 threshold; '
                          'CatBoost CPU 400 trees/depth4/learning-rate .03/l2 5/class balanced; seed17; '
                          'no test-fold tuning',
              'images': len(table), 'studies': int(table.study.nunique()),
              'positives': int(table.target.sum()), 'features': FEATURES,
              'catboost_version': catboost.__version__,
              'labels_sha256': sha256(labels_path), 'source_fingerprint': fingerprint,
              'baseline_oof_sha256': sha256(args.baseline_oof),
              'code_sha256': sha256(Path(__file__)), 'variants': variants,
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Combined label has no rotation subtypes or anatomical landmark truth.',
                              'The organiser cohort was already examined during model development.',
                              'Study grouping does not prove patient independence.',
                              'Known anatomical region is supplied; routing and final binary QC are not measured.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    table.to_csv(args.output.with_name(args.output.stem+'_oof.csv'), index=False)
    print(json.dumps(variants, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
