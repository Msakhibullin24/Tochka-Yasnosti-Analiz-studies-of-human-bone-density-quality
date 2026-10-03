"""Research-only nested selection of DXA criterion classifiers.

Outer identities and routing are frozen to the baseline OOF. Models, scalers,
imputers and thresholds are fitted without the held-out studies. The measured
axis constraint is retained. This is not evaluation of a new clinical release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, SEED, _lr, _matrix, _rf, best_f1_threshold, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402
from verify_metric_candidate import compare  # noqa: E402

VARIANTS = ('baseline_ensemble', 'geometry_rf', 'geometry_lr',
            'embedding_lr_001', 'embedding_lr_01', 'combined_lr_001')


def positive_probability(model, values):
    probabilities = model.predict_proba(values)
    if 1 not in model.classes_:
        return np.zeros(len(values))
    return probabilities[:, list(model.classes_).index(1)]


def fit_head(kind, geometry, embedding, target, train):
    known = train[np.isfinite(target[train])]
    if not len(known):
        raise ValueError('Criterion has no training labels')
    if kind == 'baseline_ensemble':
        return tuple(fit_head(k, geometry, embedding, target, train)
                     for k in ('geometry_rf', 'geometry_lr', 'embedding_lr_001'))
    values = geometry if kind.startswith('geometry') else embedding
    if kind == 'combined_lr_001':
        values = np.column_stack([geometry, embedding])
    y = target[known].astype(int)
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy='constant', constant=int(y[0]))
        # DummyClassifier ignores values; avoid all-missing geometry warnings.
        model.fit(np.zeros((len(known), 1)), y)
    else:
        model = _rf() if kind == 'geometry_rf' else _lr(
            .3 if kind == 'geometry_lr' else .1 if kind == 'embedding_lr_01' else .01)
        model.fit(values[known], y)
    return kind, model


def predict_head(head, geometry, embedding, test):
    if len(head) == 3:
        return np.mean([predict_head(h, geometry, embedding, test) for h in head], axis=0)
    kind, model = head
    values = geometry if kind.startswith('geometry') else embedding
    if kind == 'combined_lr_001':
        values = np.column_stack([geometry, embedding])
    if isinstance(model, DummyClassifier):
        values = np.zeros((len(values), 1))
    return positive_probability(model, values[test])


def select_head(target, scores):
    """Select only from inner OOF; deterministic sensitivity/AP tie breaks."""
    known = np.isfinite(target)
    if not known.any() or any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Incomplete inner OOF scores or targets')
    truth = target[known].astype(int)
    ranked = []
    for serial, (name, values) in enumerate(scores.items()):
        threshold = best_f1_threshold(truth, values[known])
        metrics = binary_metrics(truth, (values[known] >= threshold).astype(int))
        ap = average_precision_score(truth, values[known]) if truth.sum() else 0.
        ranked.append(((metrics['f1'], metrics['sensitivity'], float(ap), -serial), name, threshold))
    _, name, threshold = max(ranked)
    return name, float(threshold)


def validate_outer(labels, baseline):
    labels = labels.reset_index(drop=True)
    if len(labels) != len(baseline) or baseline['index'].duplicated().any():
        raise ValueError('Incomplete or duplicated baseline')
    expected = labels.iloc[baseline['index'].to_numpy(dtype=int)]
    for column, reference in (('source_path', 'first_source_path'), ('study', 'study_key'),
                              ('true_region', 'region'), ('quality_true', 'quality_class')):
        if not np.array_equal(baseline[column].astype(str).to_numpy(), expected[reference].astype(str).to_numpy()):
            # pandas reads quality labels as floats in the label table.
            if column != 'quality_true' or not np.array_equal(baseline[column].to_numpy(), expected[reference].to_numpy()):
                raise ValueError(f'Baseline differs from labels: {column}')
    if set(baseline['repeat']) != {0} or baseline.groupby('study')['fold'].nunique().max() != 1:
        raise ValueError('Baseline splits a study or has multiple repeats')


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    validate_outer(labels, baseline)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    features, embeddings, _ = build_table(args.dataset, labels, HERE / '.cache', fingerprint)
    embedding_metadata = None
    if getattr(args, 'embeddings', None):
        from gpu_research.common import load_feature_bundle
        embeddings, embedding_metadata = load_feature_bundle(
            args.embeddings, labels, fingerprint, sha256(args.labels))
    features = [dict(f) for f in features]
    # Use exactly the baseline's runtime angle, including numbered-mask axes.
    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            angle = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
            features[row.index]['spine_abs_angle_deg'] = angle
    groups = labels.study_key.astype(str).to_numpy()
    anatomy = np.array([group_of(r) for r in labels.region])
    args.output.mkdir(parents=True)
    candidates = {name: baseline.copy() for name in (*VARIANTS, 'adaptive')}
    selection_audit = []
    for fold in sorted(baseline.fold.unique()):
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        train = np.flatnonzero(~np.isin(groups, list(held)))
        test_rows = baseline.index[baseline.fold == fold].to_numpy()
        for group in CRITERIA:
            selected = train[anatomy[train] == group]
            testing = test_rows[np.array([group_of(r) for r in baseline.loc[test_rows, 'predicted_region']]) == group]
            tests = baseline.loc[testing, 'index'].to_numpy(dtype=int)
            scores, thresholds = {}, {}
            for criterion, columns in CRITERIA[group].items():
                target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                geometry = _matrix(features, columns)
                inner_scores = {name: np.full(len(selected), np.nan) for name in VARIANTS}
                split = StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 100)
                for itr, iva in split.split(selected, labels.quality_class.iloc[selected], groups[selected]):
                    if set(groups[selected[itr]]) & set(groups[selected[iva]]):
                        raise ValueError('Inner study leakage')
                    # Reuse the component fits rather than training the ensemble twice.
                    heads = {name: fit_head(name, geometry, embeddings, target, selected[itr])
                             for name in VARIANTS if name != 'baseline_ensemble'}
                    parts = {name: predict_head(head, geometry, embeddings, selected[iva])
                             for name, head in heads.items()}
                    parts['baseline_ensemble'] = np.mean([parts[k] for k in
                        ('geometry_rf', 'geometry_lr', 'embedding_lr_001')], axis=0)
                    for name in VARIANTS:
                        inner_scores[name][iva] = parts[name]
                chosen, chosen_threshold = select_head(target[selected], inner_scores)
                heads = {name: fit_head(name, geometry, embeddings, target, selected)
                         for name in VARIANTS if name != 'baseline_ensemble'}
                parts = {name: predict_head(head, geometry, embeddings, tests) for name, head in heads.items()}
                parts['baseline_ensemble'] = np.mean([parts[k] for k in
                    ('geometry_rf', 'geometry_lr', 'embedding_lr_001')], axis=0)
                known = np.isfinite(target[selected])
                for name in VARIANTS:
                    scores.setdefault(name, {})[criterion] = parts[name]
                    thresholds.setdefault(name, {})[criterion] = best_f1_threshold(
                        target[selected][known].astype(int), inner_scores[name][known])
                scores.setdefault('adaptive', {})[criterion] = parts[chosen]
                thresholds.setdefault('adaptive', {})[criterion] = chosen_threshold
                selection_audit.append({'fold': int(fold), 'group': group, 'criterion': criterion,
                                        'selected': chosen, 'threshold': chosen_threshold,
                                        'training_studies': sorted(set(groups[selected])),
                                        'held_studies': sorted(held)})
            for name, predictions in scores.items():
                for position, row_index in enumerate(testing):
                    index = int(baseline.at[row_index, 'index'])
                    criterion_scores = {k: float(s[position]) for k, s in predictions.items()}
                    # Ranking score is a noisy OR, decision is the union of typed failures.
                    probability = float(1 - np.prod([1 - p for p in criterion_scores.values()]))
                    decision = decide(group, probability, criterion_scores, features[index],
                                      float(np.nextafter(1., np.inf)), thresholds[name])
                    updates = {'quality_score': probability, 'quality_pred': decision['quality'],
                               'violation_type': official_violation_type(decision['violations']),
                               'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                               'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                               'criterion_thresholds': json.dumps(thresholds[name], sort_keys=True),
                               'quality_threshold': decision['quality_threshold'],
                               'decision_version': 'research-nested-criterion-1',
                               'decision_reason': decision['decision_reason']}
                    for key, value in updates.items():
                        candidates[name].at[row_index, key] = value
        print(f'Completed outer fold {fold}', flush=True)
    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=200),
                         'screen': compare(args.baseline, path, args.labels)}
        print(name, reports[name]['metrics']['overall_quality']['f1'], flush=True)
    report = {'protocol': __doc__, 'variants': reports, 'selection_audit': selection_audit,
              'source_fingerprint': fingerprint, 'labels_sha256': sha256(args.labels),
              'baseline_sha256': sha256(args.baseline), 'code_sha256': sha256(Path(__file__)),
              'external_embedding_metadata': embedding_metadata,
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Previously inspected internal studies; patient independence unverified.',
                              'Baseline routing and runtime axes reused; not raw-input end-to-end CV.',
                              'No independent expert anatomical landmarks or masks.',
                              'Noisy-OR ranking score is not a calibrated clinical probability.',
                              'Multiple exploratory comparisons; no automatic release promotion.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, help='Audited frozen GPU features.npy with matching JSON')
    with threadpool_limits(limits=2):
        run(parser.parse_args())
