"""Train calibrated QC heads with all score generation inside each outer split.

Global OOF tables are NOT valid training features for a second outer CV: models
that generated their training rows may have seen the current outer-test labels.
Here inner OOF scores are rebuilt exclusively on outer-training studies. Frozen
features may be shared, provided the encoder never learned from this cohort.
Routing and angles still come from the historical baseline: conditional research
QC, not an independent or raw-input end-to-end validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide
from dxaqc.model import CRITERIA, _matrix, group_of, official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from experiments.nested_criterion_upgrade import fit_head, predict_head, validate_outer
from experiments.verify_metric_candidate import compare
from source_integrity import inspect_sources
from train import build_table


def threshold(y, scores):
    """Enumerate attainable decisions, including all-negative; conservative tie break."""
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    if not len(y) or not np.isfinite(scores).all():
        raise ValueError('Empty labels or nonfinite scores')
    choices = np.r_[np.unique(scores), np.nextafter(1., np.inf)]
    best = (-1., -1., -1.)
    cut = choices[-1]
    for value in choices:
        p = scores >= value
        tp, fp = np.sum(p & (y == 1)), np.sum(p & (y == 0))
        fn = np.sum(~p & (y == 1))
        rank = (2 * tp / max(2 * tp + fp + fn, 1), -float(fp), float(value))
        if rank > best:
            best, cut = rank, float(value)
    return cut


def grouped_threshold(y, scores, groups, bags, seed):
    """Resample complete studies, preserving dependence between both hips."""
    unique = np.unique(groups)
    rows = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    cuts = []
    for _ in range(bags):
        selected = np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        if len(np.unique(y[selected])) == 2:
            cuts.append(threshold(y[selected], scores[selected]))
    return float(np.median(cuts)) if cuts else threshold(y, scores)


def calibrated(model, values):
    values = np.asarray(values, float)
    if isinstance(model, float):
        return np.full(values.shape, model)
    # A regularized sigmoid on the raw score, no class balancing: retain prevalence.
    return model.predict_proba(values.reshape(-1, 1))[:, 1]


def train_criterion(geometry, embedding, target, studies, train, test,
                    kind='baseline_ensemble', bags=100, seed=17):
    """Only target[train] can influence returned models, thresholds and predictions."""
    train, test = np.asarray(train), np.asarray(test)
    if set(studies[train]) & set(studies[test]):
        raise ValueError('Outer study leakage')
    known = train[np.isfinite(target[train])]
    if not len(known) or len(np.unique(studies[train])) < 2:
        raise ValueError('Need labelled training rows and at least two training studies')
    scores = np.full(len(train), np.nan)
    splits = []
    splitter = GroupKFold(min(4, len(np.unique(studies[train]))))
    for itr, iva in splitter.split(train, groups=studies[train]):
        inner_train, inner_val = train[itr], train[iva]
        if not np.isfinite(target[inner_train]).any():
            raise ValueError('Inner split has no labels')
        head = fit_head(kind, geometry, embedding, target, inner_train)
        scores[iva] = predict_head(head, geometry, embedding, inner_val)
        splits.append({'training_studies': sorted(set(studies[inner_train])),
                       'validation_studies': sorted(set(studies[inner_val]))})
    y = target[train]
    mask = np.isfinite(y)
    y, calibration_scores = y[mask].astype(int), scores[mask]
    if not np.isfinite(calibration_scores).all():
        raise ValueError('Incomplete inner OOF')
    if len(np.unique(y)) == 1:
        platt = float(y[0])
    else:
        platt = LogisticRegression(C=1., max_iter=2000, random_state=seed)
        platt.fit(calibration_scores.reshape(-1, 1), y)
    head = fit_head(kind, geometry, embedding, target, train)
    raw_test = predict_head(head, geometry, embedding, test)
    raw_cut = grouped_threshold(y, calibration_scores, studies[train][mask], bags, seed)
    cal_cut = grouped_threshold(y, calibrated(platt, calibration_scores), studies[train][mask], bags, seed)
    return {'raw': raw_test, 'platt': calibrated(platt, raw_test),
            'raw_threshold': raw_cut, 'platt_threshold': cal_cut,
            'head': head, 'calibrator': platt,
            'audit': {'training_studies': sorted(set(studies[train])),
                      'test_studies': sorted(set(studies[test])), 'inner_splits': splits,
                      'labelled_training_rows': len(y), 'training_positives': int(y.sum()),
                      'raw_threshold': raw_cut, 'platt_threshold': cal_cut,
                      'calibration_slope': None if isinstance(platt, float) else float(platt.coef_[0, 0])}}


def run(args):
    from gpu_research.common import dump, fingerprint, load_feature_bundle
    if args.output.exists() or args.bags < 0:
        raise ValueError('Choose a new output directory and nonnegative bags')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    validate_outer(labels, baseline)
    if set(baseline['index']) != set(range(len(labels))):
        raise ValueError('Incomplete baseline indices')
    source = inspect_sources([('organiser', args.labels, args.dataset)])
    source_id = fingerprint(source['entries'])
    features, embeddings, _ = build_table(args.dataset, labels, HERE / '.cache', source_id)
    metadata = None
    if args.embeddings:
        embeddings, metadata = load_feature_bundle(args.embeddings, labels, source_id, sha256(args.labels))
    features = [dict(f) for f in features]
    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            features[row.index]['spine_abs_angle_deg'] = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
    studies = labels.study_key.astype(str).to_numpy()
    anatomy = np.array([group_of(r) for r in labels.region])
    candidates = {name: baseline.copy() for name in ('raw', 'platt')}
    args.output.mkdir(parents=True)
    dump(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    audit = []
    for fold in sorted(baseline.fold.unique()):
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        train = np.flatnonzero(~np.isin(studies, list(held)))
        by_criterion = {}
        for group, criteria in CRITERIA.items():
            tr = train[anatomy[train] == group]
            te = baseline.index[(baseline.fold == fold) & baseline.predicted_region.map(group_of).eq(group)].to_numpy()
            for criterion, columns in criteria.items():
                target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                result = train_criterion(_matrix(features, columns), embeddings, target, studies,
                                         tr, te, args.head, args.bags, args.seed)
                by_criterion[criterion] = result
                audit.append({'fold': int(fold), 'criterion': criterion, **result['audit']})
                joblib.dump({'head': result['head'], 'calibrator': result['calibrator'],
                             'raw_threshold': result['raw_threshold'], 'platt_threshold': result['platt_threshold'],
                             'geometry_columns': columns, 'training_studies': result['audit']['training_studies'],
                             'clinical_validation': False}, args.output / f'fold-{fold}-{criterion}.joblib')
            for variant, table in candidates.items():
                cuts = {c: by_criterion[c][f'{variant}_threshold'] for c in criteria}
                for position, row_index in enumerate(te):
                    scores = {c: float(by_criterion[c][variant][position]) for c in criteria}
                    probability = float(1 - np.prod([1 - p for p in scores.values()]))
                    decision = decide(group, probability, scores, features[row_index],
                                      float(np.nextafter(1., np.inf)), cuts)
                    updates = {'quality_score': probability, 'quality_pred': decision['quality'],
                               'violation_type': official_violation_type(decision['violations']),
                               'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                               'criterion_scores': json.dumps(scores, sort_keys=True),
                               'criterion_thresholds': json.dumps(cuts, sort_keys=True),
                               'quality_threshold': decision['quality_threshold'],
                               'decision_version': 'research-strict-nested-calibration-1',
                               'decision_reason': decision['decision_reason']}
                    for key, value in updates.items():
                        table.at[row_index, key] = value
        dump(args.output / 'split_audit.json', audit)
        print(f'Completed outer fold {fold}', flush=True)
    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=2000),
                         'screen': compare(args.baseline, path, args.labels)}
        print(name, reports[name]['metrics']['overall_quality'], flush=True)
    dump(args.output / 'evaluation.json', {
        'protocol': __doc__, 'variants': reports, 'source_fingerprint': source_id,
        'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
        'code_sha256': sha256(Path(__file__)), 'external_embedding_metadata': metadata,
        'head': args.head, 'bags': args.bags, 'seed': args.seed,
        'clinical_validation': False, 'automatic_promotions': 0,
        'limitations': ['Previously inspected cohort; independent patient-level test still missing.',
                        'Historical routing and angles reused; conditional QC only.',
                        'Per-criterion calibration does not guarantee calibrated noisy-OR quality.',
                        'Calibrators differ by fold: pooled AUC and AP can change.',
                        'Exploratory comparison; saved fold heads are research artifacts, not release weights.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=HERE / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', choices=('baseline_ensemble', 'geometry_lr', 'embedding_lr_001'), default='baseline_ensemble')
    parser.add_argument('--bags', type=int, default=100)
    parser.add_argument('--seed', type=int, default=17)
    with threadpool_limits(limits=4):
        run(parser.parse_args())
