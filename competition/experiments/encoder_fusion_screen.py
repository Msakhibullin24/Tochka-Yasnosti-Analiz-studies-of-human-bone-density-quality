"""Encoder-fusion screen: combine several frozen embeddings and the geometry block.

Same outer folds, same inner split, same routing and same decision layer as
nested_criterion_upgrade.py, so the numbers are directly comparable with the
reference run. New machinery: PCA/whitening fitted inside the training folds
only, and a multi-encoder concatenation. No automatic release promotion.
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, SEED, _lr, _matrix, _rf, best_f1_threshold, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402
from experiments.nested_criterion_upgrade import positive_probability, select_head, validate_outer  # noqa: E402

# Anchors kept verbatim so the run is comparable with the reference protocol.
ANCHORS = ('geometry_rf', 'geometry_lr', 'resnet18_lr_001')
FUSIONS = ('fused_pca32_lr', 'fused_pca64_lr', 'geom_fused_pca32_lr',
           'fused_pca32_lda', 'fused_pca32_svc')


def variant_names(bundle_keys):
    """One linear-probe variant per supplied bundle, plus the fusion block."""
    return (*ANCHORS, *(f'{k}_lr_001' for k in bundle_keys), *FUSIONS)


class FoldPCA:
    """Whitened PCA refitted on every training split; never sees validation data."""

    def __init__(self, components, seed=SEED):
        self.components, self.seed = components, seed

    def fit(self, values, n_train):
        k = int(max(2, min(self.components, n_train - 1, values.shape[1])))
        self.pca_ = PCA(n_components=k, whiten=True, random_state=self.seed).fit(values)
        return self

    def transform(self, values):
        return self.pca_.transform(values)

    @property
    def n_components(self):
        return self.pca_.n_components_


def _estimator(kind):
    if kind == 'geometry_rf':
        return _rf()
    if kind == 'geometry_lr':
        return _lr(.3)
    if kind.endswith('_lr_001'):
        return _lr(.01)
    if kind.endswith('_lda'):
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    if kind.endswith('_svc'):
        return make_pipeline(StandardScaler(), SVC(C=1., gamma='scale', probability=True, random_state=SEED))
    if kind.endswith('_pca32_lr') or kind.endswith('_pca64_lr'):
        return _lr(.1)
    raise KeyError(kind)


def fit_head(kind, geometry, embedding, target, train):
    known = train[np.isfinite(target[train])]
    if not len(known):
        raise ValueError('Criterion has no training labels')
    y = target[known].astype(int)
    model = _estimator(kind)
    fused = kind.startswith('fused_pca') or kind.startswith('geom_fused')
    if fused:
        blocks = [geometry[known]] if kind.startswith('geom_fused') else []
        blocks.append(embedding[known])
        stacked = np.column_stack(blocks)
        reducer = FoldPCA(32 if 'pca32' in kind else 64).fit(stacked, len(known))
        if len(np.unique(y)) < 2:
            dummy = DummyClassifier(strategy='constant', constant=int(y[0]))
            dummy.fit(np.zeros((len(known), 1)), y)
            return kind, dummy, reducer, True
        model.fit(reducer.transform(stacked), y)
        return kind, model, reducer, True
    if len(np.unique(y)) < 2:
        dummy = DummyClassifier(strategy='constant', constant=int(y[0]))
        dummy.fit(np.zeros((len(known), 1)), y)
        return kind, dummy, None, False
    values = geometry if kind.startswith('geometry') else embedding
    model.fit(values[known], y)
    return kind, model, None, False


def apply_head(head, geometry, embedding, rows):
    kind, model, reducer, _ = head
    if reducer is not None:
        blocks = []
        if kind.startswith('geom_fused'):
            blocks.append(geometry[rows])
        blocks.append(embedding[rows])
        values = reducer.transform(np.column_stack(blocks))
    else:
        values = geometry[rows] if kind.startswith('geometry') else embedding[rows]
    if isinstance(model, DummyClassifier):
        return np.zeros(len(rows))
    return positive_probability(model, values)


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
    features, resnet18, _ = build_table(args.dataset, labels, HERE / '.cache', fingerprint)

    from gpu_research.common import load_feature_bundle
    bundles = {}
    for name, path in args.bundle.items():
        values, meta = load_feature_bundle(path, labels, fingerprint, sha256(args.labels))
        bundles[name] = values
        print(f'{name}: {values.shape} from {Path(path).parent.name}', flush=True)
    variants = variant_names(list(bundles))
    stacks = {'fused': np.column_stack([resnet18] + list(bundles.values()))}
    if len(stacks['fused'][0]) != resnet18.shape[1] + sum(v.shape[1] for v in bundles.values()):
        raise ValueError('Embedding concatenation is inconsistent')

    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            angle = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
            features[row.index]['spine_abs_angle_deg'] = angle
    features = [dict(f) for f in features]
    groups = labels.study_key.astype(str).to_numpy()
    anatomy = np.array([group_of(r) for r in labels.region])
    args.output.mkdir(parents=True)
    pools = {'geometry_rf': None, 'geometry_lr': None, 'resnet18_lr_001': resnet18}
    pools.update({f'{k}_lr_001': v for k, v in bundles.items()})
    pools.update({k: stacks['fused'] for k in FUSIONS})
    candidates = {name: baseline.copy() for name in (*variants, 'adaptive')}
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
                inner = {n: np.full(len(selected), np.nan) for n in variants}
                split = StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 100)
                for itr, iva in split.split(selected, labels.quality_class.iloc[selected], groups[selected]):
                    if set(groups[selected[itr]]) & set(groups[selected[iva]]):
                        raise ValueError('Inner study leakage')
                    heads = {n: fit_head(n, geometry, pools[n], target, selected[itr]) for n in variants}
                    for n in variants:
                        inner[n][iva] = apply_head(heads[n], geometry, pools[n], selected[iva])
                chosen, chosen_threshold = select_head(target[selected], inner)
                heads = {n: fit_head(n, geometry, pools[n], target, selected) for n in variants}
                parts = {n: apply_head(h, geometry, pools[n], tests) for n, h in heads.items()}
                known = np.isfinite(target[selected])
                for n in variants:
                    scores.setdefault(n, {})[criterion] = parts[n]
                    thresholds.setdefault(n, {})[criterion] = best_f1_threshold(
                        target[selected][known].astype(int), inner[n][known])
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
                    probability = float(1 - np.prod([1 - p for p in criterion_scores.values()]))
                    decision = decide(group, probability, criterion_scores, features[index],
                                      float(np.nextafter(1., np.inf)), thresholds[name])
                    for key, value in {'quality_score': probability, 'quality_pred': decision['quality'],
                                       'violation_type': official_violation_type(decision['violations']),
                                       'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                                       'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                                       'criterion_thresholds': json.dumps(thresholds[name], sort_keys=True),
                                       'quality_threshold': decision['quality_threshold'],
                                       'decision_version': 'research-encoder-fusion-1',
                                       'decision_reason': decision['decision_reason']}.items():
                        candidates[name].at[row_index, key] = value
        print(f'Completed outer fold {fold}', flush=True)
    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=200),
                         'screen': compare(args.baseline, path, args.labels)}
        print(f'{name:24} {reports[name]["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)
    dump = {'protocol': __doc__, 'variants': reports, 'selection_audit': selection_audit,
            'embedding_dimensions': {'resnet18': int(resnet18.shape[1]),
                                     **{k: int(v.shape[1]) for k, v in bundles.items()},
                                     'fused': int(stacks['fused'].shape[1])},
            'source_fingerprint': fingerprint, 'labels_sha256': sha256(args.labels),
            'baseline_sha256': sha256(args.baseline), 'code_sha256': sha256(Path(__file__)),
            'clinical_validation': False, 'model_affects_decision': False,
            'limitations': ['Previously inspected internal studies; patient independence unverified.',
                             'Baseline routing and runtime axes reused; not raw-input end-to-end CV.',
                             'PCA refitted inside every training split; components limited by fold size.',
                             'Multiple exploratory comparisons; no automatic release promotion.']}
    (args.output / 'evaluation.json').write_text(json.dumps(dump, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', action='append', default=[], metavar='NAME=PATH',
                        help='Audited frozen features.npy, repeatable')
    args = parser.parse_args()
    bundles = {}
    for item in args.bundle:
        name, sep, path = item.partition('=')
        if not sep or not path:
            parser.error(f'--bundle expects NAME=PATH, got {item!r}')
        if name in bundles:
            parser.error(f'duplicate bundle name {name!r}')
        bundles[name] = Path(path)
    if not bundles:
        parser.error('at least one --bundle NAME=PATH is required')
    args.bundle = bundles
    with threadpool_limits(limits=4):
        run(args)
