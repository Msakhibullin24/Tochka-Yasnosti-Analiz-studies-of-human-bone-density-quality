"""Fold-local PCA and shallow gradient boosting for global DXA quality.

Conditional historical routing/canonicalization; repeatedly inspected 249-image
cohort, not independent patient validation. Four fixed configurations selected
only by outer-training inner OOF AUC. Runtime models are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.model import _matrix
from evaluate_organizer_dataset import evaluate, sha256
from experiments.nested_criterion_upgrade import validate_outer, positive_probability
from experiments.nested_model_council import FEATURES, apply_predictions, quality_cut
from experiments.three_iteration_upgrade import dump, require_validation_bank
from experiments.verify_metric_candidate import compare
from gpu_research.common import load_feature_bundle

SOURCE = Path(__file__).read_bytes()
SOURCE_SHA = hashlib.sha256(SOURCE).hexdigest()
CONFIGURATIONS = ((4, 2), (4, 3), (8, 2), (8, 3))


def fit_model(embeddings, geometry, regions, targets, train, configuration):
    """Every fitted transform, including unsupervised PCA, sees train only."""
    components, depth = configuration
    y = targets[train]
    if not np.isfinite(y).all() or len(np.unique(y)) != 2:
        raise ValueError('Training quality requires both known classes')
    projections, blocks = [], []
    for features in embeddings:
        projector = PCA(n_components=components, svd_solver='randomized', random_state=17)
        blocks.append(projector.fit_transform(features[train]))
        projections.append(projector)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    g = scaler.fit_transform(imputer.fit_transform(geometry[train]))
    geometry_projector = PCA(n_components=8, svd_solver='randomized', random_state=17)
    blocks.extend((geometry_projector.fit_transform(g), regions[train, None]))
    head = HistGradientBoostingClassifier(max_depth=depth, max_iter=100,
        min_samples_leaf=10, l2_regularization=10., learning_rate=.05,
        random_state=17, early_stopping=False)
    head.fit(np.column_stack(blocks), y.astype(int))
    return {'projections': projections, 'imputer': imputer, 'scaler': scaler,
            'geometry_projector': geometry_projector, 'head': head,
            'configuration': configuration}


def predict_model(model, embeddings, geometry, regions, query):
    blocks = [projector.transform(x[query]) for projector, x in
              zip(model['projections'], embeddings, strict=True)]
    g = model['scaler'].transform(model['imputer'].transform(geometry[query]))
    blocks.extend((model['geometry_projector'].transform(g), regions[query, None]))
    result = positive_probability(model['head'], np.column_stack(blocks))
    if not np.isfinite(result).all():
        raise ValueError('Invalid quality probabilities')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--labels', type=Path, default=ROOT/'labels/image_labels.csv')
    p.add_argument('--reference', type=Path, default=Path(
        'data/gpu-runs/anatomy-upgrade-2026-10-15/quality-gated-own-roi/candidate_oof.csv'))
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    manifest = json.loads((args.bank/'manifest.json').read_text())
    require_validation_bank(manifest)
    if (sha256(args.bank/'bank.joblib') != manifest['bank_sha256'] or
            sha256(args.labels) != manifest['labels_sha256']):
        raise ValueError('Bank/label identity differs')
    bank = joblib.load(args.bank/'bank.joblib')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = bank['baseline'].sort_values('index').reset_index(drop=True)
    np.testing.assert_array_equal(baseline['index'], np.arange(len(labels)))
    validate_outer(labels, baseline)
    reference = pd.read_csv(args.reference).sort_values('index').reset_index(drop=True)
    identity = ['index', 'repeat', 'fold', 'study', 'source_path', 'quality_true']
    if not baseline[identity].equals(reference[identity]):
        raise ValueError('Reference identity differs')
    embeddings, feature_audit = [], []
    for path in map(Path, FEATURES[:4]):
        x, meta = load_feature_bundle(path, labels, manifest['source_fingerprint'], manifest['labels_sha256'])
        embeddings.append(x)
        feature_audit.append({'path': str(path), 'sha256': sha256(path),
            'metadata_sha256': sha256(path.with_suffix('.json')), 'shape': list(x.shape),
            'encoder_finetuned': meta['encoder_finetuned'], 'preprocessing': meta.get('preprocessing')})
    columns = sorted({name for state in bank['folds'] for name in state['columns']['quality']})
    geometry = _matrix(bank['features'], columns)
    regions = baseline.predicted_region.ne('spine').to_numpy(int)
    studies = labels.study_key.astype(str).to_numpy()
    targets = labels.quality_class.to_numpy(int)
    angles = np.array([f.get('spine_abs_angle_deg', np.nan) for f in bank['features']], float)
    forced = (regions == 0) & np.isfinite(angles) & (abs(angles) > 5.)
    args.output.mkdir(parents=True)
    (args.output/'executed_pca_boost_quality.py').write_bytes(SOURCE)
    protocol = {'protocol': __doc__, 'primary': 'selected_by_inner_auc',
        'configurations': CONFIGURATIONS, 'code_sha256': SOURCE_SHA,
        'bank_sha256': manifest['bank_sha256'], 'labels_sha256': manifest['labels_sha256'],
        'reference_sha256': sha256(args.reference), 'source_fingerprint': manifest['source_fingerprint'],
        'feature_audit': feature_audit, 'geometry_columns': columns,
        'geometry_pca_components': 8, 'external_study_folds': 5, 'inner_study_folds': 4,
        'clinical_validation': False, 'deployment_ready': False,
        'score_calibration_established': False, 'automatic_promotions': 0,
        'dependency_code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
            ('nested_model_council.py', 'three_iteration_upgrade.py', 'quality_gated_decoder.py')}}
    dump(args.output/'protocol.json', protocol)
    table = baseline.copy()
    saved, audit = [], []
    with threadpool_limits(limits=2):
        for fold in sorted(baseline.fold.unique()):
            test = np.flatnonzero(baseline.fold.to_numpy() == fold)
            held = set(studies[test])
            train = np.flatnonzero(~np.isin(studies, list(held)))
            safe = np.full(len(targets), np.nan)
            safe[train] = targets[train]
            predictions, scores, split_audit = [], [], []
            splits = list(GroupKFold(4).split(train, groups=studies[train]))
            for it, iv in splits:
                tr, va = train[it], train[iv]
                if set(studies[tr]) & (set(studies[va]) | held) or set(studies[va]) & held:
                    raise ValueError('Study leakage')
                split_audit.append({'training_studies': sorted(set(studies[tr])),
                                    'validation_studies': sorted(set(studies[va]))})
            for configuration in CONFIGURATIONS:
                inner = np.full(len(train), np.nan)
                for it, iv in splits:
                    model = fit_model(embeddings, geometry, regions, safe, train[it], configuration)
                    inner[iv] = predict_model(model, embeddings, geometry, regions, train[iv])
                if not np.isfinite(inner).all():
                    raise ValueError('Incomplete inner OOF')
                predictions.append(inner)
                scores.append(float(roc_auc_score(safe[train], inner)))
                print(json.dumps({'fold': int(fold), 'configuration': configuration,
                                  'inner_auc': scores[-1]}), flush=True)
            best = max(range(len(CONFIGURATIONS)), key=lambda i: (scores[i], -i))
            cut = quality_cut(safe[train].astype(int), predictions[best], forced[train])
            model = fit_model(embeddings, geometry, regions, safe, train, CONFIGURATIONS[best])
            quality = predict_model(model, embeddings, geometry, regions, test)
            apply_predictions(table, bank, fold, quality, {0: cut, 1: cut})
            table.loc[test, 'decision_version'] = 'research-pca-boost-selected-inner-1'
            saved.append({'fold': int(fold), 'train': train, 'test': test, 'model': model,
                          'quality_cut': cut, 'inner_oof': predictions[best], 'outer_quality': quality})
            audit.append({'fold': int(fold), 'training_studies': sorted(set(studies[train])),
                'test_studies': sorted(held), 'inner_splits': split_audit,
                'configuration_inner_auc': scores, 'selected_configuration': CONFIGURATIONS[best],
                'quality_cut': cut, 'outer_labels_used_for_selection': False})
            dump(args.output/'selection_audit.json', audit)
            joblib.dump({'folds': saved, 'geometry_columns': columns}, args.output/'models.joblib', compress=3)
    output = args.output/'candidate_oof.csv'
    table.to_csv(output, index=False)
    metrics = evaluate(args.labels, output, repeats=args.bootstrap)
    screen = compare(args.reference, output, args.labels)
    result = {**protocol, 'metrics': metrics, 'screen': screen, 'selection_audit': audit}
    dump(args.output/'evaluation.json', result)
    print(json.dumps({'primary': 'selected_by_inner_auc', 'quality': metrics['overall_quality'],
                      'criterion_macro_f1': metrics['criterion_macro_f1'], 'screen': screen}), flush=True)


if __name__ == '__main__':
    main()
