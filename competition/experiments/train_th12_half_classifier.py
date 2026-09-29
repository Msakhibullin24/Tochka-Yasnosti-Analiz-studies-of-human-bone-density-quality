"""Source-only direct half-height classifier with parent-group conformal abstention."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_th12_visibility import FEATURE_CONTRACT
from experiments.train_th12_visibility import features
from dxaqc.embedding import WEIGHTS_SHA256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_threshold(scores, labels, groups, alpha=.1):
    """One calibration score per source image: its worst variant."""
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    if (scores.ndim != 1 or scores.shape != labels.shape or scores.shape != groups.shape or
            not len(scores) or not 0 < alpha < 1 or not np.isfinite(scores).all() or
            ((scores < 0) | (scores > 1)).any() or not np.isin(labels, [0, 1]).all()):
        raise ValueError('Invalid grouped calibration inputs')
    errors = np.where(labels == 1, 1-scores, scores)
    maxima = np.array([errors[groups == group].max() for group in np.unique(groups)])
    rank = int(np.ceil((len(maxima)+1)*(1-alpha)))
    if rank > len(maxima):
        raise ValueError('Too few calibration parents')
    return float(np.sort(maxima)[rank-1])


def candidate_rule(probability, threshold):
    if not (np.isfinite(probability) and 0 <= probability <= 1 and
            np.isfinite(threshold) and 0 <= threshold <= 1):
        raise ValueError('Invalid half-body probability or threshold')
    include_less = probability <= threshold
    include_half = 1-probability <= threshold
    if include_less and include_half:
        return 'undetermined'
    if include_less:
        return 'candidate_less_than_half_height'
    if include_half:
        return 'candidate_at_least_half_height'
    return 'undetermined'  # An empty prediction set cannot safely decide.


def assess(bundle, pixels):
    if (bundle.get('schema_version') != 2 or bundle.get('scope') != 'th12_source_half_height' or
            bundle.get('status') != 'research_only' or bundle.get('feature_contract') != FEATURE_CONTRACT or
            bundle.get('encoder_sha256') != WEIGHTS_SHA256 or bundle.get('clinical_validation') is not False):
        raise ValueError('Incompatible source-only half-height bundle')
    scores = np.asarray(bundle['model'].predict_proba(features(pixels)[None]))
    if scores.shape != (1, 2) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid half-height model probabilities')
    probability = float(scores[0, 1])
    return {'positive_probability': probability,
            'candidate_rule': candidate_rule(probability, bundle['calibration_threshold']),
            'clinical_verdict': 'undetermined', 'clinical_validation': False,
            'scope': 'Source-calibrated synthetic ordinary-XR crop classifier; DXA accuracy is not established'}


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    manifest_path = args.dataset/'dataset.json'
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((args.regression/'evaluation.json').read_text())
    if (manifest.get('version') != 1 or sha(manifest_path) != report['dataset_sha256'] or
            manifest['derived_images'] != len(manifest['cases'])):
        raise ValueError('Source training dataset differs from verified feature cache')
    rows = manifest['cases']
    x = np.load(args.regression/'features.npy', mmap_mode='r')
    if x.shape != (len(rows), 1792) or not np.isfinite(x).all():
        raise ValueError('Invalid or mismatched cached source features')
    parents = np.array([r['parent_group'] for r in rows])
    train_parents = sorted(set(r['parent_group'] for r in rows if r['split'] == 'train'))
    test_parents = sorted(set(r['parent_group'] for r in rows if r['split'] == 'test'))
    if set(train_parents) & set(test_parents) or len(train_parents) != 340 or len(test_parents) != 85:
        raise ValueError('Unexpected original parent holdout')
    calibration = set(np.random.default_rng(17).choice(train_parents, len(train_parents)//5, replace=False))
    splits = np.array(['test' if r['split']=='test' else 'calibration' if r['parent_group'] in calibration else 'train' for r in rows])
    y = np.array([int(r['reference']['visible_vertical_extent_fraction'] >= .5) for r in rows])
    model = ExtraTreesClassifier(n_estimators=128, min_samples_leaf=2, max_features='sqrt',
                                 class_weight='balanced', random_state=17, n_jobs=4)
    model.fit(x[splits=='train'], y[splits=='train'])
    if list(model.classes_) != [0, 1]:
        raise ValueError('Half-body classes absent from training')
    probabilities = model.predict_proba(x)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite classifier output')
    cal, test = splits=='calibration', splits=='test'
    threshold = calibration_threshold(probabilities[cal], y[cal], parents[cal])
    cases = [{'image': r['image'], 'parent_group': r['parent_group'], 'reference': int(y[i]),
              'positive_probability': float(probabilities[i]),
              'candidate_rule': candidate_rule(float(probabilities[i]), threshold)}
             for i, r in enumerate(rows) if test[i]]
    determined = [c for c in cases if c['candidate_rule'] != 'undetermined']
    wrong = sum((c['candidate_rule'] == 'candidate_at_least_half_height') != bool(c['reference']) for c in determined)
    covered = [c['candidate_rule'] == 'undetermined' or
               (c['candidate_rule'] == 'candidate_at_least_half_height') == bool(c['reference']) for c in cases]
    parent_covered = [all(ok for c, ok in zip(cases, covered) if c['parent_group'] == group) for group in test_parents]
    args.output.mkdir(parents=True)
    bundle = {'schema_version': 2, 'scope': 'th12_source_half_height', 'feature_contract': FEATURE_CONTRACT,
              'encoder_sha256': WEIGHTS_SHA256, 'model': model, 'calibration_threshold': threshold,
              'clinical_validation': False, 'status': 'research_only'}
    joblib.dump(bundle, args.output/'candidate.joblib')
    output = {'protocol': 'Fixed ExtraTrees128 leaf2 sqrt balanced seed17; original source parents 272 train/68 calibration/85 test; group-max 90% conformal threshold.',
              'training_views': int((splits=='train').sum()), 'calibration_views': int(cal.sum()),
              'test_views': int(test.sum()), 'training_parents': 272, 'calibration_parents': 68,
              'test_parents': 85, 'threshold': threshold, 'determined_test_views': len(determined),
              'wrong_determined_test_views': wrong, 'test_view_coverage': float(np.mean(covered)),
              'test_parent_all_views_coverage': float(np.mean(parent_covered)),
              'raw_test_accuracy': float(np.mean((probabilities[test] >= .5) == y[test])),
              'cases': cases, 'model_sha256': sha(args.output/'candidate.joblib'),
              'dataset_sha256': sha(manifest_path), 'feature_cache_sha256': sha(args.regression/'features.npy'),
              'code_sha256': sha(Path(__file__)), 'clinical_validation': False,
              'release_modified': False, 'requirements_complete': False,
              'limitations': ['Derived ordinary radiograph crops; no independently reviewed full-body DXA references.',
                              'Previously examined source holdout; exploratory model comparison, not a fresh test.',
                              'Source groups are original images, not verified patient identities.',
                              'Source conformal coverage does not transfer to DXA or establish a clinical half-body rule.']}
    (args.output/'evaluation.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('threshold','determined_test_views','wrong_determined_test_views','raw_test_accuracy','test_parent_all_views_coverage')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'regression', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
