"""Three controlled QC iterations on outer-study-exclusive score generation.

1: fixed geometry/image ensembles and an independently trained quality score.
2: joint typed-decoder thresholds, selected on inner OOF only.
3: regularized quality stacking and quality-conditioned criterion scores.

Historical routing and physical axis measurements are frozen. This evaluates
conditional QC on previously inspected studies, not the complete anatomy path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.decision import decide
from dxaqc.model import CRITERIA, _matrix, _rf, _lr, group_of, official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from experiments.nested_criterion_upgrade import validate_outer, positive_probability
from experiments.strict_nested_calibration import threshold
from experiments.verify_metric_candidate import compare
from source_integrity import inspect_sources
from train import build_table


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def f1(y, predicted):
    tp = np.sum((y == 1) & predicted)
    return float(2 * tp / max(1, int(np.sum(y == 1) + np.sum(predicted))))


def fit_family(geometry, embeddings, target, train):
    """No target outside train is read, including for one-class fallbacks."""
    known = train[np.isfinite(target[train])]
    if not len(known):
        raise ValueError('No training labels')
    y = target[known].astype(int)
    values = [geometry, geometry, *embeddings]
    models = []
    for i, x in enumerate(values):
        if len(np.unique(y)) == 1:
            model = DummyClassifier(strategy='constant', constant=int(y[0]))
            model.fit(np.zeros((len(y), 1)), y)
        else:
            model = _rf() if i == 0 else _lr(.3 if i == 1 else .01)
            model.fit(x[known], y)
        models.append(model)
    return models


def predict_family(models, geometry, embeddings, selected):
    values = [geometry, geometry, *embeddings]
    probabilities = []
    for model, x in zip(models, values, strict=True):
        query = np.zeros((len(selected), 1)) if isinstance(model, DummyClassifier) else x[selected]
        probabilities.append(positive_probability(model, query))
    # Geometry and imaging each carry half the weight, irrespective of count.
    return .5 * np.mean(probabilities[:2], axis=0) + .5 * np.mean(probabilities[2:], axis=0)


def build_bank(args):
    from gpu_research.common import fingerprint, load_feature_bundle
    if args.output.exists():
        raise ValueError('Choose a new bank directory')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    validate_outer(labels, baseline)
    source = inspect_sources([('organiser', args.labels, args.dataset)])
    source_id = fingerprint(source['entries'])
    features, _, _ = build_table(args.dataset, labels, ROOT / '.cache', source_id)
    features = [dict(f) for f in features]
    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            features[row.index]['spine_abs_angle_deg'] = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
    embeddings, embedding_audit = [], []
    for path in args.embeddings:
        values, metadata = load_feature_bundle(path, labels, source_id, sha256(args.labels))
        embeddings.append(values)
        embedding_audit.append({'path': str(path), 'sha256': metadata['features_sha256'],
                                'shape': list(values.shape), 'encoder_finetuned': False})
    if not embeddings:
        raise ValueError('At least one audited external frozen encoder required')
    args.output.mkdir(parents=True)
    studies = labels.study_key.astype(str).to_numpy()
    anatomy = labels.region.map(group_of).to_numpy()
    folds, audit = [], []
    for fold in sorted(baseline.fold.unique()):
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        for group, criteria in CRITERIA.items():
            train = np.flatnonzero(~np.isin(studies, list(held)) & (anatomy == group))
            test = np.flatnonzero((baseline.fold.to_numpy() == fold) & baseline.predicted_region.map(group_of).eq(group).to_numpy())
            if not len(test):
                continue
            codes = list(criteria)
            columns = {c: criteria[c] for c in codes}
            columns['quality'] = sorted(set(sum(list(criteria.values()), [])))
            targets = {c: pd.to_numeric(labels[c], errors='coerce').to_numpy(float) for c in codes}
            targets['quality'] = labels.quality_class.to_numpy(float)
            matrices = {c: _matrix(features, names) for c, names in columns.items()}
            inner = {c: np.full(len(train), np.nan) for c in targets}
            split_audit = []
            for itr, iva in GroupKFold(4).split(train, groups=studies[train]):
                tr, va = train[itr], train[iva]
                if set(studies[tr]) & (set(studies[va]) | held) or set(studies[va]) & held:
                    raise ValueError('Study leakage')
                for code, target in targets.items():
                    heads = fit_family(matrices[code], embeddings, target, tr)
                    inner[code][iva] = predict_family(heads, matrices[code], embeddings, va)
                split_audit.append({'training_studies': sorted(set(studies[tr])),
                                    'validation_studies': sorted(set(studies[va]))})
            heads, outer = {}, {}
            for code, target in targets.items():
                if not np.isfinite(inner[code]).all():
                    raise ValueError('Incomplete inner OOF')
                heads[code] = fit_family(matrices[code], embeddings, target, train)
                outer[code] = predict_family(heads[code], matrices[code], embeddings, test)
            folds.append({'fold': int(fold), 'group': group, 'train': train, 'test': test,
                          'codes': codes, 'columns': columns, 'heads': heads,
                          'inner': inner, 'outer': outer,
                          'criterion_truth': np.column_stack([targets[c][train] for c in codes]),
                          'quality_truth': targets['quality'][train].astype(int),
                          'angles': np.array([features[i].get('spine_abs_angle_deg', np.nan)
                                              if features[i].get('spine_abs_angle_deg') is not None else np.nan
                                              for i in train], float)})
            audit.append({'fold': int(fold), 'group': group, 'training_studies': sorted(set(studies[train])),
                          'test_studies': sorted(held), 'inner_splits': split_audit})
        dump(args.output / 'split_audit.json', audit)
        print(f'Bank outer fold {fold} complete', flush=True)
    joblib.dump({'schema_version': 1, 'baseline': baseline, 'features': features, 'folds': folds,
                 'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline)},
                args.output / 'bank.joblib', compress=3)
    dump(args.output / 'manifest.json', {'schema_version': 1, 'bank_sha256': sha256(args.output / 'bank.joblib'),
         'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
         'source_fingerprint': source_id, 'embeddings': embedding_audit, 'code_sha256': sha256(Path(__file__)),
         'clinical_validation': False, 'automatic_promotions': 0})


def decisions(scores, cuts, codes, angles):
    predicted = scores >= np.asarray(cuts)
    if 'spine_axis' in codes:
        predicted[:, codes.index('spine_axis')] = np.isfinite(angles) & (np.abs(angles) > 5.)
    return predicted


def utility(target, criterion_truth, predicted):
    quality = f1(target, predicted.any(axis=1))
    individual = []
    for j in range(predicted.shape[1]):
        known = np.isfinite(criterion_truth[:, j])
        individual.append(f1(criterion_truth[known, j], predicted[known, j]))
    # Fixed research regularizer, not a claimed organizer scoring formula.
    return quality + .1 * float(np.mean(individual)), quality


def joint_thresholds(target, truth, scores, codes, angles, initial):
    cuts = np.asarray(initial, float).copy()
    for _ in range(3):
        changed = False
        for j, code in enumerate(codes):
            if code == 'spine_axis':
                continue  # Physical >5 degree rule is never optimized against labels.
            best = (*utility(target, truth, decisions(scores, cuts, codes, angles)), float(cuts[j]))
            chosen = cuts[j]
            choices = np.unique(np.r_[0., np.quantile(scores[:, j], np.linspace(0, 1, 31)), cuts[j], np.nextafter(1., np.inf)])
            for cut in choices:
                proposal = cuts.copy()
                proposal[j] = cut
                ranking = (*utility(target, truth, decisions(scores, proposal, codes, angles)), float(cut))
                if ranking > best:
                    best, chosen = ranking, cut
            changed |= chosen != cuts[j]
            cuts[j] = chosen
        if not changed:
            break
    return cuts


def quality_inputs(criteria, direct):
    return np.column_stack([direct, 1 - np.prod(1 - criteria, axis=1),
                            criteria.max(axis=1), criteria.mean(axis=1)])


def fit_decoder(state, iteration):
    codes, target = state['codes'], state['quality_truth']
    truth = state['criterion_truth']
    raw = np.column_stack([state['inner'][c] for c in codes])
    direct = state['inner']['quality']
    alpha = max((0., .5, 1.), key=lambda a: (
        roc_auc_score(target, a * direct + (1 - a) * (1 - np.prod(1 - raw, axis=1))), -abs(a - .5)))
    calibrator, exponent = None, 0.
    if iteration == 3:
        calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, random_state=17))
        calibrator.fit(quality_inputs(raw, direct), target)
        q = calibrator.predict_proba(quality_inputs(raw, direct))[:, 1]
    else:
        q = alpha * direct + (1 - alpha) * (1 - np.prod(1 - raw, axis=1))
    best, result = None, None
    for power in ((0., .5, 1.) if iteration == 3 else (0.,)):
        scores = raw * q[:, None] ** power
        initial = np.array([threshold(truth[np.isfinite(truth[:, j]), j].astype(int),
                                      scores[np.isfinite(truth[:, j]), j]) for j in range(len(codes))])
        cuts = joint_thresholds(target, truth, scores, codes, state['angles'], initial) if iteration >= 2 else initial
        rank = (*utility(target, truth, decisions(scores, cuts, codes, state['angles'])), -power)
        if best is None or rank > best:
            best, exponent, result = rank, power, cuts
    return {'alpha': alpha, 'calibrator': calibrator, 'exponent': exponent,
            'thresholds': dict(zip(codes, result)), 'inner_utility': best[0], 'inner_quality_f1': best[1]}


def decode_scores(spec, codes, raw, direct):
    q = (spec['calibrator'].predict_proba(quality_inputs(raw, direct))[:, 1]
         if spec['calibrator'] is not None else
         spec['alpha'] * direct + (1 - spec['alpha']) * (1 - np.prod(1 - raw, axis=1)))
    return raw * q[:, None] ** spec['exponent'], q


def run_iteration(args):
    if args.output.exists():
        raise ValueError('Choose a new iteration directory')
    manifest = json.loads((args.bank / 'manifest.json').read_text())
    if (manifest['bank_sha256'] != sha256(args.bank / 'bank.joblib') or
            manifest['labels_sha256'] != sha256(args.labels) or manifest['baseline_sha256'] != sha256(args.baseline)):
        raise ValueError('Bank, labels or baseline checksum differs')
    bank = joblib.load(args.bank / 'bank.joblib')
    args.output.mkdir(parents=True)
    table, audit, specs = bank['baseline'].copy(), [], []
    for state in bank['folds']:
        spec = fit_decoder(state, args.iteration)
        codes, test = state['codes'], state['test']
        raw = np.column_stack([state['outer'][c] for c in codes])
        scores, quality = decode_scores(spec, codes, raw, state['outer']['quality'])
        # A singleton and the complete test batch must have identical score transforms.
        for i in range(len(test)):
            one_scores, one_quality = decode_scores(spec, codes, raw[i:i+1], state['outer']['quality'][i:i+1])
            np.testing.assert_allclose(one_scores[0], scores[i], rtol=0, atol=1e-12)
            np.testing.assert_allclose(one_quality[0], quality[i], rtol=0, atol=1e-12)
        for pos, row_index in enumerate(test):
            criterion_scores = dict(zip(codes, map(float, scores[pos])))
            decision = decide(state['group'], float(quality[pos]), criterion_scores, bank['features'][row_index],
                              float(np.nextafter(1., np.inf)), spec['thresholds'])
            updates = {'quality_score': float(quality[pos]), 'quality_pred': decision['quality'],
                       'violation_type': official_violation_type(decision['violations']),
                       'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                       'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                       'criterion_thresholds': json.dumps(spec['thresholds'], sort_keys=True),
                       'quality_threshold': decision['quality_threshold'], 'decision_reason': decision['decision_reason'],
                       'decision_version': f'research-three-iteration-{args.iteration}'}
            for key, value in updates.items():
                table.at[row_index, key] = value
        audit.append({'fold': state['fold'], 'group': state['group'], 'alpha': spec['alpha'],
                      'quality_conditioning_exponent': spec['exponent'], 'thresholds': spec['thresholds'],
                      'inner_utility': spec['inner_utility'], 'inner_quality_f1': spec['inner_quality_f1']})
        specs.append({'fold': state['fold'], 'group': state['group'], 'decoder': spec})
    joblib.dump(specs, args.output / 'decoders.joblib')
    path = args.output / 'candidate_oof.csv'
    table.to_csv(path, index=False)
    metrics = evaluate(args.labels, path, repeats=2000)
    screen = compare(args.baseline, path, args.labels)
    dump(args.output / 'evaluation.json', {'iteration': args.iteration, 'metrics': metrics, 'screen': screen,
         'selection_audit': audit, 'bank_sha256': manifest['bank_sha256'],
         'code_sha256': sha256(Path(__file__)), 'clinical_validation': False, 'automatic_promotions': 0,
         'singleton_batch_score_check_passed': True, 'limitations': [
             'Previously inspected study groups, not independent patients or clinical validation.',
             'Frozen historical routing and measured angles; not full raw-input end-to-end CV.',
             'Joint threshold objective includes a fixed 0.1 criterion-F1 regularizer, not organizer weights.',
             'Noisy-OR and quality-conditioned criterion scores are research combinations.',
             'No anatomical completeness established by binary QC metrics.']})
    print(json.dumps({'iteration': args.iteration, 'metrics': metrics['overall_quality'],
                      'macro_f1': metrics['criterion_macro_f1'], 'screen_passed': screen['passes_internal_metric_screen']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bank', 'iteration'))
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    parser.add_argument('--embeddings', type=Path, nargs='+', default=[])
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--iteration', type=int, choices=(1, 2, 3))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'bank' and (args.dataset is None or not args.embeddings):
        parser.error('Bank requires --dataset and --embeddings')
    if args.command == 'iteration' and (args.bank is None or args.iteration is None):
        parser.error('Iteration requires --bank and --iteration')
    with threadpool_limits(limits=4):
        build_bank(args) if args.command == 'bank' else run_iteration(args)
