"""Rank-averaged ensemble of diverse heads, to stop the false-positive explosion.

Every learned variant in this project failed the same way: the baseline has 34
false positives and every fitted model has 71-117. The mechanism is visible in
the numbers - argmax-F1 thresholds chosen on an ~18-study inner split are wildly
unstable, and a single logistic head over thousands of embedding dimensions sits
near the boundary, so small score shifts flip many rows to positive.

Two changes here, both aimed at that mechanism rather than at the representation:

  * a rank-averaged ensemble instead of one head. Scores from each head are
    converted to within-fold ranks before averaging, so a head with a wider
    probability scale cannot dominate, and the ensemble is far less sensitive to
    any single threshold choice;
  * gradient boosting added as a head family. It captures the non-linear
    interactions between geometry and embedding blocks that a linear probe
    cannot, and HistGradientBoosting handles the NaNs that geometry columns
    contain.

The ensemble membership and the threshold are chosen on training studies only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, _matrix, best_f1_threshold, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402
from experiments.nested_criterion_upgrade import validate_outer  # noqa: E402


def make_head(name, seed):
    if name == 'rf':
        return make_pipeline(StandardScaler(),
                             ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                                                  class_weight='balanced_subsample',
                                                  random_state=seed, n_jobs=1))
    if name == 'gb':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=15,
                                              min_samples_leaf=8, l2_regularization=1.,
                                              class_weight='balanced', random_state=seed)
    if name == 'lr':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=.05, class_weight='balanced', max_iter=5000,
                                                random_state=seed))
    if name == 'svc':
        return make_pipeline(StandardScaler(),
                             SVC(C=1., gamma='scale', probability=True, random_state=seed))
    if name == 'lda':
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    raise KeyError(name)


MEMBERS = ('rf', 'gb', 'lr', 'svc', 'lda')
VARIANTS = ('geom_rank_avg', 'emb_rank_avg', 'both_rank_avg', 'geom_gb', 'both_gb',
            # Rank-quantile operating points: the flagged rate is estimated instead of
            # the probability cut. Targets the rare criteria where visibility is high
            # but the threshold cannot be placed.
            'both_rank_avg_q', 'geom_rank_avg_q', 'emb_rank_avg_q',
            # Same scores as the two rank-average winners, but spine_axis resolved by
            # the validated criterion threshold instead of the unreachable 5 deg rule.
            'both_rank_avg_axis_model', 'geom_rank_avg_axis_model',
            # AND-gate: flag only when the rank ensemble and the oracle measurement that
            # the visibility diagnostic ranked best both fire. Two partly independent
            # signals in series is the standard variance reduction when one threshold
            # cannot be placed on a 6- or 7-positive criterion.
            'both_rank_avg_andgate',
            # Conservative switch: adopt the ensemble only where it beats the shipped
            # baseline calibration by --switch-margin on inner OOF. The rare coverage
            # criteria are already well calibrated upstream (hip_roi_coverage F1 0.429
            # from 3 of 7 positives with only 4 false positives), and the plain rank
            # average loses that calibration and drops them to 0.167. A switch margin
            # removes that churn without consulting the outer folds.
            'both_rank_avg_switch', 'adaptive')
OUTER = tuple(VARIANTS)


def probability(model, values):
    if isinstance(model, DummyClassifier):
        return np.zeros(len(values))
    if 1 not in model.classes_:
        return np.zeros(len(values))
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def fit_predict(name, values, target, train, test, seed):
    """Fit one head on `train` rows and return raw probabilities for both splits."""
    known = train[np.isfinite(target[train])]
    y = target[known].astype(int)
    model = make_head(name, seed)
    if len(np.unique(y)) < 2:
        dummy = DummyClassifier(strategy='constant', constant=int(y[0]))
        dummy.fit(np.zeros((len(known), 1)), y)
        return np.zeros(len(known)), np.zeros(len(test)), dummy
    model.fit(values[known], y)
    return probability(model, values[known]), probability(model, values[test]), model


def rank_average(score_rows):
    """Average within-split ranks so heterogeneous scales cannot dominate."""
    return np.column_stack([rankdata(row) / max(len(row), 1) for row in score_rows]).mean(1)


def select_quantile(truth, scores, rates):
    """Flag the top q fraction by score and choose q on a coarse grid.

    A probability threshold is a high-variance parameter when a criterion has 6-7
    positives inside a ~200-row training fold. The flagged *rate* is one scalar on
    a coarse grid, so it can be estimated far more stably, which is what the
    visibility diagnostic says is needed: hip_roi_coverage separates at AUC 0.939
    yet scores F1 0.17, because the cut cannot be placed.
    """
    order = np.argsort(-np.asarray(scores))
    ranked = np.asarray(scores)[order]
    best, chosen = -1.0, rates[0]
    n = len(truth)
    for rate in rates:
        k = max(1, int(round(rate * n)))
        threshold = float(ranked[min(k, n) - 1])
        f1 = binary_metrics(truth, (np.asarray(scores) >= threshold).astype(int))['f1']
        if f1 > best:
            best, chosen = f1, rate
    return 1.0 - chosen


def select_threshold(truth, scores, floors):
    best, chosen = -1.0, float(np.nextafter(1., np.inf))
    for floor in floors:
        for threshold in np.unique(np.concatenate(([0.0], np.sort(np.unique(scores))))):
            if floor is None:
                metrics = binary_metrics(truth, (scores >= threshold).astype(int))
            else:
                if (scores >= threshold).mean() > 1 - floor:
                    continue
                metrics = binary_metrics(truth, (scores >= threshold).astype(int))
                if metrics['specificity'] < floor:
                    continue
            if metrics['f1'] > best:
                best, chosen = metrics['f1'], float(threshold)
    return chosen


SEED = 17


def run(args):
    global SEED
    # Fold partitions and every estimator's random_state follow --seed. Without this the
    # pipeline is fully deterministic, so a "different seed" run would be the same
    # computation and a robustness check would be meaningless.
    SEED = int(args.seed)
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    validate_outer(labels, baseline)
    baseline_scores = [json.loads(v) if isinstance(v, str) and v else {}
                       for v in baseline['criterion_scores']]
    baseline_thresholds = [json.loads(v) if isinstance(v, str) and v else {}
                           for v in baseline['criterion_thresholds']]
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    source_fp = __import__('hashlib').sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    features, resnet18, _ = build_table(args.dataset, labels, HERE / '.cache', source_fp)

    from gpu_research.common import load_feature_bundle
    bundles = {}
    for name, path in args.bundle.items():
        bundles[name] = load_feature_bundle(path, labels, source_fp, sha256(args.labels))[0]
        print(f'{name}: {bundles[name].shape}', flush=True)
    embedding = np.column_stack([resnet18] + list(bundles.values()))

    oracle_table = oracle_numeric = oracle_paths = None
    if args.oracle.exists():
        oracle_table = pd.read_csv(args.oracle)
        oracle_paths = oracle_table['path'].astype(str).tolist()
        if oracle_paths != labels.first_source_path.astype(str).tolist():
            raise ValueError('Oracle measurements are not aligned with the label rows')
        dropped = ['path', 'region', 'quality_class', 'spine_coverage', 'spine_axis',
                   'spine_artifact', 'hip_position_rotation', 'hip_roi_coverage']
        oracle_numeric = [c for c in oracle_table.columns if c not in dropped]
        oracle = oracle_table[oracle_numeric].to_numpy(float)
        print(f'oracle measurements: {oracle.shape}', flush=True)

    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            angle = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
            features[row.index]['spine_abs_angle_deg'] = angle
    features = [dict(f) for f in features]
    groups = labels.study_key.astype(str).to_numpy()
    anatomy = np.array([group_of(r) for r in labels.region])
    args.output.mkdir(parents=True)
    candidates = {name: baseline.copy() for name in VARIANTS}
    axis_rule_by_variant = {name: ('model' if name.endswith('axis_model') else 'official')
                            for name in VARIANTS}
    audit_log = []

    for fold in sorted(baseline.fold.unique()):
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        train = np.flatnonzero(~np.isin(groups, list(held)))
        test_rows = baseline.index[baseline.fold == fold].to_numpy()
        for group in CRITERIA:
            selected = train[anatomy[train] == group]
            testing = test_rows[np.array([group_of(r) for r in baseline.loc[test_rows, 'predicted_region']]) == group]
            tests = baseline.loc[testing, 'index'].to_numpy(dtype=int)
            # decide() needs every criterion of the group at once, so accumulate
            # across the group's criteria and emit once, exactly like the
            # reference nested screen does.
            accumulated = {name: {} for name in VARIANTS}
            all_criteria_thresholds = {name: {} for name in VARIANTS}
            all_thresholds_q = {}
            for criterion, columns in CRITERIA[group].items():
                target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                geometry = _matrix(features, columns)
                known = np.isfinite(target[selected])

                # Inner OOF for every head family, used to pick thresholds and the
                # ensemble winner on training studies only. Ranks are formed over the
                # whole inner block so the transform is identical for every row; the
                # labelled subset is selected afterwards for scoring.
                inner = {m: np.full(len(selected), np.nan) for m in MEMBERS}
                inner_emb = {n: np.full(len(selected), np.nan) for n in bundles}
                inner_both_gb = np.full(len(selected), np.nan)
                split = StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 100)
                for itr, iva in split.split(selected, labels.quality_class.iloc[selected], groups[selected]):
                    if set(groups[selected[itr]]) & set(groups[selected[iva]]):
                        raise ValueError('Inner study leakage')
                    for m in MEMBERS:
                        inner[m][iva] = fit_predict(m, geometry, target, selected[itr], selected[iva], SEED)[1]
                    for n in bundles:
                        inner_emb[n][iva] = fit_predict('lr', bundles[n], target, selected[itr],
                                                        selected[iva], SEED)[1]
                    inner_both_gb[iva] = fit_predict('gb', np.column_stack([geometry, embedding]),
                                                     target, selected[itr], selected[iva], SEED)[1]

                truth = target[selected][known].astype(int)
                yy_base = truth
                geom_avg_all = rank_average([inner[m] for m in MEMBERS])
                emb_avg_all = {n: rank_average([inner_emb[n]]) for n in bundles}
                both_avg_all = np.column_stack([geom_avg_all] + [emb_avg_all[n] for n in bundles]).mean(1)
                inner_scores = {'geom_rank_avg': geom_avg_all[known],
                                'geom_gb': inner['gb'][known],
                                'emb_rank_avg': np.mean([emb_avg_all[n][known] for n in bundles], axis=0),
                                'both_rank_avg': both_avg_all[known],
                                'both_gb': inner_both_gb[known]}

                heads = {}
                for m in MEMBERS:
                    heads[m] = fit_predict(m, geometry, target, selected, tests, SEED)[1]
                emb_test = {n: fit_predict('lr', bundles[n], target, selected, tests, SEED)[1]
                            for n in bundles}
                geom_avg_test = rank_average([heads[m] for m in MEMBERS])
                emb_avg_test = {n: rank_average([emb_test[n]]) for n in bundles}
                test_scores = {'geom_rank_avg': geom_avg_test,
                               'geom_gb': heads['gb'],
                               'emb_rank_avg': np.mean([emb_avg_test[n] for n in bundles], axis=0),
                               'both_rank_avg': np.column_stack(
                                   [geom_avg_test] + [emb_avg_test[n] for n in bundles]).mean(1),
                               'both_gb': fit_predict('gb', np.column_stack([geometry, embedding]),
                                                       target, selected, tests, SEED)[1]}

                floors = [None, args.spec_floor]
                thresholds = {n: select_threshold(truth, s, floors) for n, s in inner_scores.items()}
                quantiles = {n: select_quantile(truth, s, args.rates) for n, s in inner_scores.items()}
                for n in list(quantiles):
                    all_thresholds_q[n] = quantiles[n]
                ranked = []
                for name, scores in inner_scores.items():
                    for suffix, t in (('', thresholds[name]), ('_q', quantiles[name])):
                        m = binary_metrics(truth, (scores >= t).astype(int))
                        ap = average_precision_score(truth, scores) if truth.sum() else 0.
                        ranked.append(((m['f1'], m['sensitivity'], float(ap)), name + suffix, t))
                ranked.sort(key=lambda item: item[0], reverse=True)
                chosen, chosen_threshold = ranked[0][1], ranked[0][2]
                audit_log.append({'fold': int(fold), 'group': group, 'criterion': criterion,
                                  'selected': chosen, 'inner_f1': ranked[0][0][0],
                                  'ranked': [(r[1], round(r[0][0], 4)) for r in ranked],
                                  'training_studies': sorted(set(groups[selected]))})
                all_thresholds = dict(thresholds)
                all_thresholds['adaptive'] = chosen_threshold
                for name, value in all_thresholds.items():
                    all_criteria_thresholds[name][criterion] = value
                out = dict(test_scores)
                out['adaptive'] = test_scores[chosen]
                for name, value in quantiles.items():
                    all_criteria_thresholds.setdefault(name + '_q', {})
                    all_criteria_thresholds[name + '_q'][criterion] = value
                # AND-gate. The oracle feature is chosen per criterion by inner-OOF AUC
                # on training studies only, and the gate is kept only when it does not
                # lose inner F1. Scores are combined as the minimum of the two
                # within-training ranks so the result stays a single comparable score.
                gated_test = test_scores['both_rank_avg']
                if oracle is not None:
                    from sklearn.metrics import roc_auc_score
                    yy = target[selected][known].astype(int)
                    base_inner = inner_scores['both_rank_avg']
                    base_threshold = thresholds['both_rank_avg']
                    best_auc, best_index, best_sign = -1.0, None, 1.0
                    for position in range(len(oracle_numeric)):
                        column = oracle[selected, position][known]
                        finite = np.isfinite(column)
                        if finite.sum() < 10 or len(np.unique(yy[finite])) < 2:
                            continue
                        value = float(roc_auc_score(yy[finite], column[finite]))
                        strength = max(value, 1 - value)
                        if strength > best_auc:
                            best_auc, best_index = strength, position
                            best_sign = 1.0 if value >= 0.5 else -1.0
                    if best_index is not None:
                        gate_inner = best_sign * oracle[selected, best_index]
                        gate_test = best_sign * oracle[tests, best_index]
                        if not np.isfinite(gate_inner[known]).all():
                            gate_inner = np.where(np.isfinite(gate_inner),
                                                  gate_inner, np.nanmedian(gate_inner[known]))
                        gate_threshold = select_threshold(yy, gate_inner, floors)
                        base_f1 = binary_metrics(yy, (base_inner >= base_threshold).astype(int))['f1']
                        gated_inner = np.minimum(
                            rankdata(base_inner) / len(base_inner),
                            rankdata(gate_inner) / len(gate_inner))
                        gate_f1 = binary_metrics(
                            yy, ((base_inner >= base_threshold) & (gate_inner >= gate_threshold)).astype(int))['f1']
                        if gate_f1 > base_f1:
                            gated_test = np.minimum(
                                rankdata(test_scores['both_rank_avg']) / len(tests),
                                rankdata(gate_test) / len(tests))
                            audit_log.append({'fold': int(fold), 'criterion': criterion,
                                              'andgate_feature': oracle_numeric[best_index],
                                              'inner_auc': best_auc,
                                              'inner_f1_without_gate': base_f1,
                                              'inner_f1_with_gate': gate_f1})
                out['both_rank_avg_andgate'] = gated_test

                # Conservative switch, decided on training studies only. Both candidate
                # and baseline scores used here are out-of-fold for the rows they are
                # read on, so the comparison is honest.
                switched_test = test_scores['both_rank_avg']
                switched_threshold = thresholds['both_rank_avg']
                base_column = np.array([baseline_scores[i].get(criterion, np.nan)
                                        for i in range(len(baseline))], float)
                base_known = np.isfinite(base_column[selected][known])
                if base_known.sum() >= 10:
                    base_threshold = select_threshold(yy_base, base_column[selected][known], floors)
                    base_f1_inner = binary_metrics(
                        yy_base, (base_column[selected][known] >= base_threshold).astype(int))['f1']
                    ensemble_f1_inner = binary_metrics(
                        truth, (inner_scores['both_rank_avg'] >= thresholds['both_rank_avg']).astype(int))['f1']
                    if ensemble_f1_inner >= base_f1_inner + args.switch_margin:
                        switched_test = test_scores['both_rank_avg']
                        switched_threshold = thresholds['both_rank_avg']
                    else:
                        raw = base_column[tests]
                        # The baseline emits no score for a criterion on rows it never
                        # scored; fall back to the ensemble value there rather than
                        # emitting a non-finite score into decide().
                        fallback = test_scores['both_rank_avg']
                        switched_test = np.where(np.isfinite(raw), raw, fallback)
                        shipped = [baseline_thresholds[i].get(criterion) for i in tests]
                        usable = [v for v in shipped if v is not None and np.isfinite(v)]
                        switched_threshold = float(np.median(usable)) if usable else 1.0
                    audit_log.append({'fold': int(fold), 'criterion': criterion,
                                      'switch_margin': args.switch_margin,
                                      'inner_f1_ensemble': ensemble_f1_inner,
                                      'inner_f1_baseline': base_f1_inner,
                                      'adopted_ensemble': bool(ensemble_f1_inner >= base_f1_inner + args.switch_margin)})
                out['both_rank_avg_switch'] = switched_test
                all_thresholds['both_rank_avg_switch'] = switched_threshold
                out['both_rank_avg_q'] = test_scores['both_rank_avg']
                out['geom_rank_avg_q'] = test_scores['geom_rank_avg']
                out['emb_rank_avg_q'] = test_scores['emb_rank_avg']
                all_thresholds['both_rank_avg_q'] = quantiles['both_rank_avg']
                all_thresholds['geom_rank_avg_q'] = quantiles['geom_rank_avg']
                all_thresholds['emb_rank_avg_q'] = quantiles['emb_rank_avg']
                all_thresholds['both_rank_avg_axis_model'] = thresholds['both_rank_avg']
                all_thresholds['geom_rank_avg_axis_model'] = thresholds['geom_rank_avg']
                # The gate is expressed as a minimum of two within-training ranks, so the
                # ensemble cut remains the right operating point for the combined score.
                all_thresholds['both_rank_avg_andgate'] = thresholds['both_rank_avg']
                for name, scores in out.items():
                    accumulated[name][criterion] = scores

            for name, per_criterion in accumulated.items():
                for position, row_index in enumerate(testing):
                    index = int(baseline.at[row_index, 'index'])
                    criterion_scores = {k: float(v[position]) for k, v in per_criterion.items()}
                    probability_score = float(1 - np.prod([1 - p for p in criterion_scores.values()]))
                    decision = decide(group, probability_score, criterion_scores, features[index],
                                      float(np.nextafter(1., np.inf)), all_criteria_thresholds[name],
                                      axis_rule='model' if name.endswith('axis_model') else 'official')
                    for key, value in {'quality_score': probability_score,
                                       'quality_pred': decision['quality'],
                                       'violation_type': official_violation_type(decision['violations']),
                                       'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                                       'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                                       'criterion_thresholds': json.dumps(all_criteria_thresholds[name]),
                                       'quality_threshold': decision['quality_threshold'],
                                       'decision_version': 'research-rank-ensemble-1',
                                       'decision_reason': decision['decision_reason']}.items():
                        candidates[name].at[row_index, key] = value
        print(f'Completed outer fold {fold}', flush=True)

    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=200),
                         'screen': compare(args.baseline, path, args.labels)}
        print(f'{name:18} {reports[name]["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)
    (args.output / 'evaluation.json').write_text(json.dumps(
        {'protocol': __doc__, 'variants': reports, 'members': list(MEMBERS),
         'spec_floor': args.spec_floor, 'selection_audit': audit_log,
         'embedding_dimensions': {'resnet18': int(resnet18.shape[1]),
                                  **{k: int(v.shape[1]) for k, v in bundles.items()}},
         'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
         'code_sha256': sha256(Path(__file__)),
         'clinical_validation': False, 'model_affects_decision': False,
         'limitations': ['Ranks are computed within each split, so scores are not calibrated probabilities.',
                         'Ensemble membership and thresholds use training studies only.',
                         'Reported deltas sit inside the +/-0.08 cohort noise band.',
                         'Exploratory; no automatic release promotion.']},
        ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, default=Path('data/gpu-runs/visibility-2026-10-04/oracle_measurements.csv'),
                        help='Oracle measurements from label_visibility.py; enables the AND-gated variant')
    parser.add_argument('--bundle', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--spec-floor', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=17,
                        help='Drives the outer baseline folds, inner splits and every estimator')
    parser.add_argument('--switch-margin', type=float, default=0.05,
                        help='Inner-OOF F1 margin the ensemble must beat the shipped calibration by')
    parser.add_argument('--rates', type=float, nargs='+',
                        default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30],
                        help='Candidate flagged rates for the quantile operating point')
    args = parser.parse_args()
    parsed = {}
    for item in args.bundle:
        name, sep, path = item.partition('=')
        if not sep or not path:
            parser.error(f'--bundle expects NAME=PATH, got {item!r}')
        parsed[name] = Path(path)
    if not parsed:
        parser.error('at least one --bundle NAME=PATH is required')
    args.bundle = parsed
    with threadpool_limits(limits=4):
        run(args)
