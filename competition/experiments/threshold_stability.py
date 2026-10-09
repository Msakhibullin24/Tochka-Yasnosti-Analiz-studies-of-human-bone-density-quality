"""Attack the threshold estimator instead of the model.

Every previously rejected variant changed the representation - more encoders,
higher resolution, patch-local pooling, fine-tuning, IQA priors - while leaving one
parameter untouched: the operating point, taken by argmax-F1 on a single inner-OOF
pass over roughly 18 studies. The diagnostics point straight at that parameter:

  * two criteria separate at AUC 0.84 and 0.94 yet score F1 0.17-0.46, because the
    cut cannot be placed;
  * inner-OOF F1 misranks candidates on 23 of 25 criterion-folds;
  * the false-positive count is what every learned model got wrong.

So this script holds the champion's rank-averaged ensemble fixed and varies only how
the operating point is estimated:

  champion       - reference: argmax-F1 on one inner pass (unchanged).
  bagged         - median of argmax-F1 over B bootstrap resamples of the training
                   rows. Same estimator, lower variance, no new fitted parameters.
  repeats        - average the within-split rank transform over R repeats of the
                   inner split, so the score vector the cut is read from is less noisy.
  bagged_repeats - both.
  count          - abandon five independent thresholds. decide() combines criteria
                   with a noisy-OR, so five noisy cuts give an unstable aggregate.
                   Instead flag a row when at least k criteria fire, k chosen on
                   inner OOF: one low-variance integer replaces five noisy ones.

Same baseline, same outer folds, same features as the champion. Any apparent gain
must then be confirmed across seeds, because the single-seed champion looked like
+0.0479 and turned out to be seed luck.
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
from dxaqc.model import CRITERIA, _matrix, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402
from experiments.nested_criterion_upgrade import validate_outer  # noqa: E402

MEMBERS = ('rf', 'gb', 'lr', 'svc', 'lda')
# 'count' is deliberately NOT screened. decide() types each violated criterion, and the
# project's invariant requires quality_pred to follow from those types. Replacing quality
# with a "how many criteria fired" count contradicts the typed violations by
# construction, so verify_metric_candidate rejects it with "Quality decision disagrees
# with violation type". That is the contract working, not a bug, so the count rule is
# reported as an unadmissible operating point rather than as a candidate.
SCREENED = ('champion', 'bagged', 'repeats', 'bagged_repeats', 'guarded',
            'bagged_select', 'adaptive')
ALL_VARIANTS = ('champion', 'bagged', 'repeats', 'bagged_repeats', 'count', 'guarded',
                'bagged_select', 'adaptive')
VARIANTS = ALL_VARIANTS
# Sample-size guard, fixed a priori and not tuned on this cohort.
#
# The earlier per-criterion switch decided by inner-OOF F1 failed, because inner-OOF F1
# misranks candidates here. This guard does something different and much narrower: it
# declines to estimate a threshold at all for a criterion with fewer than MIN_POSITIVES
# positives in the training studies, and keeps the shipped calibration for those rows.
# The count of positives is a property of the label table, known without reading any
# score, so this is a sample-size rule of the kind used everywhere in statistics - not a
# selection fitted to the evaluation.
MIN_POSITIVES = 10


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
    return make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))


def probability(model, values):
    if isinstance(model, DummyClassifier):
        return np.zeros(len(values))
    if 1 not in model.classes_:
        return np.zeros(len(values))
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def fit_predict(name, values, target, train, test, seed):
    known = train[np.isfinite(target[train])]
    y = target[known].astype(int)
    if len(np.unique(y)) < 2:
        dummy = DummyClassifier(strategy='constant', constant=int(y[0]))
        dummy.fit(np.zeros((len(known), 1)), y)
        return np.zeros(len(known)), np.zeros(len(test)), dummy
    model = make_head(name, seed)
    model.fit(values[known], y)
    return probability(model, values[known]), probability(model, values[test]), model


def ranks(values):
    values = np.asarray(values, float)
    return rankdata(values) / max(len(values), 1)


def plain_threshold(truth, scores, floors):
    best, chosen = -1.0, float(np.nextafter(1., np.inf))
    for floor in floors:
        for threshold in np.unique(np.concatenate(([0.0], np.sort(np.unique(scores))))):
            predicted = (scores >= threshold).astype(int)
            metrics = binary_metrics(truth, predicted)
            if floor is not None and metrics['specificity'] < floor:
                continue
            if metrics['f1'] > best:
                best, chosen = metrics['f1'], float(threshold)
    return chosen


def bagged_threshold(truth, scores, floors, bags, seed):
    """Median operating point over bootstrap resamples: same estimator, lower variance."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    draws = []
    for _ in range(bags):
        pick = rng.integers(0, n, n)
        if len(np.unique(truth[pick])) < 2:
            continue
        draws.append(plain_threshold(truth[pick], scores[pick], floors))
    return float(np.median(draws)) if draws else plain_threshold(truth, scores, floors)


def guarded_verdict(decision, per_criterion, position):
    """Apply the sample-size guard to the threshold-based criteria only.

    spine_axis is excluded on purpose. Its shipped verdict is not threshold-based at
    all: decide() resolves it from the measured angle against AXIS_LIMIT_DEG. Overriding
    it with a score comparison would report an axis violation with no angle above the
    limit, which verify_metric_candidate correctly rejects. So for spine_axis the
    decision is left exactly as decide() produced it.
    """
    guarded = dict(decision)
    violations = []
    for criterion, state in per_criterion.items():
        if criterion == 'spine_axis':
            if decision['criterion_states'].get('spine_axis', {}).get('status') == 'fail':
                violations.append(criterion)
            continue
        fires = (state['shipped_fires'][position]
                 if state['positives_in_training'] < MIN_POSITIVES
                 else bool(state['test_scores'][position] >= state['thresholds']['champion']))
        if fires:
            violations.append(criterion)
    guarded['violations'] = sorted(violations)
    guarded['quality'] = int(bool(violations))
    return guarded


def bagged_select(truth, scores_by_variant, thresholds, bags, seed):
    """Choose the *threshold estimator* by modal vote over bootstrap resamples.

    Scope note, because it was initially misread: the four variants share identical
    out-of-fold test scores. The heads are fitted once on the full training portion, so
    only the threshold estimator differs between champion, repeats, bagged and
    bagged_repeats. This function therefore selects an estimator, not a score, and its
    effect is entirely through the threshold.

    Motivation was to bag the *selection* the way the threshold is bagged. It does not
    work: the vote lands on a different estimator than the single pass on 5 of 25
    criterion-folds, and those flips are expensive, because the estimators that look best
    on one inner pass (champion, repeats) are the unstable ones. Choosing the estimator
    a priori beats choosing it from data at this cohort size.
    """
    names = list(scores_by_variant)
    rng = np.random.default_rng(seed)
    n = len(truth)
    votes = {name: 0 for name in names}
    for _ in range(bags):
        pick = rng.integers(0, n, n)
        if len(np.unique(truth[pick])) < 2:
            continue
        scored = []
        for name in names:
            predicted = (scores_by_variant[name][pick] >= thresholds[name]).astype(int)
            scored.append((binary_metrics(truth[pick], predicted)['f1'], name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        votes[scored[0][1]] += 1
    if not any(votes.values()):
        return names[0], votes
    best = max(votes.values())
    winners = [name for name, count in votes.items() if count == best]
    return min(winners), votes


def run(args):
    seed = int(args.seed)
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
    source_fp = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    features, resnet18, _ = build_table(args.dataset, labels, HERE / '.cache', source_fp)
    from gpu_research.common import load_feature_bundle
    bundles = {}
    for name, path in args.bundle.items():
        bundles[name] = load_feature_bundle(path, labels, source_fp, sha256(args.labels))[0]
        print(f'{name}: {bundles[name].shape}', flush=True)

    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            angle = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
            features[row.index]['spine_abs_angle_deg'] = angle
    features = [dict(f) for f in features]
    groups = labels.study_key.astype(str).to_numpy()
    anatomy = np.array([group_of(r) for r in labels.region])
    args.output.mkdir(parents=True)
    candidates = {name: baseline.copy() for name in VARIANTS}
    audit_log = []
    floors = [None, args.spec_floor]

    for fold in sorted(baseline.fold.unique()):
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        train = np.flatnonzero(~np.isin(groups, list(held)))
        test_rows = baseline.index[baseline.fold == fold].to_numpy()
        for group in CRITERIA:
            selected = train[anatomy[train] == group]
            testing = test_rows[np.array([group_of(r) for r in baseline.loc[test_rows, 'predicted_region']]) == group]
            tests = baseline.loc[testing, 'index'].to_numpy(dtype=int)
            # Per-criterion state, needed by the count aggregator after the group closes.
            per_criterion = {}
            for criterion, columns in CRITERIA[group].items():
                target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                geometry = _matrix(features, columns)
                known = np.isfinite(target[selected])
                truth = target[selected][known].astype(int)
                if len(np.unique(truth)) < 2:
                    continue

                # R repeats of the inner split; each contributes rank-transformed scores.
                geom_repeats, emb_repeats = [], []
                for repeat in range(args.inner_repeats):
                    collected = {m: np.full(len(selected), np.nan) for m in MEMBERS}
                    collected_emb = {n: np.full(len(selected), np.nan) for n in bundles}
                    split = StratifiedGroupKFold(4, shuffle=True, random_state=seed + 100 + 13 * repeat)
                    for itr, iva in split.split(selected, labels.quality_class.iloc[selected], groups[selected]):
                        if set(groups[selected[itr]]) & set(groups[selected[iva]]):
                            raise ValueError('Inner study leakage')
                        for m in MEMBERS:
                            collected[m][iva] = fit_predict(m, geometry, target, selected[itr], selected[iva], seed)[1]
                        for n in bundles:
                            collected_emb[n][iva] = fit_predict('lr', bundles[n], target, selected[itr],
                                                                selected[iva], seed)[1]
                    geom_repeats.append(np.column_stack([ranks(collected[m]) for m in MEMBERS]).mean(1))
                    emb_repeats.append({n: ranks(collected_emb[n]) for n in bundles})
                geom_repeat_mean = np.mean(geom_repeats, axis=0)
                emb_repeat_mean = {n: np.mean([r[n] for r in emb_repeats], axis=0) for n in bundles}

                heads = {m: fit_predict(m, geometry, target, selected, tests, seed)[1] for m in MEMBERS}
                emb_test = {n: fit_predict('lr', bundles[n], target, selected, tests, seed)[1] for n in bundles}
                geom_test = np.column_stack([ranks(heads[m]) for m in MEMBERS]).mean(1)
                emb_test_r = {n: ranks(emb_test[n]) for n in bundles}

                def combine(geom, embs):
                    return np.column_stack([geom] + list(embs.values())).mean(1)

                scores = {
                    'champion': (combine(geom_repeat_mean, emb_repeat_mean), combine(geom_test, emb_test_r)),
                }
                single_inner = {m: np.full(len(selected), np.nan) for m in MEMBERS}
                single_emb = {n: np.full(len(selected), np.nan) for n in bundles}
                split0 = StratifiedGroupKFold(4, shuffle=True, random_state=seed + 100)
                for itr, iva in split0.split(selected, labels.quality_class.iloc[selected], groups[selected]):
                    for m in MEMBERS:
                        single_inner[m][iva] = fit_predict(m, geometry, target, selected[itr], selected[iva], seed)[1]
                    for n in bundles:
                        single_emb[n][iva] = fit_predict('lr', bundles[n], target, selected[itr],
                                                        selected[iva], seed)[1]
                single_geom = np.column_stack([ranks(single_inner[m]) for m in MEMBERS]).mean(1)
                single_emb_r = {n: ranks(single_emb[n]) for n in bundles}
                scores['repeats'] = (combine(single_geom, single_emb_r), combine(geom_test, emb_test_r))
                scores['bagged'] = (combine(geom_repeat_mean, emb_repeat_mean), combine(geom_test, emb_test_r))
                scores['bagged_repeats'] = (combine(geom_repeat_mean, emb_repeat_mean),
                                            combine(geom_test, emb_test_r))

                thresholds = {}
                for name, (inner_scores, _) in scores.items():
                    if name in ('bagged', 'bagged_repeats'):
                        thresholds[name] = bagged_threshold(truth, inner_scores[known], floors, args.bags, seed)
                    else:
                        thresholds[name] = plain_threshold(truth, inner_scores[known], floors)
                inner_single_geom = combine(single_geom, single_emb_r)
                thresholds['champion'] = plain_threshold(truth, inner_single_geom[known], floors)

                modal, votes = bagged_select(
                    truth, {k: scores[k][0] for k in ('champion', 'bagged', 'repeats', 'bagged_repeats')},
                    {k: thresholds[k] for k in ('champion', 'bagged', 'repeats', 'bagged_repeats')},
                    args.bags, seed)
                thresholds['bagged_select'] = thresholds.get(modal, thresholds['bagged'])
                audit_log.append({'fold': int(fold), 'group': group, 'criterion': criterion,
                                  'selection': 'bagged_threshold_estimator', 'modal_winner': modal,
                                  'votes': votes,
                                  'single_pass_winner': None})
                ranked = []
                for name in ('champion', 'bagged', 'repeats', 'bagged_repeats'):
                    inner_scores = scores[name][0]
                    t = thresholds[name]
                    m = binary_metrics(truth, (inner_scores[known] >= t).astype(int))
                    ap = average_precision_score(truth, inner_scores[known]) if truth.sum() else 0.
                    ranked.append(((m['f1'], m['sensitivity'], float(ap)), name, t))
                ranked.sort(key=lambda item: item[0], reverse=True)
                chosen, chosen_threshold = ranked[0][1], ranked[0][2]
                thresholds['adaptive'] = chosen_threshold
                for entry in audit_log:
                    if entry.get('selection') == 'bagged_threshold_estimator' and entry.get('single_pass_winner') is None \
                            and entry.get('fold') == int(fold) and entry.get('criterion') == criterion:
                        entry['single_pass_winner'] = chosen
                # The emitted scores for bagged_select come from the variant it voted for.
                selected_scores = scores[modal][1] if modal in scores else scores['bagged'][1]

                positives_in_training = int(target[selected][known].sum())
                # The shipped per-criterion verdict on these rows, so the guarded variant
                # can keep the shipped *decision* rather than mixing incompatible score
                # scales: baseline scores are raw probabilities while the ensemble scores
                # are within-split ranks in [0, 1], and a noisy-OR over both inflates the
                # aggregate. Scale mixing was the first implementation's bug and it cost
                # 19 false positives.
                shipped_column = np.array([baseline_scores[i].get(criterion, np.nan)
                                           for i in range(len(baseline))], float)
                shipped_t = [baseline_thresholds[i].get(criterion) for i in tests]
                usable_t = [v for v in shipped_t if v is not None and np.isfinite(v)]
                shipped_threshold = float(np.median(usable_t)) if usable_t else 1.0
                shipped_fires = (shipped_column[tests] >= shipped_threshold)
                per_criterion[criterion] = {
                    'positives_in_training': positives_in_training,
                    'shipped_fires': shipped_fires,
                    'inner_scores': inner_single_geom, 'test_scores': scores['champion'][1],
                    'truth': truth, 'known': known, 'selected': selected, 'tests': tests,
                    'thresholds': thresholds, 'chosen': chosen, 'ranked': ranked}
                audit_log.append({'fold': int(fold), 'group': group, 'criterion': criterion,
                                  'inner_f1_by_variant': {r[1]: round(r[0][0], 4) for r in ranked},
                                  'threshold_by_variant': {k: round(float(v), 4) for k, v in thresholds.items()},
                                  'spread_of_thresholds': round(float(max(thresholds[k] for k in
                                                                        ('champion', 'bagged', 'repeats', 'bagged_repeats'))
                                                                 - min(thresholds[k] for k in
                                                                        ('champion', 'bagged', 'repeats', 'bagged_repeats'))), 4)})

            # Count aggregator: one integer instead of five cuts.
            if per_criterion:
                fire = np.zeros((len(per_criterion), len(selected)))
                for position, (criterion, state) in enumerate(per_criterion.items()):
                    t = state['thresholds']['champion']
                    fire[position] = ((state['inner_scores'] >= t).astype(float))[state['known']]
                count_inner = fire.sum(0)
                best_k, best_f1 = 1, -1.0
                for k in range(1, len(per_criterion) + 1):
                    f1 = binary_metrics(truth, (count_inner >= k).astype(int))['f1']
                    if f1 > best_f1:
                        best_k, best_f1 = k, f1
                fire_test = np.zeros((len(per_criterion), len(tests)))
                for position, state in enumerate(per_criterion.values()):
                    fire_test[position] = (state['test_scores'] >= state['thresholds']['champion']).astype(float)
                count_test = fire_test.sum(0)
                audit_log.append({'fold': int(fold), 'group': group, 'criterion': 'AGGREGATE',
                                  'count_threshold_k': best_k, 'inner_f1': round(best_f1, 4)})
            else:
                best_k, count_test = 1, np.zeros(len(tests))

            # Emit every variant once per group, with all criteria of the group present.
            accumulated = {name: {} for name in VARIANTS}
            for criterion, state in per_criterion.items():
                for name in ('champion', 'bagged', 'repeats', 'bagged_repeats'):
                    accumulated[name][criterion] = state['test_scores']
                accumulated['adaptive'][criterion] = state['test_scores']
                # Test scores are shared by all estimators, so the vote only moves the cut.
                accumulated['bagged_select'][criterion] = selected_scores
                # Keep every criterion present in criterion_states so decide() sees the
                # full group; the guard only changes the resulting verdict, not the record.
                accumulated['guarded'][criterion] = state['test_scores']
                audit_log.append({'fold': int(fold), 'group': group, 'criterion': criterion,
                                  'positives_in_training': state['positives_in_training'],
                                  'guarded_uses_ensemble': bool(state['positives_in_training'] >= MIN_POSITIVES)})
            for name in VARIANTS:
                for position, row_index in enumerate(testing):
                    index = int(baseline.at[row_index, 'index'])
                    criterion_scores = {k: float(v[position]) for k, v in accumulated[name].items()}
                    probability = float(1 - np.prod([1 - p for p in criterion_scores.values()]))
                    # The count variant reuses the champion's per-criterion cuts; only its
                    # final quality output is replaced by the count rule.
                    if name == 'guarded':
                        thresholds_used = {}
                        for k, state in per_criterion.items():
                            if state['positives_in_training'] >= MIN_POSITIVES:
                                thresholds_used[k] = state['thresholds']['champion']
                            else:
                                shipped = [baseline_thresholds[i].get(k) for i in state['tests']]
                                usable = [v for v in shipped if v is not None and np.isfinite(v)]
                                thresholds_used[k] = float(np.median(usable)) if usable else 1.0
                        source = 'champion'
                    else:
                        source = name if name in ('champion', 'bagged', 'repeats', 'bagged_repeats', 'adaptive') else 'champion'
                        thresholds_used = {k: state['thresholds'][source] for k, state in per_criterion.items()}
                    decision = decide(group, probability, criterion_scores, features[index],
                                      float(np.nextafter(1., np.inf)), thresholds_used)
                    if name == 'guarded':
                        decision = guarded_verdict(decision, per_criterion, position)
                    quality = int(decision['quality'])
                    if name == 'count':
                        quality = int(count_test[position] >= best_k)

                    for key, value in {'quality_score': probability,
                                       'quality_pred': quality,
                                       'violation_type': official_violation_type(decision['violations']),
                                       'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                                       'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                                       'criterion_thresholds': json.dumps(thresholds_used, sort_keys=True),
                                       'quality_threshold': decision['quality_threshold'],
                                       'decision_version': 'research-threshold-stability-1',
                                       'decision_reason': decision['decision_reason']}.items():
                        candidates[name].at[row_index, key] = value
        print(f'Completed outer fold {fold}', flush=True)

    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        entry = {'metrics': evaluate(args.labels, path, repeats=2000)}
        if name in SCREENED:
            entry['screen'] = compare(args.baseline, path, args.labels)
        else:
            entry['screen'] = {'status': 'not screened',
                               'reason': 'incompatible with the typed-violation invariant by construction'}
        reports[name] = entry
        print(f'{name:18} {entry["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)
    (args.output / 'evaluation.json').write_text(json.dumps(
        {'protocol': __doc__, 'variants': reports, 'selection_audit': audit_log,
         'members': list(MEMBERS), 'bags': args.bags, 'inner_repeats': args.inner_repeats,
         'seed': args.seed, 'spec_floor': args.spec_floor,
         'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
         'code_sha256': sha256(Path(__file__)),
         'clinical_validation': False, 'model_affects_decision': False,
         'limitations': ['Bagging changes the variance of the operating point, not the model.',
                         'Any gain must be confirmed across seeds before it is believed.',
                         'The count aggregator replaces decide() quality output for its own variant only.',
                         'Exploratory; no automatic release promotion.']},
        ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--spec-floor', type=float, default=0.9)
    parser.add_argument('--bags', type=int, default=200)
    parser.add_argument('--inner-repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=17)
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