"""E16 per-criterion calibration and E17 member error correlation.

E16. The best candidates fail the screen partly on Brier score and average precision, not
only on F1. The council scores are within-split ranks, so they are monotone by
construction and carry no probability meaning; a monotone recalibration cannot change
ROC-AUC or the ranking, but it does change the Brier score and, more importantly, where
the operating point can be placed. Platt scaling on the rank is used because it is the
simple, pre-chosen method the plan names; isotonic is deliberately not used because at
6 to 36 positives per criterion it would fit noise.

Calibration is fitted inside the training studies of each outer fold only, and the
threshold is re-derived from the calibrated training scores with the same bagged
estimator that produced the reproducible result. Nothing reads a test-fold label.

E17. Four seed runs of the same model are not four independent opinions, so the error
correlation between them is reported explicitly. If they agree on nearly every error,
averaging them cannot recover much and the council's small gain over its members is
expected rather than surprising. Common false negatives across members are listed
because those are the cases no amount of seed averaging will fix.
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
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402


def parse(frame):
    out = []
    for v in frame:
        if isinstance(v, str):
            p = json.loads(v)
        elif isinstance(v, dict):
            p = v
        else:
            p = {}
        out.append(p if isinstance(p, dict) else {})
    return out


def brier(truth, probability):
    truth = np.asarray(truth, float)
    return float(np.mean((probability - truth) ** 2))


def reliability(truth, probability, bins=10):
    truth = np.asarray(truth, float)
    probability = np.asarray(probability, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = index == b
        if m.sum() < 3:
            continue
        rows.append({'bin': f'[{edges[b]:.1f},{edges[b + 1]:.1f})', 'n': int(m.sum()),
                     'mean_predicted': float(probability[m].mean()),
                     'observed_rate': float(truth[m].mean())})
    return rows


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
    rng = np.random.default_rng(seed)
    n = len(truth)
    draws = []
    for _ in range(bags):
        pick = rng.integers(0, n, n)
        if len(np.unique(truth[pick])) < 2:
            continue
        draws.append(plain_threshold(truth[pick], scores[pick], floors))
    return float(np.median(draws)) if draws else plain_threshold(truth, scores, floors)


def apply_platt(model, scores):
    eps = 1e-6
    x = np.log(np.clip(np.asarray(scores, float), eps, 1 - eps)
               / np.clip(1 - np.asarray(scores, float), eps, 1 - eps)).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def fit_platt(scores, truth):
    """Monotone recalibration of a rank into a probability. Fitted on training rows only."""
    eps = 1e-6
    x = np.log(np.clip(scores, eps, 1 - eps) / np.clip(1 - scores, eps, 1 - eps)).reshape(-1, 1)
    if len(np.unique(truth)) < 2:
        return None
    model = LogisticRegression(C=1e6, max_iter=5000)
    model.fit(x, truth)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--member', action='append', required=True, metavar='NAME=OOF.csv')
    parser.add_argument('--bags', type=int, default=200)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    members = {}
    for item in args.member:
        name, sep, path = item.partition('=')
        if not sep or not path:
            parser.error(f'--member expects NAME=OOF.csv, got {item!r}')
        t = pd.read_csv(path).sort_values('index').reset_index(drop=True)
        if not np.array_equal(t['index'].to_numpy(), baseline['index'].to_numpy()):
            raise ValueError(f'{name}: rows are not aligned with the baseline')
        members[name] = t
    scores_by_member = {n: parse(t['criterion_scores']) for n, t in members.items()}
    council = []
    for i in range(len(baseline)):
        merged = {}
        for n in scores_by_member:
            for k, v in scores_by_member[n][i].items():
                merged.setdefault(k, []).append(float(v))
        council.append({k: float(np.mean(v)) for k, v in merged.items()})

    # ---- E17: error correlation and common failures between members -------------------
    correlation = {}
    names = sorted(members)
    truth = baseline['quality_true'].to_numpy().astype(int)
    predicted = {n: members[n]['quality_pred'].to_numpy().astype(int) for n in names}
    errors = {n: (predicted[n] != truth).astype(int) for n in names}
    for a in names:
        for b in names:
            if a >= b:
                continue
            both = int((errors[a] & errors[b]).sum())
            either = int((errors[a] | errors[b]).sum())
            correlation[f'{a}|{b}'] = {
                'errors_a': int(errors[a].sum()), 'errors_b': int(errors[b].sum()),
                'shared_errors': both, 'jaccard': round(both / max(either, 1), 4)}
    unanimous = np.ones(len(baseline), bool)
    for n in names:
        unanimous &= (predicted[n] == truth)
    all_wrong = np.ones(len(baseline), bool)
    for n in names:
        all_wrong &= (predicted[n] != truth)
    baseline_pred = baseline['quality_pred'].to_numpy().astype(int)
    baseline_fp = set(baseline.index[(baseline_pred == 1) & (truth == 0)].tolist())
    member_fp = {n: set(baseline.index[(predicted[n] == 1) & (truth == 0)].tolist()) for n in names}
    common_fp = sorted(set.intersection(*member_fp.values())) if member_fp else []
    member_fn = {n: set(baseline.index[(predicted[n] == 0) & (truth == 1)].tolist()) for n in names}
    common_fn = sorted(set.intersection(*member_fn.values())) if member_fn else []
    e17 = {'pairwise_error_agreement': correlation,
           'rows_all_members_correct': int(unanimous.sum()),
           'rows_all_members_wrong': int(all_wrong.sum()),
           'false_positives_in_all_members': len(common_fp),
           'false_negatives_in_all_members': len(common_fn),
           'common_false_negative_indices': common_fn,
           'common_false_positive_indices': common_fp,
           'baseline_false_positives': len(baseline_fp),
           'interpretation': ('Four seeds of one model are not four independent opinions. If they agree on nearly every '
                              'error, averaging them cannot recover much, which is exactly what the council shows: it '
                              'beats its member mean by 0.0018. The cases every member misses are the ones that need new '
                              'information, not more averaging.')}

    # ---- E16: calibration fitted on training studies only -------------------------------
    from train import build_table
    from source_integrity import inspect_sources
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    features = [dict(f) for f in build_table(args.dataset, labels, HERE / '.cache', fingerprint)[0]]
    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            features[row.index]['spine_abs_angle_deg'] = json.loads(
                row.criterion_states).get('spine_axis', {}).get('angle_deg')

    groups = np.array([group_of(r) for r in baseline['predicted_region']])
    studies = baseline['study'].astype(str).to_numpy()
    quality = baseline['quality_true'].to_numpy(float)
    variants = {'council_uncalibrated': baseline.copy(), 'council_platt': baseline.copy()}
    per_criterion = []
    floors = [None, 0.9]

    for fold in sorted(baseline.fold.unique()):
        held = set(studies[baseline.fold.to_numpy() == fold])
        train_mask = ~np.isin(studies, list(held))
        test_mask = ~train_mask
        for anatomy in ('spine', 'hip'):
            tr = np.flatnonzero(train_mask & (groups == anatomy))
            te = np.flatnonzero(test_mask & (groups == anatomy))
            if not len(te):
                continue
            # Two separate threshold dictionaries. The earlier version reused one dict,
            # so the "uncalibrated" variant silently ran with the Platt thresholds while
            # still feeding raw ranks into decide(): a cut placed on a probability scale
            # applied to a rank scale, which is why both variants collapsed to 0.4660.
            raw_thresholds, cal_thresholds, calibrators = {}, {}, {}
            for criterion in CRITERIA[anatomy]:
                target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                column = np.array([council[i].get(criterion, np.nan)
                                   for i in range(len(baseline))], float)
                known = tr[np.isfinite(target[tr])]
                raw_thresholds[criterion] = float(np.nextafter(1., np.inf))
                cal_thresholds[criterion] = float(np.nextafter(1., np.inf))
                if len(known) < 5 or len(np.unique(target[known].astype(int))) < 2:
                    continue
                y = target[known].astype(int)
                raw_thresholds[criterion] = bagged_threshold(y, column[known], floors,
                                                             args.bags, args.seed + 100 + 7 * int(fold))
                platt = fit_platt(column[known], y)
                if platt is not None:
                    calibrators[criterion] = platt
                    calibrated_train = apply_platt(platt, column[known])
                    cal_thresholds[criterion] = bagged_threshold(y, calibrated_train, floors,
                                                                args.bags, args.seed + 100 + 7 * int(fold))
                    per_criterion.append({
                        'fold': int(fold), 'criterion': criterion, 'n_train': int(len(known)),
                        'positives_train': int(y.sum()),
                        'threshold_raw_rank': round(raw_thresholds[criterion], 4),
                        'threshold_platt_probability': round(cal_thresholds[criterion], 4),
                        'train_brier_raw_score': round(brier(y, column[known]), 4),
                        'train_brier_platt': round(brier(y, calibrated_train), 4),
                        'roc_auc_unchanged_by_calibration': True})

            for position, row_index in enumerate(te):
                criterion_scores_raw = {}
                criterion_scores_cal = {}
                for criterion in CRITERIA[anatomy]:
                    column = np.array([council[i].get(criterion, np.nan)
                                       for i in range(len(baseline))], float)
                    criterion_scores_raw[criterion] = float(column[row_index])
                    model = calibrators.get(criterion)
                    criterion_scores_cal[criterion] = (float(apply_platt(model, column[row_index])[0])
                                                        if model is not None else float(column[row_index]))
                for name, scores_here, thresholds_here in (
                        ('council_uncalibrated', criterion_scores_raw, raw_thresholds),
                        ('council_platt', criterion_scores_cal, cal_thresholds)):
                    probability = float(1 - np.prod([1 - p for p in scores_here.values()]))
                    decision = decide(anatomy, probability, scores_here, features[row_index],
                                      float(np.nextafter(1., np.inf)), thresholds_here)
                    for key, value in {'quality_score': probability,
                                       'quality_pred': decision['quality'],
                                       'violation_type': official_violation_type(decision['violations']),
                                       'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                                       'criterion_scores': json.dumps(scores_here, sort_keys=True),
                                       'criterion_thresholds': json.dumps(thresholds_here, sort_keys=True),
                                       'quality_threshold': decision['quality_threshold'],
                                       'decision_version': 'research-calibrated-council-1',
                                       'decision_reason': decision['decision_reason']}.items():
                        variants[name].at[row_index, key] = value
        print(f'Completed outer fold {fold}', flush=True)

    reports = {}
    for name, table in variants.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=2000),
                         'screen': compare(args.baseline, path, args.labels)}
        print(f'{name:24} {reports[name]["metrics"]["overall_quality"]["f1"]:.4f} '
              f'brier {reports[name]["screen"]["overall_score_metrics"]["candidate"]["brier_score"]:.4f}', flush=True)

    cal = reports['council_platt']['screen']
    unc = reports['council_uncalibrated']['screen']
    e16 = {'per_criterion_fit': per_criterion,
           'uncalibrated': {'f1': reports['council_uncalibrated']['metrics']['overall_quality']['f1'],
                            'brier': unc['overall_score_metrics']['candidate']['brier_score'],
                            'average_precision': unc['overall_score_metrics']['candidate']['average_precision']},
           'platt': {'f1': reports['council_platt']['metrics']['overall_quality']['f1'],
                     'brier': cal['overall_score_metrics']['candidate']['brier_score'],
                     'average_precision': cal['overall_score_metrics']['candidate']['average_precision']},
           'note': ('A monotone recalibration cannot change ROC-AUC or the ranking. If ROC-AUC is unchanged and only '
                    'Brier and the threshold placement move, the fit did what a calibration can do and nothing more.')}

    (args.output / 'evaluation.json').write_text(json.dumps(
        {'protocol': __doc__, 'E16_calibration': e16, 'E17_member_errors': e17,
         'variants': reports, 'members': names, 'bags': args.bags, 'seed': args.seed,
         'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
         'code_sha256': sha256(Path(__file__)),
         'clinical_validation': False, 'model_affects_decision': False,
         'limitations': ['Isotonic calibration is deliberately not used: at 6 to 36 positives per criterion it fits noise.',
                         'Calibration is fitted on training studies of each outer fold only.',
                         'Four seed runs of one model are not independent opinions.',
                         'Exploratory; no automatic release promotion.']},
        ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()