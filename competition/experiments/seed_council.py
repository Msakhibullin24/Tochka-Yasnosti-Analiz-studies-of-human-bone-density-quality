"""Council of independently fitted models: average out-of-fold scores across seeds.

Bagging reduced the variance of one fitted quantity, the threshold. This applies the
same template one level up: the four seed runs are four independent draws of the inner
partition and of every estimator's random_state, and each produces its own honest
out-of-fold criterion scores for the same 249 rows. Averaging them is plain score-level
ensembling of independently fitted models, and it costs nothing - every score is already
computed.

Nothing here re-fits and nothing touches a test-fold label:

  * every constituent score is out-of-fold for the row it is read on;
  * for each outer fold the operating point is chosen on the training studies only, from
    the seed-averaged scores of those rows;
  * the decision layer, routing and measured spine axis are the shipped ones.

The council is evaluated alongside its members so the ensemble effect can be separated
from the member quality. Because all four members share the same 249 outer rows by
protocol, this reduces estimator variance rather than adding independent data, and the
seed spread reported alongside says by how much.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, best_f1_threshold, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402

MIN_POSITIVES = 10


def bagged_threshold(truth, scores, bags, seed, floors=(None, 0.9)):
    """Median argmax-F1 over bootstrap resamples.

    The council scores are a mean of four within-split rank vectors, which compresses the
    score distribution toward the middle. A single argmax-F1 cut on that distribution is
    placed differently from the cut on any member, which is why the unbagged council
    scored below its own members. Applying the same variance reduction that worked for the
    single model is the consistent move.
    """
    from dxaqc.model import best_f1_threshold as _best
    from evaluate_organizer_dataset import binary_metrics as _bm
    rng = np.random.default_rng(seed)
    n = len(truth)
    draws = []
    for _ in range(bags):
        pick = rng.integers(0, n, n)
        if len(np.unique(truth[pick])) < 2:
            continue
        best, chosen = -1.0, float(np.nextafter(1., np.inf))
        for floor in floors:
            for threshold in np.unique(np.concatenate(([0.0], np.sort(np.unique(scores[pick]))))):
                predicted = (scores[pick] >= threshold).astype(int)
                metrics = _bm(truth[pick], predicted)
                if floor is not None and metrics['specificity'] < floor:
                    continue
                if metrics['f1'] > best:
                    best, chosen = metrics['f1'], float(threshold)
        draws.append(chosen)
    return float(np.median(draws)) if draws else float(_best(truth, scores))


def parse_scores(frame):
    out = []
    for value in frame:
        if isinstance(value, str):
            parsed = json.loads(value)
        elif isinstance(value, dict):
            parsed = value
        else:
            parsed = {}
        out.append(parsed if isinstance(parsed, dict) else {})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bags', type=int, default=200,
                        help='Bootstrap resamples for the council threshold; 0 disables bagging')
    parser.add_argument('--seed', type=int, default=17)
    # Kept as a raw string: type=Path would convert "name=path" wholesale and break the split.
    parser.add_argument('--member', action='append', required=True, metavar='NAME=OOF.csv',
                        help='Out-of-fold CSV from one seed run, repeatable')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    members = {}
    for item in args.member:
        name, sep, path = item.partition('=')
        if not sep or not path:
            parser.error(f'--member expects NAME=OOF.csv, got {item!r}')
        table = pd.read_csv(path).sort_values('index').reset_index(drop=True)
        if len(table) != len(baseline) or not np.array_equal(table['index'].to_numpy(),
                                                             baseline['index'].to_numpy()):
            raise ValueError(f'{name}: rows are not aligned with the baseline')
        members[name] = table
        print(f'{name}: {table.shape}', flush=True)
    if len(members) < 2:
        parser.error('A council needs at least two members')

    from train import build_table
    import hashlib
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

    member_scores = {name: parse_scores(table['criterion_scores']) for name, table in members.items()}
    member_thresholds = {name: parse_scores(table['criterion_thresholds']) for name, table in members.items()}
    # Council score: mean of the member scores for each criterion. Every member score is
    # already a within-split rank in [0, 1], so the scales are commensurable.
    council = []
    for i in range(len(baseline)):
        merged = {}
        for name in member_scores:
            for key, value in member_scores[name][i].items():
                merged.setdefault(key, []).append(float(value))
        council.append({k: float(np.mean(v)) for k, v in merged.items()})

    groups = np.array([group_of(r) for r in baseline['predicted_region']])
    studies = baseline['study'].astype(str).to_numpy()
    args.output.mkdir(parents=True)
    candidates = {name: members[name].copy() for name in members}
    candidates['council'] = baseline.copy()
    audit_log = []

    for fold in sorted(baseline.fold.unique()):
        held = set(studies[baseline['fold'].to_numpy() == fold])
        train_mask = ~np.isin(studies, list(held))
        test_mask = ~train_mask
        for anatomy in ('spine', 'hip'):
            tr = np.flatnonzero(train_mask & (groups == anatomy))
            te = np.flatnonzero(test_mask & (groups == anatomy))
            if not len(te):
                continue
            accumulated, thresholds = {}, {}
            for criterion in CRITERIA[anatomy]:
                target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                known = np.isfinite(target[tr])
                column = np.array([council[i].get(criterion, np.nan) for i in range(len(baseline))], float)
                if known.sum() < 5 or len(np.unique(target[tr][known].astype(int))) < 2:
                    thresholds[criterion] = float(np.nextafter(1., np.inf))
                    accumulated[criterion] = column[te]
                    continue
                truth = target[tr][known].astype(int)
                if args.bags > 0:
                    threshold = bagged_threshold(truth, column[tr][known], args.bags,
                                                 args.seed + 100 + 7 * int(fold))
                else:
                    threshold = float(best_f1_threshold(truth, column[tr][known]))
                thresholds[criterion] = threshold
                accumulated[criterion] = column[te]
                audit_log.append({'fold': int(fold), 'criterion': criterion,
                                  'positives_in_training': int(truth.sum()),
                                  'threshold': round(threshold, 4)})
            for position, row_index in enumerate(te):
                criterion_scores = {k: float(v[position]) for k, v in accumulated.items()}
                probability = float(1 - np.prod([1 - p for p in criterion_scores.values()]))
                decision = decide(anatomy, probability, criterion_scores, features[row_index],
                                  float(np.nextafter(1., np.inf)), thresholds)
                for key, value in {'quality_score': probability,
                                   'quality_pred': decision['quality'],
                                   'violation_type': official_violation_type(decision['violations']),
                                   'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                                   'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                                   'criterion_thresholds': json.dumps(thresholds, sort_keys=True),
                                   'quality_threshold': decision['quality_threshold'],
                                   'decision_version': 'research-seed-council-1',
                                   'decision_reason': decision['decision_reason']}.items():
                    candidates['council'].at[row_index, key] = value
        print(f'Completed outer fold {fold}', flush=True)

    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=2000),
                         'screen': compare(args.baseline, path, args.labels)}
        print(f'{name:14} {reports[name]["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)

    f1 = {n: reports[n]['metrics']['overall_quality']['f1'] for n in reports}
    member_values = [v for k, v in f1.items() if k != 'council']
    (args.output / 'evaluation.json').write_text(json.dumps(
        {'protocol': __doc__, 'variants': reports, 'members': sorted(members),
         'member_mean_f1': float(np.mean(member_values)) if member_values else None,
         'member_sd_f1': float(np.std(member_values, ddof=1)) if len(member_values) > 1 else None,
         'council_minus_member_mean': (f1['council'] - float(np.mean(member_values))) if member_values else None,
         'selection_audit': audit_log,
         'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
         'code_sha256': sha256(Path(__file__)),
         'clinical_validation': False, 'model_affects_decision': False,
         'limitations': ['All members share the same 249 outer rows by protocol, so this reduces estimator variance '
                         'rather than adding independent data.',
                         'Member scores are within-split ranks, which makes averaging the scales commensurable.',
                         'The operating point is chosen on training studies only.',
                         'Exploratory; no automatic release promotion.']},
        ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()