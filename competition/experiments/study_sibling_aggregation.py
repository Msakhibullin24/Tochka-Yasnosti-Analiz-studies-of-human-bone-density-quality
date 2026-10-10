"""E20: aggregate the two hips of one study, which the data says are the same measurement.

The 249 labelled rows are only 100 studies, and 72 of them carry a spine, a left hip and a
right hip frame from one acquisition session. Every image is currently scored in isolation,
which throws away the fact that the two hips are not independent observations.

The measurement says they are not independent. Left and right hip scores inside one study
correlate at rho 0.62 for hip_position_rotation and rho 0.70 for hip_roi_coverage, both with
p below 1e-4, while a control that permutes the study assignment of one side collapses to
rho -0.14 and 0.06. The spine, by contrast, correlates with neither hip at all, every rho
below 0.15 with a control that looks identical, so cross-region aggregation is not merely
ineffective, there is nothing in it. Only the hip pair qualifies.

If the two hip scores share a true component and carry partly independent noise, then
averaging them raises the signal-to-noise ratio of the ranking. That is variance reduction
on the decision variable, not new information, and it is the mechanism this tests.

The decision layer, the routing and the measured spine axis are the shipped ones, reused
verbatim from the council pipeline, because a cheaper reimplementation was tried first and
silently scored F1 0.372 instead of 0.617: dropping decide() and the quality threshold changes
what is being measured. The weight-zero arm here reproduces the published council F1, and the
script fails loudly if it does not.

Two properties keep it honest. The blend weight is fixed at one half in advance, because with
two equally reliable measurements the minimum-variance combination is the equal one, so
nothing is tuned and there is no weight to overfit. And the split is by study, verified: no
study has frames in two outer folds, so a held-out frame may borrow its held-out sibling's
score without ever seeing a label. Only scores move, never labels.

A weight sweep is reported as a diagnostic curve so the shape of the response is visible, but
the decision is taken at the pre-registered 0.5 and the sweep is not used to pick it.
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
from dxaqc.model import CRITERIA, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import evaluate  # noqa: E402
from experiments.seed_council import bagged_threshold  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402
import hashlib  # noqa: E402

HIP_CRITERIA = {'hip_position_rotation', 'hip_roi_coverage'}
# Pre-registered. Equal weights are the minimum-variance blend for two equally reliable
# measurements, so this is a choice, not a tuned parameter.
W_FIXED = 0.5
SWEEP = (0.0, 0.25, W_FIXED, 0.75, 1.0)


def parse_scores(frame: pd.DataFrame) -> list[dict]:
    """Tolerant of NaN: a spine row has no hip criteria and pandas leaves those cells empty."""
    return [json.loads(v) if isinstance(v, str) and v.strip() else {} for v in frame['criterion_scores']]


def sibling_map(frame: pd.DataFrame) -> dict[int, int]:
    out: dict[int, int] = {}
    for _, g in frame.groupby('study'):
        left = g.index[g.true_region == 'hip_left'].tolist()
        right = g.index[g.true_region == 'hip_right'].tolist()
        if left and right:
            out[left[0]] = right[0]
            out[right[0]] = left[0]
    return out


def blend(column: np.ndarray, sibling: dict[int, int], weight: float) -> np.ndarray:
    out = column.copy()
    for i, j in sibling.items():
        if np.isfinite(column[i]) and np.isfinite(column[j]):
            out[i] = (1.0 - weight) * column[i] + weight * column[j]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bags', type=int, default=200)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--member', action='append', required=True, metavar='NAME=OOF.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)

    from train import build_table
    from source_integrity import inspect_sources

    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    members = {}
    for item in args.member:
        name, sep, path = item.partition('=')
        if not sep or not path:
            parser.error(f'--member expects NAME=OOF.csv, got {item!r}')
        table = pd.read_csv(path).sort_values('index').reset_index(drop=True)
        if not np.array_equal(table['index'].to_numpy(), baseline['index'].to_numpy()):
            raise ValueError(f'{name}: rows are not aligned with the baseline')
        members[name] = table

    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps([(r['path'], r['pixel_sha256_current'])
                                           for r in audit['entries']]).encode()).hexdigest()
    features = [dict(f) for f in build_table(args.dataset, labels, HERE / '.cache', fingerprint)[0]]
    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            features[row.index]['spine_abs_angle_deg'] = json.loads(
                row.criterion_states).get('spine_axis', {}).get('angle_deg')

    score_tables = {k: parse_scores(v) for k, v in members.items()}
    n = len(baseline)
    council = []
    for i in range(n):
        merged: dict[str, list[float]] = {}
        for table in score_tables.values():
            for key, value in table[i].items():
                merged.setdefault(key, []).append(float(value))
        council.append({k: float(np.mean(v)) for k, v in merged.items()})

    frame = baseline.copy()
    frame['true_region'] = labels.region.values
    frame['study'] = labels.study_key.values
    studies = frame['study'].astype(str).to_numpy()
    groups = np.array([group_of(r) for r in frame['predicted_region']])
    regions = labels.region.to_numpy()
    if int((frame.groupby('study')['fold'].nunique() > 1).sum()):
        raise ValueError('split is not by study; sibling aggregation would leak')
    sib = sibling_map(frame)
    print(f'split by study verified; {len(sib)} hip frames have an opposite-hip sibling')

    def column_of(criterion: str, weight: float) -> np.ndarray:
        col = np.array([council[i].get(criterion, np.nan) for i in range(n)], float)
        return blend(col, sib, weight) if criterion in HIP_CRITERIA else col

    reports, audit_log = {}, []
    for weight in SWEEP:
        name = f'sibling_w{weight:.2f}'
        table = baseline.copy()
        for fold in sorted(frame.fold.unique()):
            held = set(studies[frame['fold'].to_numpy() == fold])
            train_mask = ~np.isin(studies, list(held))
            test_mask = ~train_mask
            for anatomy in ('spine', 'hip'):
                tr = np.flatnonzero(train_mask & (groups == anatomy))
                te = np.flatnonzero(test_mask & (groups == anatomy))
                if not len(te):
                    continue
                accumulated, thresholds = {}, {}
                for criterion in CRITERIA[anatomy]:
                    col = column_of(criterion, weight)
                    target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
                    known = np.isfinite(target[tr])
                    if known.sum() < 5 or len(np.unique(target[tr][known].astype(int))) < 2:
                        thresholds[criterion] = float(np.nextafter(1., np.inf))
                        accumulated[criterion] = col[te]
                        continue
                    truth = target[tr][known].astype(int)
                    thr = (bagged_threshold(truth, col[tr][known], args.bags, args.seed + 100 + 7 * int(fold))
                           if args.bags > 0 else float(np.nan))
                    thresholds[criterion] = thr
                    accumulated[criterion] = col[te]
                    audit_log.append({'variant': name, 'fold': int(fold), 'criterion': criterion,
                                      'positives_in_training': int(truth.sum()),
                                      'threshold': round(float(thr), 4)})
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
                                       'decision_version': f'research-{name}',
                                       'decision_reason': decision['decision_reason']}.items():
                        table.at[row_index, key] = value
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=2000),
                         'screen': compare(args.baseline, path, args.labels)}
        print(f'{name:18} F1={reports[name]["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)

    ref = json.loads(Path('data/gpu-runs/council-bagged/evaluation.json').read_text())
    ref_f1 = ref['variants']['council']['metrics']['overall_quality']['f1']
    zero_f1 = reports['sibling_w0.00']['metrics']['overall_quality']['f1']
    if abs(zero_f1 - ref_f1) > 1e-9:
        raise ValueError(f'weight-zero arm gives {zero_f1}, council published {ref_f1}; '
                         'this script is not measuring the council')
    fixed = reports[f'sibling_w{W_FIXED:.2f}']['metrics']['overall_quality']['f1']

    (args.output / 'evaluation.json').write_text(json.dumps({
        'protocol': __doc__, 'pre_registered_weight': W_FIXED, 'sibling_pairs': len(sib),
        'split_by_study_verified': True, 'reproduces_council_at_zero_weight': True,
        'reference_council_f1': ref_f1, 'variants': reports,
        'delta_f1_at_pre_registered_weight': fixed - zero_f1,
        'selection_audit': audit_log,
        'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'clinical_validation': False, 'model_affects_decision': False}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(f'\nreference council F1 {ref_f1:.4f} | pre-registered w=0.5 F1 {fixed:.4f} | '
          f'delta {fixed - zero_f1:+.4f}')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
