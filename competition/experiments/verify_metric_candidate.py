"""Compare candidate DXA QC OOF to a frozen baseline without hiding tradeoffs.

Passing is only an internal research screen. It never establishes clinical
validation or patient-disjoint performance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_quality_metric_priorities import confusion, sha256
from dxaqc.model import SEED, VIOLATION_LABEL
from evaluate_organizer_dataset import evaluate

IDENTITY = ('index', 'repeat', 'fold', 'study', 'source_path', 'true_region', 'quality_true')
REGIONS = ('spine', 'hip_left', 'hip_right')


def read_oof(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {*IDENTITY, 'quality_pred', 'quality_score', 'predicted_region',
                'violation_type', 'criterion_states'}
    if not required <= set(table.columns):
        raise ValueError(f'Missing OOF columns: {sorted(required - set(table.columns))}')
    if table.duplicated(['repeat', 'index']).any():
        raise ValueError('Duplicate OOF image index')
    if set(table.repeat) != {0} or len(table) != 249:
        raise ValueError('Expected exactly one 249-image OOF repeat')
    if not table.quality_pred.isin([0, 1]).all() or not table.quality_true.isin([0, 1]).all():
        raise ValueError('Invalid binary labels or predictions')
    if not np.isfinite(table.quality_score).all() or not table.quality_score.between(0, 1).all():
        raise ValueError('Invalid quality probabilities')
    if table.source_path.duplicated().any() or set(table['index']) != set(range(249)):
        raise ValueError('Duplicate paths or incomplete image indices')
    if table.groupby('study')['fold'].nunique().max() != 1:
        raise ValueError('Study is split across test folds')
    if not table.true_region.isin(REGIONS).all() or not table.predicted_region.isin(REGIONS).all():
        raise ValueError('Invalid anatomical region')
    table = table.sort_values(['repeat', 'index']).reset_index(drop=True)
    types = table.violation_type.fillna('').astype(str)
    if ((table.quality_pred == 0) & types.ne('')).any() or ((table.quality_pred == 1) & types.eq('')).any():
        raise ValueError('Quality decision disagrees with violation type')
    allowed = set(VIOLATION_LABEL.values())
    if any(part not in allowed for value in types[types.ne('')] for part in value.split(';')):
        raise ValueError('Violation type is outside the organizer vocabulary')
    for row in table.itertuples():
        parts = str(row.violation_type).split(';') if pd.notna(row.violation_type) else []
        region_codes = ('spine_coverage', 'spine_axis', 'spine_artifact') if row.predicted_region == 'spine' else ('hip_position_rotation', 'hip_roi_coverage')
        if len(parts) != len(set(parts)) or any(p not in {VIOLATION_LABEL[k] for k in region_codes} for p in parts):
            raise ValueError('Duplicate type or type outside predicted region vocabulary')
        if VIOLATION_LABEL['spine_axis'] not in str(row.violation_type):
            continue
        state = json.loads(row.criterion_states)['spine_axis']
        angle = state.get('angle_deg')
        if row.true_region != 'spine' or not isinstance(angle, (int, float)) or not math.isfinite(angle) or abs(angle) <= 5:
            raise ValueError('Axis type violates the measured >5 degree rule')
    return table


def compare(baseline_path: Path, candidate_path: Path, labels_path: Path | None = None) -> dict:
    baseline = read_oof(baseline_path)
    candidate = read_oof(candidate_path)
    if not baseline[list(IDENTITY)].equals(candidate[list(IDENTITY)]):
        raise ValueError('Candidate and baseline image identities or folds differ')
    labels_path = labels_path or Path(__file__).resolve().parents[1] / 'labels/image_labels.csv'
    # Bind both inputs to the authoritative label table, not merely to each other.
    full_before = evaluate(labels_path, baseline_path, repeats=200)
    full_after = evaluate(labels_path, candidate_path, repeats=200)
    truth = baseline.quality_true.to_numpy(dtype=int)
    before = baseline.quality_pred.to_numpy(dtype=int)
    after = candidate.quality_pred.to_numpy(dtype=int)
    baseline_metrics = confusion(truth, before)
    candidate_metrics = confusion(truth, after)
    by_region = {}
    for region in REGIONS:
        indices = np.flatnonzero(baseline.true_region.eq(region).to_numpy())
        by_region[region] = {
            'baseline': confusion(truth[indices], before[indices]),
            'candidate': confusion(truth[indices], after[indices]),
        }
    groups = baseline.study.astype(str).to_numpy()
    group_rows = [np.flatnonzero(groups == study) for study in np.unique(groups)]
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(2000):
        sampled = np.concatenate([group_rows[i] for i in rng.integers(len(group_rows), size=len(group_rows))])
        deltas.append(confusion(truth[sampled], after[sampled])['f1'] -
                      confusion(truth[sampled], before[sampled])['f1'])
    ci = [float(v) for v in np.percentile(deltas, [2.5, 97.5])]
    reasons = []
    if candidate_metrics['f1'] <= baseline_metrics['f1']:
        reasons.append('overall_f1_not_higher')
    if candidate_metrics['sensitivity'] < baseline_metrics['sensitivity']:
        reasons.append('sensitivity_lower')
    if candidate_metrics['specificity'] < baseline_metrics['specificity']:
        reasons.append('specificity_lower')
    if candidate_metrics['tn_fp_fn_tp'][2] > baseline_metrics['tn_fp_fn_tp'][2]:
        reasons.append('additional_false_safe')
    for region, pair in by_region.items():
        if pair['candidate']['f1'] < pair['baseline']['f1']:
            reasons.append(f'{region}_f1_lower')
        for metric in ('sensitivity', 'specificity'):
            if pair['candidate'][metric] < pair['baseline'][metric]:
                reasons.append(f'{region}_{metric}_lower')
        for metric in ('roc_auc', 'average_precision'):
            if full_after['by_region'][region][metric] < full_before['by_region'][region][metric]:
                reasons.append(f'{region}_{metric}_lower')
    if ci[0] <= 0:
        reasons.append('paired_f1_ci_includes_nonimprovement')
    for metric in ('roc_auc', 'average_precision', 'precision', 'balanced_accuracy'):
        if full_after['overall_quality'][metric] < full_before['overall_quality'][metric]:
            reasons.append(f'{metric}_lower')
    # Brier includes ranking and probability calibration; lower is better.
    brier_before = float(np.mean((baseline.quality_score.to_numpy() - truth) ** 2))
    brier_after = float(np.mean((candidate.quality_score.to_numpy() - truth) ** 2))
    if brier_after > brier_before + 1e-12:
        reasons.append('brier_score_higher')
    if full_after['criterion_macro_f1'] < full_before['criterion_macro_f1']:
        reasons.append('criterion_macro_f1_lower')
    for criterion, before_metrics in full_before['by_criterion'].items():
        after_metrics = full_after['by_criterion'][criterion]
        for metric in ('f1', 'sensitivity', 'specificity'):
            if after_metrics[metric] < before_metrics[metric]:
                reasons.append(f'{criterion}_{metric}_lower')
        for metric in ('roc_auc', 'average_precision'):
            if before_metrics.get(metric) is not None:
                if after_metrics.get(metric) is None:
                    reasons.append(f'{criterion}_{metric}_unavailable')
                elif after_metrics[metric] < before_metrics[metric]:
                    reasons.append(f'{criterion}_{metric}_lower')
    return {
        'scope': 'Internal study-held-out OOF comparison, not clinical validation',
        'images': len(baseline), 'study_groups': len(group_rows),
        'baseline': baseline_metrics, 'candidate': candidate_metrics,
        'by_region': by_region,
        'overall_score_metrics': {
            'baseline': {**full_before['overall_quality'], 'brier_score': brier_before},
            'candidate': {**full_after['overall_quality'], 'brier_score': brier_after},
        },
        'by_criterion': {k: {'baseline': full_before['by_criterion'][k],
                            'candidate': full_after['by_criterion'][k]}
                         for k in full_before['by_criterion']},
        'criterion_macro_f1': {'baseline': full_before['criterion_macro_f1'],
                               'candidate': full_after['criterion_macro_f1']},
        'paired_delta_f1': candidate_metrics['f1'] - baseline_metrics['f1'],
        'paired_delta_f1_ci95': ci,
        'passes_internal_metric_screen': not reasons,
        'failure_reasons': reasons,
        'baseline_oof_sha256': sha256(baseline_path),
        'candidate_oof_sha256': sha256(candidate_path),
        'labels_sha256': sha256(labels_path),
        'screen_code_sha256': sha256(Path(__file__)),
        'clinical_validation': False,
        'limitations': [
            'Candidate and baseline are compared only on already inspected organizer DXA.',
            'Study disjointness does not prove patient disjointness in this anonymized export.',
            'Passing this screen does not establish anatomical or external accuracy.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, help='Authoritative organizer label CSV')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new report path')
    report = compare(args.baseline, args.candidate, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('baseline', 'candidate', 'paired_delta_f1',
                                             'paired_delta_f1_ci95', 'passes_internal_metric_screen',
                                             'failure_reasons')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
