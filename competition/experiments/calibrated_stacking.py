"""Calibrated stacking of a frozen-encoder candidate onto the baseline OOF.

Every attempt to *learn* the criterion scores on 249 images lost to the baseline,
always through the same mechanism: the false-positive count exploded (baseline 34,
learned variants 71-117). The representation was never the bottleneck; the
threshold chosen by argmax-F1 on an ~18-study inner validation split is.

This script keeps both sets of scores strictly out-of-fold and instead fixes the
operating point:

  * blend weight between the baseline score and the candidate score, chosen on
    the training studies of each outer fold only;
  * optional specificity floor - pick, among thresholds whose training
    specificity clears the baseline's, the one with the best training F1;
  * noisy-OR aggregation and the unchanged decision layer.

No test-fold label is ever read while choosing a weight or a threshold.
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
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, best_f1_threshold, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def parse_scores(frame):
    out = []
    for value in frame:
        if isinstance(value, str):
            parsed = json.loads(value)
        elif isinstance(value, dict):
            parsed = value
        else:  # pandas turns missing JSON into NaN
            parsed = {}
        out.append(parsed if isinstance(parsed, dict) else {})
    return out


def noisy_or(scores):
    return float(1 - np.prod([1 - p for p in scores.values()]))


def pick_threshold(truth, values, floor=None):
    """Best F1 threshold, optionally among thresholds meeting a specificity floor."""
    if not len(truth):
        return float(np.nextafter(1., np.inf))
    best, best_threshold = -1.0, float(np.nextafter(1., np.inf))
    for threshold in np.unique(np.concatenate(([0.0], np.sort(np.unique(values))))):
        predicted = (values >= threshold).astype(int)
        metrics = binary_metrics(truth, predicted)
        if floor is not None and metrics['specificity'] < floor:
            continue
        if metrics['f1'] > best:
            best, best_threshold = metrics['f1'], float(threshold)
    return best_threshold


def pick_weight(truth, base, cand, floor):
    best, best_weight = -1.0, WEIGHTS[0]
    for weight in WEIGHTS:
        blended = (1 - weight) * base + weight * cand
        threshold = pick_threshold(truth, blended, floor)
        score = binary_metrics(truth, (blended >= threshold).astype(int))['f1']
        if score > best:
            best, best_weight = score, weight
    return best_weight


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    base = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    cand = pd.read_csv(args.candidate).sort_values('index').reset_index(drop=True)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    if len(base) != len(cand) or not np.array_equal(base['index'].to_numpy(), cand['index'].to_numpy()):
        raise ValueError('Baseline and candidate rows are not aligned')
    if set(base['repeat']) != {0} or base.groupby('study')['fold'].nunique().max() != 1:
        raise ValueError('Baseline splits a study or has multiple repeats')
    if not np.array_equal(base['study'].astype(str).to_numpy(), cand['study'].astype(str).to_numpy()):
        raise ValueError('Candidate study assignment differs from the baseline')
    audit_sources = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit_sources['entries']],
        ensure_ascii=False).encode()).hexdigest()

    features = [dict(f) for f in build_table(args.dataset, labels, HERE / '.cache',
                                             fingerprint)[0]]
    if len(features) != len(base):
        raise ValueError('Feature table length differs from the OOF rows')
    for row in base.itertuples():
        if row.predicted_region == 'spine':
            angle = json.loads(row.criterion_states).get('spine_axis', {}).get('angle_deg')
            features[row.Index]['spine_abs_angle_deg'] = angle

    base_scores, cand_scores = parse_scores(base['criterion_scores']), parse_scores(cand['criterion_scores'])
    groups = np.array([group_of(r) for r in base['predicted_region']])
    truths = {c: pd.to_numeric(labels[c], errors='coerce').to_numpy(float) for c in CRITERIA['spine'] | CRITERIA['hip']}
    modes = ('candidate_only', 'baseline_only') + tuple(f'spec{floor:g}' for floor in args.spec_floors)
    candidates = {name: base.copy() for name in modes}
    args.output.mkdir(parents=True)
    audit = []
    for fold in sorted(base.fold.unique()):
        held = set(base.loc[base.fold == fold, 'study'].astype(str))
        train_mask = ~base['study'].astype(str).isin(held).to_numpy()
        test_mask = ~train_mask
        group = groups[test_mask]
        test_positions = np.flatnonzero(test_mask)
        for anatomy in sorted(set(groups)):
            tr = np.flatnonzero(train_mask & (groups == anatomy))
            te = test_positions[groups[test_positions] == anatomy]
            if not len(te):
                continue
            for criterion in CRITERIA[anatomy]:
                target = truths[criterion]
                known = tr[np.isfinite(target[tr])]
                if not len(known):
                    raise ValueError(f'{criterion}: no training labels')
                b = np.array([base_scores[i].get(criterion, 0.0) for i in range(len(base))])
                c = np.array([cand_scores[i].get(criterion, 0.0) for i in range(len(base))])
                y = target[known].astype(int)
                # Operating points are learned on training studies only.
                base_threshold = pick_threshold(y, b[known])
                weight_only = pick_weight(y, b[known], c[known], None)
                base_metrics = binary_metrics(y, (b[known] >= base_threshold).astype(int))
                plan = {'candidate_only': (1.0, None), 'baseline_only': (0.0, None)}
                for floor in args.spec_floors:
                    plan[f'spec{floor:g}'] = (pick_weight(y, b[known], c[known], base_metrics['specificity'] * floor),
                                              base_metrics['specificity'] * floor)
                for mode in modes:
                    weight, floor = plan[mode]
                    train_blended = (1 - weight) * b[known] + weight * c[known]
                    threshold = pick_threshold(y, train_blended, floor)
                    blended = (1 - weight) * b + weight * c
                    scores = {criterion: blended[te]}
                    audit.append({'fold': int(fold), 'anatomy': anatomy, 'criterion': criterion,
                                  'mode': mode, 'weight': weight, 'threshold': threshold,
                                  'train_specificity_target': floor,
                                  'train_base_f1': base_metrics['f1'],
                                  'train_base_specificity': base_metrics['specificity'],
                                  'weight_only_selected': weight_only})
                    _apply(candidates[mode], features, te, criterion, blended, threshold,
                           anatomy, mode)
        print(f'Completed outer fold {fold}', flush=True)
    reports = {}
    for name, table in candidates.items():
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        reports[name] = {'metrics': evaluate(args.labels, path, repeats=200),
                         'screen': compare(args.baseline, path, args.labels)}
        print(f'{name:24} {reports[name]["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)
    (args.output / 'evaluation.json').write_text(json.dumps(
        {'protocol': __doc__, 'variants': reports, 'selection_audit': audit,
         'spec_floors': args.spec_floors, 'weights': WEIGHTS,
         'baseline_sha256': sha256(args.baseline), 'candidate_sha256': sha256(args.candidate),
         'labels_sha256': sha256(args.labels), 'code_sha256': sha256(Path(__file__)),
         'clinical_validation': False, 'model_affects_decision': False,
         'limitations': ['Scores are strictly out-of-fold; only weights and thresholds use training labels.',
                         'Blending cannot recover information absent from both score sets.',
                         'Reported deltas are inside the +/-0.08 cohort noise band.',
                         'Exploratory; no automatic release promotion.']},
        ensure_ascii=False, indent=2) + '\n')


def _apply(table, features, te, criterion, blended, threshold, anatomy, mode):
    """Recompute the decision for the test rows of one criterion and merge it."""
    for row_index in te:
        index = int(table.at[row_index, 'index'])
        existing = json.loads(table.at[row_index, 'criterion_scores']) if isinstance(
            table.at[row_index, 'criterion_scores'], str) else {}
        existing[criterion] = float(blended[row_index])
        thresholds = json.loads(table.at[row_index, 'criterion_thresholds']) if isinstance(
            table.at[row_index, 'criterion_thresholds'], str) else {}
        thresholds[criterion] = float(threshold)
        probability = noisy_or(existing)
        decision = decide(anatomy, probability, existing, features[index],
                          float(np.nextafter(1., np.inf)), thresholds)
        table.at[row_index, 'quality_score'] = probability
        table.at[row_index, 'quality_pred'] = decision['quality']
        table.at[row_index, 'violation_type'] = official_violation_type(decision['violations'])
        table.at[row_index, 'criterion_states'] = json.dumps(decision['criterion_states'], sort_keys=True)
        table.at[row_index, 'criterion_scores'] = json.dumps(existing, sort_keys=True)
        table.at[row_index, 'criterion_thresholds'] = json.dumps(thresholds, sort_keys=True)
        table.at[row_index, 'quality_threshold'] = decision['quality_threshold']
        table.at[row_index, 'decision_version'] = 'research-calibrated-stacking-1'
        table.at[row_index, 'decision_reason'] = decision['decision_reason']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec-floors', type=float, nargs='+', default=(1.0, 0.98, 0.95))
    args = parser.parse_args()
    if not args.spec_floors:
        parser.error('--spec-floors needs at least one value')
    with threadpool_limits(limits=4):
        run(args)
