"""Describe complementary errors in aligned OOF; no ensemble fitting or promotion.

Stored criterion threshold votes are audited separately from physical decisions.
The audit cannot establish training provenance or patient independence from CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.model import CRITERIA, REGIONS, group_of  # noqa: E402
from evaluate_organizer_dataset import binary_metrics, sha256  # noqa: E402

IDENTITY = ('index', 'repeat', 'fold', 'study', 'source_path', 'true_region',
            'predicted_region', 'quality_true')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def binary(value):
    if value not in ('0', '1'):
        raise ValueError('Invalid binary label or decision')
    return int(value)


def finite_number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a numeric score')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Non-finite score or threshold')
    return number


def load_aligned(labels_path: Path, experts: dict[str, Path]):
    if len(experts) < 2:
        raise ValueError('At least two named experts are required')
    labels = [r for r in read_csv(labels_path) if r.get('quality_class') in ('0', '1')]
    paths = [r['first_source_path'] for r in labels]
    if not labels or len(set(paths)) != len(paths):
        raise ValueError('Empty labels or duplicate reference paths')
    parsed, reference = {}, None
    for name, path in experts.items():
        rows = read_csv(path)
        if len(rows) != len(labels) or len({r['source_path'] for r in rows}) != len(rows):
            raise ValueError('Each expert must cover every labelled image exactly once')
        lookup = {r['source_path']: r for r in rows}
        if set(lookup) != set(paths):
            raise ValueError('Expert paths differ from reference')
        rows = [lookup[p] for p in paths]
        folds = {}
        for index, (row, label) in enumerate(zip(rows, labels)):
            if (row['index'] != str(index) or row['repeat'] != '0'
                    or row['study'] != label['study_key']
                    or row['true_region'] != label['region']
                    or row['quality_true'] != label['quality_class']):
                raise ValueError('OOF identity or truth differs from reference')
            if row['true_region'] not in REGIONS or row['predicted_region'] not in REGIONS:
                raise ValueError('Unsupported anatomical region')
            if not row['study'] or not row['fold']:
                raise ValueError('Missing study or fold')
            if folds.setdefault(row['study'], row['fold']) != row['fold']:
                raise ValueError('Study is split across folds')
            row['_quality'] = binary(row['quality_pred'])
            score = finite_number(row['quality_score'])
            if not 0 <= score <= 1:
                raise ValueError('Quality score outside [0,1]')
            row['_scores'] = json.loads(row['criterion_scores'])
            row['_thresholds'] = json.loads(row['criterion_thresholds'])
            row['_states'] = json.loads(row['criterion_states'])
            expected = set(CRITERIA[group_of(row['predicted_region'])])
            for field in ('_scores', '_thresholds', '_states'):
                if not isinstance(row[field], dict) or set(row[field]) != expected:
                    raise ValueError('Criterion fields differ from predicted region contract')
            for criterion in expected:
                score = finite_number(row['_scores'][criterion])
                if not 0 <= score <= 1:
                    raise ValueError('Criterion score outside [0,1]')
                finite_number(row['_thresholds'][criterion])
                state = row['_states'][criterion]
                if not isinstance(state, dict) or state.get('status') not in ('pass', 'fail', 'undetermined'):
                    raise ValueError('Invalid criterion state')
        identity = [tuple(row[k] for k in IDENTITY) for row in rows]
        if reference is not None and identity != reference:
            raise ValueError('Experts differ in held-out folds or routing')
        reference = identity
        parsed[name] = rows
    return labels, parsed


def summarize(truth: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    """NaN means unavailable, never a negative vote. Pair counts use common coverage."""
    result = {'reference_n': len(truth), 'reference_positives': int(truth.sum()),
              'experts': {}, 'pairs': {}}
    for name, pred in predictions.items():
        available = np.isfinite(pred)
        result['experts'][name] = {
            'evaluated': int(available.sum()), 'unavailable': int((~available).sum()),
            'unavailable_positives': int(((~available) & (truth == 1)).sum()),
            'metrics_on_evaluated': binary_metrics(truth[available], pred[available].astype(int))
                                   if available.any() else None}
    for left, right in combinations(predictions, 2):
        a, b = predictions[left], predictions[right]
        common = np.isfinite(a) & np.isfinite(b)
        y, a, b = truth[common], a[common], b[common]
        ea, eb = a != y, b != y
        positive = y == 1
        negative = y == 0
        correlation = (float(np.corrcoef(ea.astype(float), eb.astype(float))[0, 1])
                       if len(y) and ea.any() and not ea.all() and eb.any() and not eb.all()
                       else None)
        result['pairs'][f'{left}__{right}'] = {
            'left': left, 'right': right, 'common_n': len(y),
            'common_positives': int(positive.sum()), 'common_negatives': int(negative.sum()),
            'disagreements': int((a != b).sum()), 'both_wrong': int((ea & eb).sum()),
            'both_false_negative': int((positive & (a == 0) & (b == 0)).sum()),
            'both_false_positive': int((negative & (a == 1) & (b == 1)).sum()),
            'left_fn_detected_by_right': int((positive & (a == 0) & (b == 1)).sum()),
            'right_fn_detected_by_left': int((positive & (a == 1) & (b == 0)).sum()),
            'left_fp_rejected_by_right': int((negative & (a == 1) & (b == 0)).sum()),
            'right_fp_rejected_by_left': int((negative & (a == 0) & (b == 1)).sum()),
            'error_correlation': correlation}
    common = np.logical_and.reduce([np.isfinite(p) for p in predictions.values()])
    y = truth[common]
    matrix = np.column_stack([p[common] for p in predictions.values()])
    result['all_experts_common'] = {
        'n': len(y), 'excluded_unavailable': int((~common).sum()),
        'positives': int(y.sum()),
        'unanimous_false_negative': int(((y == 1) & (matrix == 0).all(axis=1)).sum()),
        'unanimous_false_positive': int(((y == 0) & (matrix == 1).all(axis=1)).sum()),
        'disagreement_images': int((matrix.min(axis=1) != matrix.max(axis=1)).sum()) if len(y) else 0,
    }
    controls = {'any_positive': matrix.max(axis=1), 'all_positive': matrix.min(axis=1)} if len(y) else {}
    if len(y) and len(predictions) % 2:
        controls['majority'] = (matrix.sum(axis=1) > len(predictions) / 2).astype(int)
    result['fixed_voting_controls_on_common'] = {
        name: binary_metrics(y, vote.astype(int)) for name, vote in controls.items()}
    # Deliberately do not fit weights or compute a label-selected "oracle" ensemble.
    return result


def audit(labels_path: Path, experts: dict[str, Path]) -> dict:
    labels, rows = load_aligned(labels_path, experts)
    truth = np.array([binary(r['quality_class']) for r in labels])
    overall = summarize(truth, {n: np.array([r['_quality'] for r in rs]) for n, rs in rows.items()})
    criteria = {}
    for group, names in CRITERIA.items():
        for criterion in names:
            indices = []
            for i, label in enumerate(labels):
                if group_of(label['region']) == group and label.get(criterion) not in ('', None):
                    binary(label[criterion])
                    indices.append(i)
            y = np.array([binary(labels[i][criterion]) for i in indices], dtype=int)
            raw_votes, final_votes = {}, {}
            for name, rs in rows.items():
                raw, final = [], []
                for i in indices:
                    row = rs[i]
                    if criterion not in row['_scores']:
                        raw.append(np.nan)
                        final.append(np.nan)
                        continue
                    raw.append(int(float(row['_scores'][criterion]) >= float(row['_thresholds'][criterion])))
                    status = row['_states'][criterion]['status']
                    final.append(1 if status == 'fail' else 0 if status == 'pass' else np.nan)
                raw_votes[name], final_votes[name] = np.array(raw), np.array(final)
            criteria[criterion] = {'classifier_threshold_votes': summarize(y, raw_votes),
                                   'final_criterion_decisions': summarize(y, final_votes)}
    return {'protocol': __doc__, 'labels_sha256': sha256(labels_path),
            'code_sha256': sha256(Path(__file__)),
            'experts': {n: {'path': str(p), 'sha256': sha256(p)} for n, p in experts.items()},
            'images': len(labels), 'studies': len({r['study_key'] for r in labels}),
            'overall_quality': overall, 'by_criterion': criteria,
            'clinical_validation': False, 'model_affects_decision': False,
            'weights_fitted': False,
            'limitations': [
                'Previously inspected internal OOF; no new independent test or measured ensemble gain.',
                'Matching folds and identities do not prove model training or selection provenance.',
                'Patient independence is not established by study identities.',
                'Shared routing and physical measurements can force shared final errors.',
                'Threshold votes are diagnostic classifier signals, not validated physical findings.',
                'Counts corrected by another expert are potential complementarity, not achievable ensemble gains.',
                'Fixed voting controls are post-hoc descriptive benchmarks, not new clinical model evaluations.',
                'Subset metrics explicitly exclude unavailable votes; coverage and unavailable positives are reported.',
                'Error correlations are descriptive; no confidence intervals or independence guarantees.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--expert', action='append', required=True, metavar='NAME=OOF_CSV')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experts = {}
    for value in args.expert:
        name, separator, path = value.partition('=')
        if not separator or not name or not path or name in experts:
            parser.error('Each expert must have a unique NAME=OOF_CSV')
        experts[name] = Path(path)
    if args.output.exists():
        parser.error('Choose a new output path to preserve previous evidence')
    report = audit(args.labels, experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'images': report['images'], 'studies': report['studies'],
                      'overall_shared_errors': report['overall_quality']['all_experts_common']}))


if __name__ == '__main__':
    main()
