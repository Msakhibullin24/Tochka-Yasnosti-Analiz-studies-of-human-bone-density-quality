"""Compare deployed QC answers and count geometry availability, not accuracy."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def answers(rows):
    def key(row):
        fields = tuple(row[k] for k in ('image_uid', 'quality_class', 'violation_type', 'quality_prob', 'processing_status'))
        states = json.loads(row['criterion_states'])
        discrete = {code: {k: v for k, v in state.items() if k not in ('criterion_score', 'angle_deg')}
                    for code, state in states.items()}
        return (*fields, json.dumps(discrete, sort_keys=True))
    return Counter(key(row) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'current', 'output'):
        parser.add_argument('--' + name, type=Path, required=True, help='Batch output directory' if name != 'output' else 'JSON output')
    args = parser.parse_args()
    old, new = [json.loads((p / 'deployment_protocol.json').read_text()) for p in (args.previous, args.current)]
    for name in ('bank_sha256', 'reference_bank_sha256', 'decoder_sha256'):
        if old['profile'][name] != new['profile'][name]:
            raise ValueError('This is not a display-only comparison: QC models differ')
    paths = [p / 'results_extended.csv' for p in (args.previous, args.current)]
    a, b = map(read_rows, paths)
    if answers(a) != answers(b):
        raise AssertionError('Deployed QC answers or discrete criterion states changed')
    previous = {r['path_to_file']: r for r in a}
    if set(previous) != {r['path_to_file'] for r in b}:
        raise ValueError('Raw numerical comparison requires identical source paths')
    score_delta, angle_delta = 0., 0.
    for row in b:
        before = json.loads(previous[row['path_to_file']]['criterion_states'])
        after = json.loads(row['criterion_states'])
        for code in before:
            for field in ('criterion_score', 'angle_deg'):
                x, y = before[code].get(field), after[code].get(field)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    if field == 'criterion_score': score_delta = max(score_delta, abs(x - y))
                    else: angle_delta = max(angle_delta, abs(x - y))
    unique = [r for r in b if not r.get('duplicate_of')]
    hips, spine, missing = [], [], []
    for row in unique:
        anatomy = json.loads(row['learned_anatomy'])
        if 'neck_roi_assessment' in anatomy:
            hips.append(anatomy)
            if anatomy['neck_roi_assessment']['status'] != 'proposed':
                missing.append({'reason': anatomy['neck_roi_assessment'].get('reason'),
                    'quality_class': row['quality_class'], 'violation_type': row['violation_type'],
                    'anatomical_accuracy_established': False})
        else:
            spine.append(anatomy)
    report = {'images': len(b), 'unique_inferences': len(unique), 'qc_answers_and_discrete_criterion_states_identical': True,
        'maximum_raw_criterion_score_abs_delta': score_delta, 'maximum_axis_angle_abs_delta_degrees': angle_delta,
        'continuous_values_identical': score_delta == angle_delta == 0,
        'comparison_scope': 'same full-data fitted QC heads, display geometry refinement; not accuracy on training images',
        'previous_csv_sha256': digest(paths[0]), 'current_csv_sha256': digest(paths[1]),
        'hip_unique_images': len(hips), 'hip_roi_proposed': sum(h['neck_roi_assessment']['status'] == 'proposed' for h in hips),
        'hip_unavailable': missing, 'spine_unique_images': len(spine),
        'spine_roi_proposals': sum(len(s['roi_proposals']) for s in spine),
        'spine_density_roi_proposals': sum(len(s['density_based_roi']['proposals']) for s in spine),
        'spine_contour_fallbacks': sum(p.get('density_boundary_unavailable', False) for s in spine for p in s['roi_proposals']),
        'anatomical_accuracy_evaluation': False, 'code_sha256': digest(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
