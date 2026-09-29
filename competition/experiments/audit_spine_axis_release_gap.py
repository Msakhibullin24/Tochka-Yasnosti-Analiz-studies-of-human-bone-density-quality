"""Join frozen DXA spine-axis OOF artifacts without changing the release.

The research and release predictions have different CV folds. This is a
case-level failure audit, not a paired model comparison or clinical validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'competition/reports/quality_review_delivery_1_11_oof_2026_09_27.csv'
CANDIDATE = ROOT / 'docs/competition/dxa_tabpfn_spine_axis_2026_09_28_oof.csv'
GEOMETRY = ROOT / 'docs/competition/spine_axis_geometry_audit_2026_09_27.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(cases: list[dict], key: str) -> dict:
    tn = sum(not c['target'] and not c[key] for c in cases)
    fp = sum(not c['target'] and c[key] for c in cases)
    fn = sum(c['target'] and not c[key] for c in cases)
    tp = sum(c['target'] and c[key] for c in cases)
    return {'tn_fp_fn_tp': [tn, fp, fn, tp],
            'f1': 2 * tp / max(1, 2 * tp + fp + fn),
            'sensitivity': tp / max(1, tp + fn),
            'specificity': tn / max(1, tn + fp)}


def audit(release: Path, candidate: Path, geometry: Path) -> dict:
    with release.open(newline='', encoding='utf-8-sig') as stream:
        release_rows = [row for row in csv.DictReader(stream) if row['true_region'] == 'spine']
    with candidate.open(newline='', encoding='utf-8-sig') as stream:
        candidate_rows = list(csv.DictReader(stream))
    geometry_rows = json.loads(geometry.read_text())['cases']
    if len(release_rows) != len(candidate_rows) or len(geometry_rows) != len(release_rows):
        raise ValueError('Spine cohort size differs across artifacts')
    by_index = {int(row['index']): row for row in release_rows}
    by_path = {row['source_path']: row for row in geometry_rows}
    if len(by_index) != len(release_rows) or len(by_path) != len(geometry_rows):
        raise ValueError('Duplicate case identity')

    cases = []
    same_fold = 0
    for research in candidate_rows:
        index = int(research['index'])
        release_row = by_index[index]
        geo = by_path[release_row['source_path']]
        if hashlib.sha256(release_row['study'].encode()).hexdigest() != research['study_sha256']:
            raise ValueError(f'Study mismatch at index {index}')
        if int(research['target']) != int(geo['axis_violation']):
            raise ValueError(f'Criterion label mismatch at index {index}')
        state = json.loads(release_row['criterion_states'])['spine_axis']
        if state['basis'] not in ('measured_axis_angle', 'supervised_untyped_review') or state['limit_deg'] != 5.0:
            raise ValueError(f'Unexpected official axis rule at index {index}')
        angle = float(state['angle_deg'])
        if abs(angle - abs(float(geo['original_angle_deg']))) > 0.011:
            raise ValueError(f'Angle differs across artifacts at index {index}')
        official = int(state['status'] == 'fail')
        measured = int(angle > 5.0)
        same_fold += release_row['fold'] == research['fold']
        cases.append({'case_index': index, 'study_sha256': research['study_sha256'],
                      'target': int(research['target']), 'official_pred': official,
                      'measured_pred': measured, 'official_basis': state['basis'],
                      'angle_deg': round(angle, 3),
                      'tabpfn_pred': int(research['tabpfn_pred']),
                      'baseline_head_pred': int(research['baseline_pred']),
                      'local_segment_max_deg': geo['segment_max_angle_deg'],
                      'numbered_candidate_deg': geo['numbered_angle_deg'],
                      'release_fold': int(release_row['fold']),
                      'research_fold': int(research['fold'])})
    cases.sort(key=lambda case: case['case_index'])
    if len(cases) != 99 or sum(c['target'] for c in cases) != 10:
        raise ValueError('Unexpected organizer spine cohort')
    missed = [c for c in cases if c['target'] and not c['official_pred']]
    return {
        'scope': 'Historical organizer DXA OOF case-level release gap audit',
        'source_sha256': {str(path.relative_to(ROOT)): digest(path)
                          for path in (release, candidate, geometry)},
        'images': len(cases), 'positive_labels': sum(c['target'] for c in cases),
        'same_cv_fold_count': same_fold,
        'different_cv_fold_count': len(cases) - same_fold,
        'official_axis': matrix(cases, 'official_pred'),
        'strict_measured_axis': matrix(cases, 'measured_pred'),
        'research_tabpfn_head': matrix(cases, 'tabpfn_pred'),
        'research_baseline_head': matrix(cases, 'baseline_head_pred'),
        'official_false_negatives_at_or_below_5_deg': sum(c['angle_deg'] <= 5 for c in missed),
        'official_axis_fail_below_threshold': [c for c in cases if c['official_pred'] and not c['measured_pred']],
        'tabpfn_positive_among_official_false_negatives': sum(c['tabpfn_pred'] for c in missed),
        'baseline_positive_among_official_false_negatives': sum(c['baseline_head_pred'] for c in missed),
        'missed_positive_cases': missed,
        'official_false_positive_cases': [c for c in cases if not c['target'] and c['official_pred']],
        'limitations': [
            'The binary organizer criterion label is not a measured angle reference.',
            'The research and release OOF folds differ; model metrics are not a paired-fold comparison.',
            'The cohort was previously inspected and is not an independent test.',
            'No independently verified vertebral landmarks or angle ground truth are available.',
        ],
        'clinical_validation': False, 'release_modified': False,
        'requirements_complete': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=RELEASE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--geometry', type=Path, default=GEOMETRY)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.release, args.candidate, args.geometry)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('images', 'same_cv_fold_count',
           'official_axis', 'research_tabpfn_head', 'official_false_negatives_at_or_below_5_deg',
           'tabpfn_positive_among_official_false_negatives')}, ensure_ascii=False, indent=2))
