"""Replay the strict 5-degree review guard on frozen historical OOF rows.

This changes no fitted model or CV split. The output is a posthoc estimate,
not independent validation of a newly packaged workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.joint_quality import constrain_review_codes
from dxaqc.model import group_of, official_violation_type
from evaluate_organizer_dataset import evaluate


def replay(labels: Path, source: Path, output: Path, report: Path) -> dict:
    if output.exists() or report.exists():
        raise ValueError('Choose new output paths for a reproducible replay')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    changed = []
    for row in rows:
        if row['decision_version'] != 'review-1' or row['decision_reason'] != 'supervised_type_review':
            continue
        states = json.loads(row['criterion_states'])
        axis = states.get('spine_axis', {})
        if axis.get('basis') != 'supervised_untyped_review' or axis.get('status') != 'fail':
            continue
        codes = [code for code, state in states.items() if state['status'] == 'fail']
        allowed, excluded = constrain_review_codes(codes, group_of(row['predicted_region']), axis.get('angle_deg'))
        if not excluded:
            continue
        axis.update(status='pass', basis='measured_axis_angle', clinical_validation=False)
        row['criterion_states'] = json.dumps(states, ensure_ascii=False, sort_keys=True)
        row['quality_pred'] = str(int(bool(allowed)))
        row['violation_type'] = official_violation_type(allowed)
        row['decision_reason'] = 'supervised_type_review' if allowed else 'supervised_normal_review'
        changed.append({'case_index': int(row['index']),
                        'study_sha256': hashlib.sha256(row['study'].encode()).hexdigest(),
                        'angle_deg': axis.get('angle_deg'), 'remaining_codes': allowed})
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = evaluate(labels, output, repeats=200)
    result = {'protocol': 'Posthoc strict measured-axis guard on frozen OOF review decisions',
              'changed': changed, 'metrics': metrics,
              'source_oof_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'corrected_oof_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest(),
              'clinical_validation': False, 'independent_validation': False,
              'limitations': ['No outer folds, weights or score thresholds were re-estimated.',
                              'The organizer cohort was previously inspected.',
                              'The fixed 5-degree rule lacks expert angle reference.']}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.labels, args.source, args.output, args.report)
    print(json.dumps({'changed': result['changed'],
                      'overall': result['metrics']['overall_quality']['tn_fp_fn_tp'],
                      'spine_axis': result['metrics']['by_criterion']['spine_axis']['tn_fp_fn_tp']},
                     ensure_ascii=False, indent=2))
