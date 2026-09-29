"""Verify that workflow 1.11.4 only removes forbidden axis types on DXA batches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


BASE = Path('data/specialists/open-source-20260928')
CHECKS = ('path_to_study', 'study_uid', 'image_uid', 'anatomical_region',
          'processing_status', 'quality_prob', 'failure_code', 'laterality')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(old: Path, new: Path, cohort: str) -> dict:
    with (old/'results_extended.csv').open(newline='', encoding='utf-8-sig') as stream:
        previous = {row['path_to_file']: row for row in csv.DictReader(stream)}
    with (new/'results_extended.csv').open(newline='', encoding='utf-8-sig') as stream:
        current = {row['path_to_file']: row for row in csv.DictReader(stream)}
    if set(previous) != set(current) or len(previous) != len(current):
        raise ValueError(f'Input path set differs for {cohort}')
    changed = []
    for path in sorted(previous):
        before, after = previous[path], current[path]
        if any(before.get(key) != after.get(key) for key in CHECKS):
            raise ValueError(f'Unrelated output field changed for {cohort}: {path}')
        if after['processing_status'] == 'Success' and 'Не выравнена ось позвоночника' in after['violation_type']:
            angle = json.loads(after['measurements']).get('spine_abs_angle_deg')
            if angle is None or abs(angle) <= 5:
                raise ValueError(f'Forbidden axis type remains in {cohort}: {path}')
        if before['quality_class'] == after['quality_class'] and before['violation_type'] == after['violation_type']:
            continue
        old_types = before['violation_type'].split(';') if before['violation_type'] else []
        new_types = after['violation_type'].split(';') if after['violation_type'] else []
        angle = json.loads(after['measurements']).get('spine_abs_angle_deg')
        if (old_types != ['Не выравнена ось позвоночника'] + new_types
                and sorted(old_types) != sorted(['Не выравнена ось позвоночника'] + new_types)):
            raise ValueError(f'Change other than axis removal in {cohort}: {path}')
        if angle is None or abs(angle) > 5 or after['quality_class'] != str(int(bool(new_types))):
            raise ValueError(f'Changed row violates axis policy in {cohort}: {path}')
        changed.append({'path_sha256': hashlib.sha256(path.encode()).hexdigest(),
                        'angle_deg': round(float(angle), 3),
                        'quality_changed': before['quality_class'] != after['quality_class'],
                        'other_type_retained': bool(new_types)})
    old_summary = json.loads((old/'summary.json').read_text())
    new_summary = json.loads((new/'summary.json').read_text())
    if new_summary['files'] != len(current) or old_summary['files'] != len(previous):
        raise ValueError(f'Summary count differs in {cohort}')
    return {'files': len(current), 'success': new_summary['success'], 'failure': new_summary['failure'],
            'violations': new_summary['violations'],
            'table_format_valid': json.loads((new/'submission_validation.json').read_text())['valid'],
            'quality_changes': sum(item['quality_changed'] for item in changed),
            'type_changes': len(changed), 'changed_cases': changed,
            'old_results_sha256': digest(old/'results_extended.csv'),
            'new_results_sha256': digest(new/'results_extended.csv'),
            'new_summary_sha256': digest(new/'summary.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cohorts = {name: compare(args.base/f'workflow-1-11-3-{name}',
                             args.base/f'workflow-1-11-4-{name}', name)
               for name in ('organiser', 'bmd', 'vfa')}
    report = {'version': '1.11.4', 'docker_image_id': 'sha256:b657d40e3cbe582811b0d88b2d8395a019084f3a0233ae50bf461c49d283dbf2',
              'profile_sha256': digest(Path('competition/models/workflow_1_11_4/profile.json')),
              'scope': 'Offline technical comparison, not clinical accuracy',
              'cohorts': cohorts, 'clinical_validation': False, 'requirements_complete': False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: {field: value for field, value in cohort.items() if field in
                      ('files', 'success', 'failure', 'violations', 'quality_changes', 'type_changes', 'table_format_valid')}
                      for key, cohort in cohorts.items()}, ensure_ascii=False, indent=2))
