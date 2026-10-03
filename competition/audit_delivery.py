"""Independently verify a delivered DICOM folder against its batch outputs.

The source manifest is collected from the input folder, never from results.
This verifies engineering contracts; it cannot supply clinical ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from dxaqc.validate_results import validate

MEDICAL_FIELDS = ('study_uid', 'image_uid', 'anatomical_region', 'quality_class',
                  'violation_type', 'processing_status', 'quality_prob', 'measurements',
                  'criterion_states', 'projection_assessment', 'anatomical_checks_complete')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_manifest(root):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError('Independent input must be an extracted DICOM folder')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.dcm')
    if not files:
        raise ValueError('No source .dcm files')
    manifest = []
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Source DICOM escapes the input folder')
        manifest.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    return manifest


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    paths = [r['path_to_file'] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate output paths')
    return {row['path_to_file']: row for row in rows}


def compare_rows(before, after):
    if before.keys() != after.keys():
        raise ValueError('Compared batches have different input paths')
    return {'images': len(before), 'exact_medical_matches': sum(
        all(before[path].get(k, '') == after[path].get(k, '') for k in MEDICAL_FIELDS)
        for path in before), 'fields': list(MEDICAL_FIELDS)}


def audit(root, results, previous=None):
    manifest = source_manifest(root)
    rows = read_rows(results / 'results_extended.csv')
    summary = json.loads((results / 'summary.json').read_text())
    if Counter(rows.keys()) != Counter(item['path'] for item in manifest):
        raise ValueError('Output does not match independently collected input manifest')
    if summary['files'] != len(rows) or summary['success'] != sum(r['processing_status'] == 'Success' for r in rows.values()):
        raise ValueError('Summary differs from output table')
    strict = validate(results / 'submission.csv', competition=True)
    complete = validate(results / 'results_extended.csv', manifest=[r['path'] for r in manifest],
                        series=results / 'additional_series.zip', timing=results / 'timing.json')
    comparison = compare_rows(read_rows(previous / 'results_extended.csv'), rows) if previous else None
    return {'scope': __doc__, 'summary': summary,
            'independent_source_files': len(manifest),
            'input_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            'input_manifest_scope': 'all .dcm files in independently extracted organizer folder',
            'strict_validation': strict, 'full_validation': complete,
            'comparison': comparison,
            'requirements': json.loads((results / 'requirements.json').read_text()),
            'timing': json.loads((results / 'timing.json').read_text()),
            'artifact_sha256': {name: digest(results / name) for name in
                ('summary.json', 'requirements.json', 'timing.json', 'submission.csv',
                 'results_extended.csv', 'additional_series.zip')},
            'audit_code_sha256': digest(Path(__file__)), 'clinical_validation': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new audit report path')
    report = audit(args.input, args.results, args.previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'images': report['independent_source_files'],
                      'strict_valid': report['strict_validation']['valid'],
                      'full_valid': report['full_validation']['valid'],
                      'comparison': report['comparison']}, indent=2))
    if not report['strict_validation']['valid'] or not report['full_validation']['valid']:
        raise SystemExit(3)
