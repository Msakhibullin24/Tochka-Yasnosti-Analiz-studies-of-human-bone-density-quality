"""Replay a pinned DXA workflow against its saved extended result table.

The report contains counts and hashes only. Source DICOM paths and identifiers
stay local and are never written to the audit artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path

from dxaqc.pipeline import Analyzer, Options, assign_study_paths, process_files


JSON_COLUMNS = {
    'violation_scores', 'measurements', 'criterion_thresholds', 'projection_assessment',
    'anatomy_assessment', 'source_roi_assessment', 'criterion_states', 'review_reasons',
    'specialist_qc', 'specialist_outputs', 'requirement_checks', 'learned_anatomy',
    'joint_evidence',
}
IGNORED_COLUMNS = {'time_of_processing', 'explanation_png'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized(value, column: str):
    if value is None or value == '':
        return None
    if column in JSON_COLUMNS:
        return json.loads(value) if isinstance(value, str) else value
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def different_columns(expected: dict, actual: dict) -> list[str]:
    return sorted(
        key for key in expected.keys() | actual.keys()
        if key not in IGNORED_COLUMNS
        and normalized(expected.get(key), key) != normalized(actual.get(key), key)
    )


def replay(dataset: Path, reference: Path, profile: Path, singleton_count: int) -> dict:
    dataset = dataset.resolve()
    expected = list(csv.DictReader(reference.open(encoding='utf-8-sig', newline='')))
    if not expected or len({row['path_to_file'] for row in expected}) != len(expected):
        raise ValueError('Reference table has no rows or has duplicate paths')
    files = [(dataset / row['path_to_file']).resolve() for row in expected]
    if any(not path.is_relative_to(dataset) or not path.is_file() for path in files):
        raise ValueError('Reference paths are missing or escape the dataset root')
    os.environ['DXAQC_WORKFLOW_PROFILE'] = str(profile.resolve())
    analyzer = Analyzer()
    opts = Options(explanations=False, xlsx=False)
    with tempfile.TemporaryDirectory(prefix='dxa-replay-') as scratch, warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
        actual = process_files(files, dataset, Path(scratch), analyzer, opts)
        assign_study_paths(actual)
        single = [process_files([path], dataset, Path(scratch), analyzer, opts)[0]
                  for path in files[:singleton_count]]
    disagreements = Counter()
    rows_different = 0
    for old, new in zip(expected, actual, strict=True):
        fields = different_columns(old, new)
        if fields:
            rows_different += 1
            disagreements.update(fields)
    singleton_disagreements = Counter()
    singleton_rows_different = 0
    for batch, one in zip(actual, single, strict=False):
        fields = [field for field in different_columns(batch, one)
                  if field not in {'duplicate_of', 'path_to_study'}]
        if fields:
            singleton_rows_different += 1
            singleton_disagreements.update(fields)
    return {
        'profile_id': analyzer.workflow_profile['profile_id'],
        'profile_sha256': sha256(profile),
        'reference_sha256': sha256(reference),
        'rows_replayed': len(actual),
        'reference_mismatched_rows': rows_different,
        'reference_mismatched_fields': dict(sorted(disagreements.items())),
        'single_vs_batch_rows': len(single),
        'single_vs_batch_mismatched_rows': singleton_rows_different,
        'single_vs_batch_mismatched_fields': dict(sorted(singleton_disagreements.items())),
        'all_matched': rows_different == singleton_rows_different == 0,
        'scope': 'Technical replay of saved results; not a measurement of clinical accuracy.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--singletons', type=int, default=6)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.singletons < 0 or args.output.exists():
        parser.error('Use a nonnegative singleton count and a new output path')
    report = replay(args.dataset, args.reference, args.profile, args.singletons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['all_matched']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
