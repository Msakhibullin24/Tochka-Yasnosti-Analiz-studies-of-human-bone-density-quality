"""Split an existing blind DXA review package into audit and error-review queues.

The audit queue is sampled without using QC labels or model decisions. The
development queue contains remaining study groups with binary QC mistakes.
Both browser packages stay blind; only the parent selection ledger records the
reason for selecting development cases. These are previously inspected DXA
studies, so the audit queue is an anatomy reference, not a new QC test set.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_key(case: dict) -> str:
    path = PurePosixPath(case['path_to_file'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe organiser review path')
    parts = path.parts
    if parts and parts[0] == 'Исследования':
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError('Expected organiser review path below a study directory')
    return '/'.join(parts)


def select_cases(reference: dict, oof_rows: list[dict], *, audit_studies: int = 25,
                 seed: int = 20260929) -> tuple[list[dict], list[dict], dict]:
    cases = reference['cases']
    if reference.get('review_package_version') != 4 or not cases:
        raise ValueError('A nonempty blind anatomy review v4 package is required')
    by_path = {_source_key(case): case for case in cases}
    if len(by_path) != len(cases) or len({case['image_uid'] for case in cases}) != len(cases):
        raise ValueError('Ambiguous source paths or image UIDs')
    groups = {key: key.split('/', 1)[0] for key in by_path}
    if any('/' not in key for key in by_path):
        raise ValueError('Source path lacks a study directory')
    if any(len({case['study_group'] for key, case in by_path.items() if groups[key] == group}) != 1
           for group in set(groups.values())):
        raise ValueError('DICOM study groups disagree within an input directory')
    if not 0 < audit_studies < len(set(groups.values())):
        raise ValueError('Audit study count must be between 1 and the number of groups')
    indexed = {}
    for row in oof_rows:
        path = row.get('source_path', '')
        if path in indexed or path not in by_path or row.get('study') != groups[path]:
            raise ValueError('OOF paths or study groups do not match the blind package')
        if row.get('quality_true') not in ('0', '1') or row.get('quality_pred') not in ('0', '1'):
            raise ValueError('OOF quality labels must be binary')
        indexed[path] = row
    if not indexed:
        raise ValueError('No matching OOF rows')
    population = sorted(set(groups.values()))
    audit_groups = set(random.Random(seed).sample(population, audit_studies))
    audit = [case for case in cases if groups[_source_key(case)] in audit_groups]
    development = [case for case in cases
                   if groups[_source_key(case)] not in audit_groups
                   and _source_key(case) in indexed
                   and indexed[_source_key(case)]['quality_true'] != indexed[_source_key(case)]['quality_pred']]
    if not audit or not development:
        raise ValueError('Both review queues must contain cases')
    reasons = Counter(('false_negative' if indexed[_source_key(case)]['quality_true'] == '1'
                       else 'false_positive') for case in development)
    ledger = {
        'seed': seed, 'audit_study_groups': audit_studies,
        'audit_cases': len(audit), 'development_cases': len(development),
        'development_reasons': dict(reasons),
        'audit_image_uids': [case['image_uid'] for case in audit],
        'development_image_uids': [case['image_uid'] for case in development],
        'scope': 'Audit selected without QC outcomes; development selected from prior OOF errors. '
                 'No queue is a new patient-independent QC test set.',
    }
    return audit, development, ledger


def _publish(browser_source: Path, reference: dict, cases: list[dict],
             expected_files: dict[str, str], output: Path) -> None:
    output.mkdir()
    (output/'images').mkdir()
    payload = copy.deepcopy(reference)
    payload['cases'] = cases
    for case in cases:
        relative = PurePosixPath(case['image'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'images':
            raise ValueError('Unsafe image path in review template')
        source = (browser_source/case['image']).resolve()
        if not source.is_relative_to(browser_source) or not source.is_file():
            raise ValueError('Review image escapes its source package')
        if expected_files.get(case['image']) != digest(source):
            raise ValueError('Review image differs from source package manifest')
        destination = output/case['image']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    assets = Path(__file__).parent/'anatomy_review'
    html = (assets/'index.html').read_text(encoding='utf-8')
    if html.count('__REVIEW_DATA__') != 1:
        raise ValueError('Review page template is incompatible')
    inline = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    (output/'index.html').write_text(html.replace('__REVIEW_DATA__', inline), encoding='utf-8')
    shutil.copy2(assets/'review.js', output/'review.js')
    (output/'template.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps({
        'results_sha256': payload['results_sha256'], 'cases': len(cases),
        'files_sha256': {str(path.relative_to(output)): digest(path)
                         for path in sorted(output.rglob('*')) if path.is_file()},
        'scope': 'Blind local review; no QC predictions or labels in browser package',
    }, indent=2)+'\n', encoding='utf-8')


def prepare(package: Path, oof: Path, output: Path, *, audit_studies: int = 25,
            seed: int = 20260929) -> dict:
    package, oof, output = package.resolve(), oof.resolve(), output.resolve()
    if not package.is_dir() or not oof.is_file() or output.exists() or output.is_relative_to(package):
        raise ValueError('Use an existing review package and OOF, and a new output outside the package')
    reference = json.loads((package/'template.json').read_text(encoding='utf-8'))
    source_manifest = json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    expected_files = source_manifest['files_sha256']
    if expected_files.get('template.json') != digest(package/'template.json'):
        raise ValueError('Review template differs from source package manifest')
    with oof.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    audit, development, ledger = select_cases(reference, rows, audit_studies=audit_studies, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.priority-anatomy-', dir=output.parent))
    try:
        _publish(package, reference, audit, expected_files, temporary/'audit')
        _publish(package, reference, development, expected_files, temporary/'development')
        ledger.update(source_package_sha256=digest(package/'template.json'), oof_sha256=digest(oof))
        (temporary/'selection.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {k: ledger[k] for k in ('audit_cases', 'development_cases', 'development_reasons')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--oof', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--audit-studies', type=int, default=25)
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.package, args.oof, args.output,
                                 audit_studies=args.audit_studies, seed=args.seed), ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'PRIORITY_REVIEW_FAILED: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
