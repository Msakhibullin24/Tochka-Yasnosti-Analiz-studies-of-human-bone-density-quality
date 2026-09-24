"""Verify labelled image identity before fitting or comparing DXA models.

The label CSV contains hashes from an older pixel normalisation. Identity and
leakage checks therefore use freshly decoded pixels, while reporting any
historical hash drift instead of treating it as a new image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

from dxaqc.dicom_io import read_any

REQUIRED = {'first_source_path', 'study_key', 'pixel_sha256', 'rows', 'columns'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sources(sources: list[tuple[str, Path, Path]], reader=read_any) -> dict:
    """Fail on missing files, changed geometry, or decoded copies across groups.

    Each tuple is (source name, label CSV, image root). Source names namespace
    study IDs; copies across different sources are always rejected.
    """
    if not sources or len({name for name, _, _ in sources}) != len(sources):
        raise ValueError('Sources must have distinct nonempty names')
    seen_paths: set[tuple[str, str]] = set()
    seen_pixels: dict[str, tuple[str, str, str]] = {}
    entries: list[dict] = []
    summaries: list[dict] = []
    for name, labels_path, root in sources:
        if not name.strip():
            raise ValueError('Source name cannot be blank')
        root = root.resolve()
        with labels_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not REQUIRED <= rows[0].keys():
            raise ValueError(f'{name}: labels are empty or required columns are missing')
        accepted = 0
        for line, row in enumerate(rows, 2):
            quality = (row.get('quality_class') or '').strip()
            if not quality:
                continue  # unreviewed candidate; never sent to training
            if quality not in ('0', '1'):
                raise ValueError(f'{name}:{line}: quality_class must be 0, 1, or blank')
            rel = row['first_source_path']
            path = (root / rel).resolve()
            if not rel or Path(rel).is_absolute() or not path.is_relative_to(root):
                raise ValueError(f'{name}:{line}: image path escapes dataset root')
            if not path.is_file():
                raise ValueError(f'{name}:{line}: image is missing: {rel}')
            if (name, rel) in seen_paths:
                raise ValueError(f'{name}:{line}: source path repeated: {rel}')
            seen_paths.add((name, rel))
            study = row['study_key'].strip()
            if not study:
                raise ValueError(f'{name}:{line}: study_key is missing')
            # Organiser DICOM contain malformed UIDs. They are unrelated to
            # decoded pixel identity and otherwise flood a 249-image audit.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
                image = reader(path)
            try:
                shape = (int(row['rows']), int(row['columns']))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{name}:{line}: invalid labelled image size') from exc
            if image.pixels.shape != shape:
                raise ValueError(f'{name}:{line}: decoded image size changed: {rel}')
            pixel_hash = image.pixel_sha256
            earlier = seen_pixels.get(pixel_hash)
            if earlier and (earlier[0] != name or earlier[1] != study):
                raise ValueError(f'{name}:{line}: decoded pixel copy crosses data sources or study groups: '
                                 f'{earlier[0]}/{earlier[2]} and {name}/{rel}')
            seen_pixels.setdefault(pixel_hash, (name, study, rel))
            entries.append({'source': name, 'path': rel, 'study_sha256': hashlib.sha256(study.encode()).hexdigest(),
                            'pixel_sha256_current': pixel_hash,
                            'historical_hash_changed': pixel_hash != row['pixel_sha256']})
            accepted += 1
        summaries.append({'name': name, 'labels_sha256': digest(labels_path),
                          'labelled_images': accepted})
    groups = Counter((entry['source'], entry['study_sha256']) for entry in entries)
    return {'schema_version': 1, 'sources': summaries, 'labelled_images': len(entries),
            'study_groups': len(groups), 'current_unique_pixels': len(seen_pixels),
            'within_study_pixel_copies': len(entries) - len(seen_pixels),
            'historical_hash_changed': sum(entry['historical_hash_changed'] for entry in entries),
            'patient_disjointness_proven': False,
            'identity_protocol': 'freshly decoded pixels; source-namespaced study groups',
            'entries': entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, metavar='NAME=LABELS=ROOT',
                        help='Repeat for organiser and each external training source')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources = []
    for spec in args.source:
        try:
            name, labels, root = spec.split('=', 2)
        except ValueError as exc:
            parser.error(f'Invalid --source {spec!r}; expected NAME=LABELS=ROOT')
        sources.append((name, Path(labels), Path(root)))
    report = inspect_sources(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'entries'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
