"""Prepare a blind, case-selected hip rotation review from saved OOF predictions.

The selected set is for error analysis only. Its class balance is artificial and
must never be reported as a new performance estimate or used as a sealed test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

from build_release_review import archive_blinded_packet, write_blinded_packet
from dxaqc.dicom_io import read_dxa
from dxaqc.model import VIOLATION_LABEL
from evaluate_organizer_dataset import sha256


HIP_OBSERVATIONS = {
    'rotation_state': ('optimal', 'under_rotated', 'over_rotated', 'unassessable'),
    'lesser_trochanter_contour': ('slight', 'large', 'absent', 'unassessable'),
    'greater_trochanter_visible': ('yes', 'no', 'unassessable'),
    'femoral_neck_visible': ('yes', 'no', 'unassessable'),
    'ischium_visible': ('yes', 'no', 'unassessable'),
}


def select(oof_rows: list[dict], labels: dict[str, dict], controls_per_class: int = 10) -> list[dict]:
    if controls_per_class < 0:
        raise ValueError('controls_per_class must be nonnegative')
    if len({row['source_path'] for row in oof_rows}) != len(oof_rows):
        raise ValueError('OOF contains duplicate source paths')
    candidates = []
    folds: dict[str, str] = {}
    for row in oof_rows:
        label = labels.get(row['source_path'])
        if (label is None or row['repeat'] != '0' or row['study'] != label['study_key']
                or row['true_region'] != label['region']
                or row['quality_true'] != label['quality_class']):
            raise ValueError('OOF identity or reference quality differs')
        if folds.setdefault(row['study'], row['fold']) != row['fold']:
            raise ValueError('Study occurs in multiple held-out folds')
        if not label['region'].startswith('hip_') or label['hip_position_rotation'] not in ('0', '1'):
            continue
        actual = label['hip_position_rotation'] == '1'
        predicted = VIOLATION_LABEL['hip_position_rotation'] in row['violation_type'].split(';')
        kind = ('true_positive' if predicted else 'false_negative') if actual else (
            'false_positive' if predicted else 'true_negative')
        candidates.append({'source_path': row['source_path'], 'region': label['region'],
                           'kind': kind, 'study': row['study'], 'fold': row['fold']})
    selected = [item for item in candidates if item['kind'] in ('false_negative', 'false_positive')]
    for kind in ('true_positive', 'true_negative'):
        controls = [item for item in candidates if item['kind'] == kind]
        controls.sort(key=lambda item: hashlib.sha256(item['source_path'].encode()).hexdigest())
        preferred = []
        for side, quota in (('hip_left', (controls_per_class + 1) // 2),
                            ('hip_right', controls_per_class // 2)):
            preferred.extend([item for item in controls if item['region'] == side][:quota])
        remaining = [item for item in controls if item not in preferred]
        selected.extend(preferred + remaining[:max(0, controls_per_class - len(preferred))])
    return selected


def prepare(oof_path: Path, labels_path: Path, dataset: Path, output: Path,
            controls_per_class: int = 10) -> dict:
    if output.exists():
        raise ValueError('Choose a new review output directory')
    with oof_path.open(encoding='utf-8-sig', newline='') as stream:
        oof_rows = list(csv.DictReader(stream))
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        labels = {row['first_source_path']: row for row in csv.DictReader(stream)
                  if row.get('quality_class') in ('0', '1')}
    if len(oof_rows) != len(labels):
        raise ValueError('OOF does not cover exactly the labelled images')
    chosen = select(oof_rows, labels, controls_per_class)
    if not chosen:
        raise ValueError('No hip cases selected')
    cases = []
    for item in chosen:
        source = (dataset / item['source_path']).resolve()
        if not source.is_relative_to(dataset.resolve()) or not source.is_file():
            raise ValueError('Source DICOM is missing or escapes the dataset')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*', category=UserWarning)
            image = read_dxa(source)
        cases.append({'pixel_sha256': image.pixel_sha256, 'source_paths': [item['source_path']],
                      'reference_region': item['region']})
    output.mkdir(parents=True)
    write_blinded_packet(cases, dataset, output, read_dxa, extra_review_fields=HIP_OBSERVATIONS)
    (output / 'blind/HIP_ROTATION_GUIDE.txt').write_text(
        'Оцените каждый снимок независимо, не открывая исходные метки и ответы модели.\n'
        'rotation_state: optimal — малый вертел слегка выступает; under_rotated — выступает чрезмерно; '
        'over_rotated — не виден из-за избыточной ротации; unassessable — оценить нельзя.\n'
        'lesser_trochanter_contour: slight, large, absent или unassessable. '
        'Для отсутствующего из-за обрезки вертела используйте unassessable и укажите причину в comment.\n'
        'Видимость большого вертела, шейки и седалищной кости: yes, no или unassessable. '
        'Заполните также quality и применимые hip_position_rotation/hip_roi_coverage.\n'
        'При uncertain/unassessable поясните причину в comment. PNG без оверлеев; '
        'ключевые точки размечаются отдельно в COCO-пакете анатомии.\n', encoding='utf-8')
    archive_sha256, archive_files = archive_blinded_packet(output)
    mapping = json.loads((output / 'blind_mapping.json').read_text())
    private = [{**mapping[index], 'selection_kind': item['kind'], 'study': item['study'],
                'fold': item['fold']} for index, item in enumerate(chosen)]
    (output / 'selection_private.json').write_text(json.dumps(private, ensure_ascii=False, indent=2) + '\n')
    summary = {'scope': 'case-selected internal OOF error review, not a population metric or independent test',
               'ground_truth_ready': False, 'oof_sha256': sha256(oof_path),
               'labels_sha256': sha256(labels_path), 'images': len(chosen),
               'selection_counts': dict(sorted(Counter(item['kind'] for item in chosen).items())),
               'controls_per_class': controls_per_class, 'blind_archive_sha256': archive_sha256,
               'blind_archive_files': archive_files}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls-per-class', type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(prepare(args.oof, args.labels, args.dataset, args.output,
                             args.controls_per_class), ensure_ascii=False))


if __name__ == '__main__':
    main()
