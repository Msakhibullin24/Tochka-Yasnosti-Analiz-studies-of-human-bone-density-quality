"""Prepare a blind, case-selected lumbar-axis review with empty landmark tasks.

All positive axis labels and nearby negative controls are selected from the
existing internal cohort. This is for explaining errors, never for estimating
population performance or training before independent expert adjudication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

from build_release_review import archive_blinded_packet, label_taxonomy_conflict, write_blinded_packet
from dxaqc.dicom_io import read_dxa
from evaluate_organizer_dataset import sha256
from prepare_annotations import KEYPOINTS


AXIS_OBSERVATIONS = {
    'axis_assessability': ('assessable', 'partially_visible', 'unassessable'),
    'axis_pattern': ('straight', 'global_tilt', 'curved', 'unassessable'),
}


def select(labels: list[dict], stress: list[dict], controls: int = 10) -> list[dict]:
    if controls < 0 or len(stress) != len(labels):
        raise ValueError('Stress audit must cover the labelled images exactly once')
    by_index = {int(row['index']): row for row in stress}
    if len(by_index) != len(stress) or set(by_index) != set(range(len(labels))):
        raise ValueError('Stress audit has missing or duplicate label indexes')
    positive, negative = [], []
    for index, label in enumerate(labels):
        row = by_index[index]
        if row['study'] != label['study_key'] or row['region'] != label['region']:
            raise ValueError('Stress audit identity differs from source labels')
        if label['region'] != 'spine':
            continue
        if label['spine_axis'] in ('0', '1') and float(row['axis_label']) != int(label['spine_axis']):
            raise ValueError('Stress audit axis label differs from source labels')
        if label['spine_axis'] not in ('0', '1'):
            continue
        angle = float(row['global_abs_angle_deg'])
        if not math.isfinite(angle):
            raise ValueError('Spine angle is not finite')
        item = {'source_path': label['first_source_path'], 'study': label['study_key'],
                'region': 'spine', 'axis_label': label['spine_axis'],
                'measured_angle_deg': angle, 'label_taxonomy_conflict': label_taxonomy_conflict(label)}
        if label['spine_axis'] == '1':
            positive.append(item)
        elif not item['label_taxonomy_conflict']:
            negative.append(item)
    negative.sort(key=lambda item: (abs(item['measured_angle_deg'] - 5.0),
                                    hashlib.sha256(item['source_path'].encode()).hexdigest()))
    selected = positive + negative[:controls]
    if not positive or len(negative) < controls or len({item['source_path'] for item in selected}) != len(selected):
        raise ValueError('Insufficient unique spine cases for review')
    # Do not reveal positive/negative status through the order of blind case IDs.
    selected.sort(key=lambda item: hashlib.sha256(item['source_path'].encode()).hexdigest())
    return selected


def prepare(labels_path: Path, stress_path: Path, dataset: Path, output: Path,
            controls: int = 10) -> dict:
    if output.exists():
        raise ValueError('Choose a new review output directory')
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        labels = [row for row in csv.DictReader(stream) if row.get('quality_class') in ('0', '1')]
    with stress_path.open(encoding='utf-8-sig', newline='') as stream:
        stress = list(csv.DictReader(stream))
    chosen = select(labels, stress, controls)
    cases = []
    for item in chosen:
        source = (dataset / item['source_path']).resolve()
        if not source.is_relative_to(dataset.resolve()) or not source.is_file():
            raise ValueError('Source DICOM is missing or escapes the dataset')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*', category=UserWarning)
            image = read_dxa(source)
        cases.append({'pixel_sha256': image.pixel_sha256,
                      'source_paths': [item['source_path']], 'reference_region': 'spine'})
    if len({case['pixel_sha256'] for case in cases}) != len(cases):
        raise ValueError('Review selection contains duplicate decoded images')
    output.mkdir(parents=True)
    write_blinded_packet(cases, dataset, output, read_dxa,
                         extra_review_fields=AXIS_OBSERVATIONS)
    blind = output / 'blind'
    manifest = json.loads((blind / 'manifest.json').read_text(encoding='utf-8'))
    images = [{'id': index, 'file_name': case['image'], 'width': case['width'],
               'height': case['height']}
              for index, case in enumerate(manifest['cases'], 1)]
    template = {'images': images, 'annotations': [],
                'categories': [{'id': 1, 'name': 'lumbar_vertebrae',
                                'keypoints': KEYPOINTS['spine'], 'skeleton': []}]}
    for reviewer in ('A', 'B'):
        (blind / f'reviewer-{reviewer}-spine-keypoints.coco.json').write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (blind / 'SPINE_AXIS_GUIDE.txt').write_text(
        'Оценивайте снимки независимо, не открывая исходные метки и ответы модели.\n'
        'В CSV укажите официальный spine_axis (1/0/пусто), пригодность оценки и характер оси.\n'
        'В своей копии COCO отметьте центры L1–L4 и, если видимы, углы тел Th12–L5. '
        'Для невидимых точек используйте visibility=0; не достраивайте их по догадке. '
        'Если уровень позвонка не устанавливается, оставьте соответствующие точки невидимыми '
        'и объясните причину в comment CSV. Пустой annotations не означает норму.\n'
        'PNG в исходной геометрии; масштаб Y/X указан в manifest.json. '
        'В одном COCO экземпляр lumbar_vertebrae содержит 30 точек из categories.keypoints.\n'
        'Два COCO-файла — независимые задания, не готовая обучающая разметка. '
        'Расхождения точек и оценок надо согласовать до обучения.\n', encoding='utf-8')
    archive_sha, archive_files = archive_blinded_packet(output)
    mapping = json.loads((output / 'blind_mapping.json').read_text(encoding='utf-8'))
    private = [{**mapping[index], **item} for index, item in enumerate(chosen)]
    (output / 'selection_private.json').write_text(
        json.dumps(private, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'scope': 'case-selected internal axis review; not an independent test',
               'ground_truth_ready': False, 'labels_sha256': sha256(labels_path),
               'stress_sha256': sha256(stress_path), 'images': len(chosen),
               'axis_positive': sum(item['axis_label'] == '1' for item in chosen),
               'boundary_negative_controls': sum(item['axis_label'] == '0' for item in chosen),
               'blind_archive_sha256': archive_sha, 'blind_archive_files': archive_files}
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--stress', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls', type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(prepare(args.labels, args.stress, args.dataset, args.output,
                             args.controls), ensure_ascii=False))


if __name__ == '__main__':
    main()
