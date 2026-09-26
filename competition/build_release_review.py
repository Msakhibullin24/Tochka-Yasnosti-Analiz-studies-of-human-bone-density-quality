"""Build a local adjudication queue for binary alarms without an official type."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

from dxaqc.model import CRITERIA, REGION_LABEL, group_of


def label_taxonomy_conflict(label):
    """Describe contradictions in complete organiser labels without editing them."""
    region = label.get('region', '')
    quality = label.get('quality_class', '')
    if region not in ('spine', 'hip_left', 'hip_right') or quality not in ('0', '1'):
        return ''
    values = [label.get(key, '') for key in CRITERIA[group_of(region)]]
    if any(value not in ('0', '1') for value in values):
        return ''  # incomplete labels need a separate completeness review
    any_failed = any(value == '1' for value in values)
    if quality == '0' and any_failed:
        return 'normal_class_with_failed_criterion'
    if quality == '1' and not any_failed:
        return 'positive_class_without_failed_criterion'
    return ''


def group_untyped_rows(rows, pixel_hash_by_path, labels_by_pixel_hash, review_hashes=frozenset()):
    """Collapse SOP copies for alarms and optionally contradictory source labels."""
    groups = defaultdict(list)
    for row in rows:
        path = row.get('path_to_file')
        if not needs_review(row) and pixel_hash_by_path.get(path) not in review_hashes:
            continue
        if not path or path not in pixel_hash_by_path:
            raise ValueError(f"No normalized pixel hash for review result path: {path!r}")
        groups[pixel_hash_by_path[path]].append(row)

    cases = []
    for pixel_hash, copies in sorted(groups.items()):
        labels = labels_by_pixel_hash.get(pixel_hash, [])
        label_signatures = {
            (label.get('quality_class', ''), label.get('spine_coverage', ''),
             label.get('spine_axis', ''), label.get('spine_artifact', ''),
             label.get('hip_position_rotation', ''), label.get('hip_roi_coverage', ''))
            for label in labels
        }
        taxonomy_conflicts = sorted({reason for label in labels
                                     if (reason := label_taxonomy_conflict(label))})
        if len(label_signatures) > 1:
            label_status = 'conflicting_reference_labels'
        elif taxonomy_conflicts:
            label_status = 'inconsistent_reference_taxonomy'
        elif labels:
            label_status = 'matched_reference_label'
        else:
            label_status = 'reference_label_missing'
        first = copies[0]
        measurements = _json_object(first.get('measurements'))
        criterion_states = _json_object(first.get('criterion_states'))
        violation_scores = _json_object(first.get('violation_scores'))
        thresholds = _json_object(first.get('criterion_thresholds'))
        references = labels[0] if labels and len(label_signatures) == 1 else {}
        reference_region = references.get('region', '')
        if reference_region in ('spine', 'hip_left', 'hip_right'):
            applicable = CRITERIA[group_of(reference_region)]
        elif any(references.get(key) in ('0', '1') for key in CRITERIA['spine']):
            applicable = CRITERIA['spine']
        elif any(references.get(key) in ('0', '1') for key in CRITERIA['hip']):
            applicable = CRITERIA['hip']
        else:
            applicable = ()
        reference_criteria = [name for name in applicable if references.get(name) == '1']
        triggers = []
        if any(needs_review(row) for row in copies):
            triggers.append('binary_only_alarm')
        if taxonomy_conflicts:
            triggers.append('label_taxonomy_conflict')
        cases.append({
            'pixel_sha256': pixel_hash,
            'copy_count': len(copies),
            'source_paths': sorted(row['path_to_file'] for row in copies),
            'study_uids': sorted({row.get('study_uid', '') for row in copies}),
            'region': first.get('anatomical_region', ''),
            'reference_region': references.get('region', ''),
            'quality_prob': first.get('quality_prob', ''),
            'decision_reason': first.get('decision_reason', ''),
            'violation_type_status': first.get('violation_type_status', ''),
            'review_reasons': first.get('review_reasons', ''),
            'reference_status': label_status,
            'review_triggers': triggers,
            'source_label_conflicts': taxonomy_conflicts,
            'reference_quality_class': references.get('quality_class', ''),
            'reference_criteria': ';'.join(reference_criteria),
            'reference_comment': references.get('comment', ''),
            'spine_abs_angle_deg': measurements.get('spine_abs_angle_deg'),
            'violation_scores': violation_scores,
            'criterion_thresholds': thresholds,
            'criterion_states': criterion_states,
            'expert_review_status': 'pending',
            'expert_reviewer': '',
            'expert_review_date': '',
            'expert_quality_class': '',
            'expert_violation_types': '',
            'expert_comment': '',
        })
    return cases


def needs_review(row):
    """Include legacy untyped positives and V4 binary-only review alarms."""
    if row.get('processing_status') != 'Success' or (row.get('violation_type') or '').strip():
        return False
    return row.get('quality_class') == '1' or row.get('decision_reason') == 'binary_only_review'


def _json_object(value):
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError('Expected a JSON object in extended results')
    return parsed


def _read_normalized_pixels(path, read_dxa):
    # The full-batch report already records the organizer archive's known invalid UI metadata.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*', category=UserWarning)
        return read_dxa(path)


def write_blinded_packet(cases, dataset: Path, output: Path, read_dxa,
                         extra_review_fields: dict[str, tuple[str, ...]] | None = None):
    """Keep expert images and blank answers separate from prediction/label evidence."""
    extra_review_fields = extra_review_fields or {}
    base_fields = {'image_id', 'reviewer_id', 'reviewed', 'quality', 'comment',
                   *CRITERIA['spine'], *CRITERIA['hip']}
    if any(not name.isidentifier() or name in base_fields or not choices
           for name, choices in extra_review_fields.items()):
        raise ValueError('Invalid extra review field contract')
    blind = output / 'blind'
    images = blind / 'images'
    images.mkdir(parents=True)
    records, mapping = [], []
    for number, case in enumerate(cases, 1):
        source = (dataset / case['source_paths'][0]).resolve()
        if not source.is_relative_to(dataset.resolve()):
            raise ValueError('Review path escapes dataset')
        image = _read_normalized_pixels(source, read_dxa)
        if image.pixel_sha256 != case['pixel_sha256']:
            raise ValueError('Review pixels no longer match the grouped case')
        image_id = f'case-{number:03d}'
        reference_region = case.get('reference_region')
        if reference_region in ('spine', 'hip_left', 'hip_right'):
            region = 'spine' if reference_region == 'spine' else 'hip'
            region_source = 'reference'
        elif case.get('region') in (REGION_LABEL['spine'], REGION_LABEL['hip_right']):
            region = 'spine' if case['region'] == REGION_LABEL['spine'] else 'hip'
            region_source = 'model_unverified'
        else:
            raise ValueError('Cannot determine anatomical region for review case')
        path = images / f'{image_id}.png'
        if not cv2.imwrite(str(path), image.pixels):
            raise ValueError(f'Cannot write expert image: {path}')
        records.append({'image_id': image_id, 'image': f'images/{image_id}.png',
                        'region': region, 'region_source': region_source,
                        'width': int(image.pixels.shape[1]), 'height': int(image.pixels.shape[0]),
                        'pixel_mm_y': float(image.pixel_mm),
                        'pixel_mm_x': float(image.pixel_mm_x or image.pixel_mm),
                        'png_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        mapping.append({'image_id': image_id, 'pixel_sha256': case['pixel_sha256'],
                        'source_paths': case['source_paths']})
    (blind / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'ground_truth_ready': False, 'cases': records,
        'extra_review_fields': extra_review_fields,
        'instructions': 'Two independent experts; no model prediction or existing quality label is supplied.'
    }, ensure_ascii=False, indent=2) + '\n')
    fields = ['image_id', 'reviewer_id', 'reviewed', 'quality',
              *CRITERIA['spine'], *CRITERIA['hip'], *extra_review_fields, 'comment']
    for reviewer in ('A', 'B'):
        with (blind / f'reviewer-{reviewer}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({'image_id': record['image_id'], 'reviewed': 'false'})
    (blind / 'README.txt').write_text(
        'Слепая экспертная проверка DXA: изображения в исходной геометрии и два независимых бланка.\n'
        'Исходные оценки качества и прогнозы нарушений отсутствуют. 1 = нарушение, 0 = нет, пусто = неизвестно.\n'
        'После проверки укажите reviewer_id и reviewed=true. Изображения могут содержать персональные надписи; '
        'передавайте только в защищённом контуре.\n'
        'Масштаб Y/X и дополнительные поля с допустимыми значениями указаны в manifest.json. '
        'Заполненные бланки не становятся обучающими метками '
        'без проверки полноты и согласования расхождений.\n', encoding='utf-8')
    (output / 'blind_mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    return len(records)


def archive_blinded_packet(output: Path) -> tuple[str, int]:
    """Package only the blind directory; never include source paths or model answers."""
    blind = output / 'blind'
    files = sorted(path for path in blind.rglob('*') if path.is_file())
    archive = output / 'blind-review-packet.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(blind).as_posix(),
                                   date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, path.read_bytes())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'blind-review-packet.zip.sha256').write_text(
        f'{checksum}  {archive.name}\n', encoding='utf-8')
    return checksum, len(files)


def build(results_path: Path, labels_path: Path, dataset: Path, output: Path,
          include_label_conflicts: bool = False):
    from dxaqc.dicom_io import read_dxa

    with results_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        labels = list(csv.DictReader(stream))
    labels_by_hash = defaultdict(list)
    review_hashes = set()
    for label in labels:
        if label.get('first_source_path') and label.get('quality_class'):
            source = dataset / label['first_source_path']
            if source.is_file():
                pixel_hash = _read_normalized_pixels(source, read_dxa).pixel_sha256
                labels_by_hash[pixel_hash].append(label)
                if include_label_conflicts and label_taxonomy_conflict(label):
                    review_hashes.add(pixel_hash)

    untyped = [row for row in rows if needs_review(row)]
    pixel_hashes = {}
    paths_to_hash = ({row.get('path_to_file', '') for row in rows if row.get('processing_status') == 'Success'}
                     if include_label_conflicts else {row.get('path_to_file', '') for row in untyped})
    for path in sorted(paths_to_hash):
        if not path:
            raise ValueError('Untyped result is missing path_to_file')
        pixel_hashes[path] = _read_normalized_pixels(dataset / path, read_dxa).pixel_sha256
    cases = group_untyped_rows(rows if include_label_conflicts else untyped,
                               pixel_hashes, labels_by_hash, review_hashes)

    output.mkdir(parents=True, exist_ok=False)
    blinded_cases = write_blinded_packet(cases, dataset, output, read_dxa)
    archive_sha256, archive_files = archive_blinded_packet(output)
    fields = list(cases[0]) if cases else [
        'pixel_sha256', 'copy_count', 'source_paths', 'study_uids', 'region', 'quality_prob',
        'decision_reason', 'violation_type_status', 'review_reasons', 'reference_status',
        'review_triggers', 'source_label_conflicts', 'reference_region',
        'reference_quality_class', 'reference_criteria', 'reference_comment', 'spine_abs_angle_deg',
        'violation_scores', 'criterion_thresholds', 'criterion_states', 'expert_review_status',
        'expert_reviewer', 'expert_review_date', 'expert_quality_class', 'expert_violation_types',
        'expert_comment']
    with (output / 'review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else value
                             for key, value in case.items()})

    statuses = defaultdict(int)
    for case in cases:
        statuses[case['reference_status']] += 1
    summary = {
        'scope': 'local adjudication of binary-only alarms and optional source-label contradictions; no model decision or source label was modified',
        'result_rows': len(untyped),  # historical field: binary-only alarm rows
        'selected_result_rows': sum(case['copy_count'] for case in cases),
        'unique_pixel_groups': len(cases),
        'source_label_conflict_groups': sum('label_taxonomy_conflict' in case['review_triggers'] for case in cases),
        'blinded_expert_cases': blinded_cases,
        'blind_archive_sha256': archive_sha256,
        'blind_archive_files': archive_files,
        'copies_collapsed': sum(case['copy_count'] - 1 for case in cases),
        'reference_status_counts': dict(sorted(statuses.items())),
        'cases': cases,
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path, help='extended batch CSV')
    parser.add_argument('--labels', required=True, type=Path, help='existing labels CSV')
    parser.add_argument('--dataset', required=True, type=Path, help='root of the DICOM dataset')
    parser.add_argument('--output', required=True, type=Path, help='new local output directory')
    parser.add_argument('--include-label-conflicts', action='store_true',
                        help='also blind-review images whose complete source quality and criterion labels contradict')
    args = parser.parse_args()
    summary = build(args.results, args.labels, args.dataset, args.output,
                    include_label_conflicts=args.include_label_conflicts)
    print(json.dumps({key: value for key, value in summary.items() if key != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
