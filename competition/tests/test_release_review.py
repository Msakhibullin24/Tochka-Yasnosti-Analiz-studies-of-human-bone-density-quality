import csv
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from build_release_review import archive_blinded_packet, group_untyped_rows, label_taxonomy_conflict, needs_review, write_blinded_packet
from dxaqc.model import REGION_LABEL
from reconcile_reviews import reconcile


def test_release_review_collapses_pixel_copies_and_attaches_reference_without_editing():
    rows = [
        {'path_to_file': 'a.dcm', 'processing_status': 'Success', 'quality_class': '1',
         'violation_type': '', 'measurements': '{"spine_abs_angle_deg": 3.2}',
         'criterion_states': '{}', 'violation_scores': '{}', 'criterion_thresholds': '{}'},
        {'path_to_file': 'copy.dcm', 'processing_status': 'Success', 'quality_class': '1',
         'violation_type': '', 'measurements': '{"spine_abs_angle_deg": 3.2}',
         'criterion_states': '{}', 'violation_scores': '{}', 'criterion_thresholds': '{}'},
        {'path_to_file': 'typed.dcm', 'processing_status': 'Success', 'quality_class': '1',
         'violation_type': 'Некорректная укладка'},
    ]
    labels = {'same-pixels': [{'quality_class': '1', 'spine_axis': '1', 'spine_coverage': '0',
                               'spine_artifact': '0', 'comment': 'reference'}]}

    cases = group_untyped_rows(rows, {'a.dcm': 'same-pixels', 'copy.dcm': 'same-pixels'}, labels)

    assert len(cases) == 1
    assert cases[0]['copy_count'] == 2
    assert cases[0]['source_paths'] == ['a.dcm', 'copy.dcm']
    assert cases[0]['reference_status'] == 'matched_reference_label'
    assert cases[0]['reference_criteria'] == 'spine_axis'
    assert cases[0]['spine_abs_angle_deg'] == 3.2
    assert cases[0]['expert_review_status'] == 'pending'
    assert cases[0]['expert_violation_types'] == ''
    assert rows[0]['quality_class'] == rows[1]['quality_class'] == '1'


def test_release_review_surfaces_conflicting_pixel_reference_labels():
    row = {'path_to_file': 'a.dcm', 'processing_status': 'Success', 'quality_class': '1',
           'violation_type': '', 'measurements': '{}', 'criterion_states': '{}',
           'violation_scores': '{}', 'criterion_thresholds': '{}'}
    labels = {'same-pixels': [
        {'quality_class': '1', 'spine_axis': '1', 'spine_coverage': '0', 'spine_artifact': '0'},
        {'quality_class': '0', 'spine_axis': '0', 'spine_coverage': '0', 'spine_artifact': '0'},
    ]}

    cases = group_untyped_rows([row], {'a.dcm': 'same-pixels'}, labels)

    assert cases[0]['reference_status'] == 'conflicting_reference_labels'
    assert cases[0]['reference_quality_class'] == ''
    assert cases[0]['reference_criteria'] == ''


def test_binary_only_alarm_remains_available_for_expert_review():
    row = {'path_to_file': 'a.dcm', 'processing_status': 'Success', 'quality_class': '0',
           'violation_type': '', 'decision_reason': 'binary_only_review',
           'measurements': '{}', 'criterion_states': '{}',
           'violation_scores': '{}', 'criterion_thresholds': '{}'}
    assert needs_review(row)
    cases = group_untyped_rows([row], {'a.dcm': 'same-pixels'}, {})
    assert len(cases) == 1
    assert cases[0]['decision_reason'] == 'binary_only_review'


def test_complete_source_label_contradictions_join_blinded_review_without_changing_verdict():
    normal_with_failed_axis = {'region': 'spine', 'quality_class': '0',
                               'spine_coverage': '0', 'spine_axis': '1', 'spine_artifact': '0'}
    positive_without_type = {'region': 'spine', 'quality_class': '1',
                             'spine_coverage': '0', 'spine_axis': '0', 'spine_artifact': '0'}
    incomplete = {**positive_without_type, 'spine_axis': ''}
    assert label_taxonomy_conflict(normal_with_failed_axis) == 'normal_class_with_failed_criterion'
    assert label_taxonomy_conflict(positive_without_type) == 'positive_class_without_failed_criterion'
    assert label_taxonomy_conflict(incomplete) == ''

    row = {'path_to_file': 'source.dcm', 'processing_status': 'Success', 'quality_class': '0',
           'violation_type': '', 'anatomical_region': REGION_LABEL['spine'],
           'measurements': '{}', 'criterion_states': '{}', 'violation_scores': '{}',
           'criterion_thresholds': '{}'}
    labels = {'same-pixels': [normal_with_failed_axis]}
    assert group_untyped_rows([row], {'source.dcm': 'same-pixels'}, labels) == []
    cases = group_untyped_rows([row], {'source.dcm': 'same-pixels'}, labels, {'same-pixels'})
    assert len(cases) == 1
    assert cases[0]['review_triggers'] == ['label_taxonomy_conflict']
    assert cases[0]['reference_status'] == 'inconsistent_reference_taxonomy'
    assert cases[0]['reference_criteria'] == 'spine_axis'
    assert row['quality_class'] == '0'


def test_blinded_packet_omits_model_answers_and_preserves_case_mapping(tmp_path):
    dataset = tmp_path / 'dicom'
    dataset.mkdir()
    (dataset / 'source.dcm').write_bytes(b'source')
    case = {'pixel_sha256': 'same-pixels', 'source_paths': ['source.dcm'],
            'reference_region': 'spine',
            'reference_quality_class': '1', 'quality_prob': '.95'}
    image = SimpleNamespace(pixels=np.full((8, 12), 42, dtype=np.uint8),
                            pixel_sha256='same-pixels', pixel_mm=1.05, pixel_mm_x=.6)
    output = tmp_path / 'review'
    output.mkdir()

    assert write_blinded_packet([case], dataset, output, lambda _: image) == 1
    checksum, file_count = archive_blinded_packet(output)
    assert len(checksum) == 64 and file_count == 5
    with zipfile.ZipFile(output / 'blind-review-packet.zip') as archive:
        assert set(archive.namelist()) == {'README.txt', 'manifest.json', 'reviewer-A.csv',
                                           'reviewer-B.csv', 'images/case-001.png'}
        assert 'source.dcm' not in str(archive.read('manifest.json'))

    manifest = json.loads((output / 'blind/manifest.json').read_text())
    assert manifest['cases'][0]['pixel_mm_x'] == .6
    assert manifest['cases'][0]['pixel_mm_y'] == 1.05
    assert not any('quality_prob' in str(record) or 'reference_quality' in str(record)
                   for record in manifest['cases'])
    with (output / 'blind/reviewer-A.csv').open(encoding='utf-8-sig', newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['reviewed'] == 'false' and row['quality'] == '' and row['spine_axis'] == ''
    mapping = json.loads((output / 'blind_mapping.json').read_text())
    assert mapping[0]['source_paths'] == ['source.dcm']
    report = reconcile(output / 'blind', output / 'blind/reviewer-A.csv',
                       output / 'blind/reviewer-B.csv', output / 'agreement.json')
    assert report['counts'] == {'pending': 1, 'agreed': 0, 'requires_adjudication': 0}


def test_blinded_packet_rejects_unverified_region_or_changed_pixels(tmp_path):
    dataset = tmp_path / 'dicom'
    dataset.mkdir()
    (dataset / 'source.dcm').write_bytes(b'source')
    image = SimpleNamespace(pixels=np.full((4, 4), 42, dtype=np.uint8),
                            pixel_sha256='actual', pixel_mm=1.05, pixel_mm_x=.6)
    case = {'pixel_sha256': 'actual', 'source_paths': ['source.dcm']}
    with pytest.raises(ValueError, match='Cannot determine anatomical region'):
        write_blinded_packet([case], dataset, tmp_path / 'missing-region', lambda _: image)
    case['region'] = REGION_LABEL['spine']
    assert write_blinded_packet([case], dataset, tmp_path / 'fallback-region', lambda _: image) == 1
    fallback = json.loads((tmp_path / 'fallback-region/blind/manifest.json').read_text())
    assert fallback['cases'][0]['region_source'] == 'model_unverified'
    case['reference_region'] = 'spine'
    case['pixel_sha256'] = 'different'
    with pytest.raises(ValueError, match='pixels no longer match'):
        write_blinded_packet([case], dataset, tmp_path / 'changed-pixels', lambda _: image)


def test_optional_hip_observations_remain_blind_and_require_agreement(tmp_path):
    dataset = tmp_path / 'dicom'
    dataset.mkdir()
    (dataset / 'hip.dcm').write_bytes(b'hip')
    image = SimpleNamespace(pixels=np.full((8, 12), 42, dtype=np.uint8),
                            pixel_sha256='hip-pixels', pixel_mm=1.05, pixel_mm_x=.6)
    output = tmp_path / 'hip-review'
    output.mkdir()
    write_blinded_packet([{'pixel_sha256': 'hip-pixels', 'source_paths': ['hip.dcm'],
                           'reference_region': 'hip_left'}], dataset, output, lambda _: image,
                         extra_review_fields={'rotation_state': ('under_rotated', 'optimal', 'over_rotated', 'unassessable')})
    manifest = json.loads((output / 'blind/manifest.json').read_text())
    assert manifest['extra_review_fields']['rotation_state'][0] == 'under_rotated'
    assert 'hip.dcm' not in str(manifest)
    for name, value in (('A', 'under_rotated'), ('B', 'over_rotated')):
        path = output / f'blind/reviewer-{name}.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            row = next(csv.DictReader(stream))
        row.update({'reviewer_id': name, 'reviewed': 'true', 'quality': '1',
                    'hip_position_rotation': '1', 'hip_roi_coverage': '0',
                    'rotation_state': value})
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    report = reconcile(output / 'blind', output / 'blind/reviewer-A.csv',
                       output / 'blind/reviewer-B.csv', output / 'agreement.json')
    assert report['counts']['requires_adjudication'] == 1
    assert report['images'][0]['disagreements'] == ['rotation_state']
