import csv

import pytest

from dxaqc.model import VIOLATION_LABEL
from experiments.label_conflict_sensitivity import audit


def _write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_exclusion_is_diagnostic_and_keeps_reference_files(tmp_path):
    labels_path = tmp_path / 'labels.csv'
    oof_path = tmp_path / 'oof.csv'
    labels = [
        {'first_source_path': 'a', 'study_key': 'a', 'region': 'spine', 'quality_class': '1',
         'spine_coverage': '0', 'spine_axis': '0', 'spine_artifact': '0',
         'hip_position_rotation': '', 'hip_roi_coverage': ''},
        {'first_source_path': 'b', 'study_key': 'b', 'region': 'spine', 'quality_class': '1',
         'spine_coverage': '0', 'spine_axis': '1', 'spine_artifact': '0',
         'hip_position_rotation': '', 'hip_roi_coverage': ''},
        {'first_source_path': 'c', 'study_key': 'c', 'region': 'hip_left', 'quality_class': '0',
         'spine_coverage': '', 'spine_axis': '', 'spine_artifact': '',
         'hip_position_rotation': '0', 'hip_roi_coverage': '0'},
        {'first_source_path': 'd', 'study_key': 'd', 'region': 'hip_right', 'quality_class': '1',
         'spine_coverage': '', 'spine_axis': '', 'spine_artifact': '',
         'hip_position_rotation': '1', 'hip_roi_coverage': '0'},
    ]
    oof = [{'source_path': row['first_source_path'], 'study': row['study_key'],
            'true_region': row['region'], 'quality_true': row['quality_class'],
            'quality_pred': prediction, 'quality_score': score,
            'violation_type': violation, 'repeat': '0', 'fold': fold}
           for row, prediction, score, violation, fold in zip(
               labels, ('0', '1', '0', '1'), ('0.2', '0.8', '0.1', '0.7'),
               ('', VIOLATION_LABEL['spine_axis'], '', VIOLATION_LABEL['hip_position_rotation']),
               ('0', '1', '0', '1'))]
    _write(labels_path, labels)
    _write(oof_path, oof)
    original_labels = labels_path.read_bytes()
    result = audit(oof_path, labels_path)
    assert result['excluded_images'] == 1
    assert result['excluded_reasons'] == {'positive_class_without_failed_criterion': 1}
    assert result['all_labels']['images'] == 4
    assert result['excluding_conflicts']['images'] == 3
    assert result['original_labels_edited'] is False
    assert labels_path.read_bytes() == original_labels
    oof[1]['study'] = 'wrong'
    _write(oof_path, oof)
    with pytest.raises(ValueError, match='identity'):
        audit(oof_path, labels_path)
