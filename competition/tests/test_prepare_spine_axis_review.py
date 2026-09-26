import pytest

from prepare_spine_axis_review import select


def test_axis_review_selects_all_positives_and_near_boundary_controls():
    labels = [
        {'first_source_path': 'positive.dcm', 'study_key': 's1', 'region': 'spine',
         'quality_class': '1', 'spine_coverage': '0', 'spine_axis': '1', 'spine_artifact': '0'},
        {'first_source_path': 'boundary.dcm', 'study_key': 's2', 'region': 'spine',
         'quality_class': '0', 'spine_coverage': '0', 'spine_axis': '0', 'spine_artifact': '0'},
        {'first_source_path': 'far.dcm', 'study_key': 's3', 'region': 'spine',
         'quality_class': '0', 'spine_coverage': '0', 'spine_axis': '0', 'spine_artifact': '0'},
        {'first_source_path': 'conflicting.dcm', 'study_key': 's4', 'region': 'spine',
         'quality_class': '1', 'spine_coverage': '0', 'spine_axis': '0', 'spine_artifact': '0'},
    ]
    stress = [{'index': str(index), 'study': label['study_key'], 'region': 'spine',
               'axis_label': label['spine_axis'] + '.0', 'global_abs_angle_deg': str(angle)}
              for index, (label, angle) in enumerate(zip(labels, (2.5, 4.9, 0.5, 5.0)))]
    selected = select(labels, stress, controls=1)
    assert {item['source_path'] for item in selected} == {'positive.dcm', 'boundary.dcm'}
    assert [item['axis_label'] for item in selected].count('1') == 1
    assert select(labels, stress, controls=1) == selected
    stress[1]['study'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        select(labels, stress, controls=1)


def test_axis_review_rejects_duplicate_stress_indexes():
    labels = [{'first_source_path': path, 'study_key': path, 'region': 'spine',
               'quality_class': quality, 'spine_coverage': '0', 'spine_axis': axis,
               'spine_artifact': '0'}
              for path, quality, axis in [('a', '1', '1'), ('b', '0', '0')]]
    stress = [{'index': '0', 'study': 'a', 'region': 'spine', 'axis_label': '1.0',
               'global_abs_angle_deg': '3.0'}] * 2
    with pytest.raises(ValueError, match='indexes'):
        select(labels, stress, controls=1)
