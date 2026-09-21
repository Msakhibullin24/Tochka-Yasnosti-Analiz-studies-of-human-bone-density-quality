from audit_model_errors import review_reasons
from dxaqc.model import VIOLATION_LABEL


def test_hip_rotation_disagreement_is_separate_from_binary_quality():
    label = {'region': 'hip_right', 'hip_position_rotation': '1'}
    row = {'quality_pred': '1', 'violation_type': VIOLATION_LABEL['hip_roi_coverage']}
    assert review_reasons(row, label) == ['missed_hip_position_rotation']


def test_false_hip_rotation_requires_explicit_negative_reference():
    row = {'quality_pred': '1', 'violation_type': VIOLATION_LABEL['hip_position_rotation']}
    assert review_reasons(row, {'region': 'hip_left', 'hip_position_rotation': '0'}) == [
        'false_hip_position_rotation']
    assert review_reasons(row, {'region': 'hip_left', 'hip_position_rotation': ''}) == []


def test_existing_axis_and_untyped_cases_are_retained():
    row = {'quality_pred': '1', 'violation_type': ''}
    label = {'region': 'spine', 'spine_axis': '1'}
    assert review_reasons(row, label) == ['untyped_positive', 'missed_axis']
