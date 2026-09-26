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


def test_region_routing_error_is_reviewed_separately():
    row = {'predicted_region': 'hip_right', 'quality_pred': '0', 'violation_type': ''}
    label = {'region': 'spine', 'spine_axis': '0'}
    assert review_reasons(row, label) == ['region_mismatch']


def test_matching_region_does_not_add_review_reason():
    row = {'predicted_region': 'hip_left', 'quality_pred': '0', 'violation_type': ''}
    label = {'region': 'hip_left', 'hip_position_rotation': '0'}
    assert review_reasons(row, label) == []


def test_hip_roi_disagreements_require_explicit_reference():
    roi_type = VIOLATION_LABEL['hip_roi_coverage']
    missed = {'quality_pred': '0', 'violation_type': ''}
    false_alarm = {'quality_pred': '1', 'violation_type': roi_type}
    assert review_reasons(missed, {'region': 'hip_left', 'hip_roi_coverage': '1'}) == [
        'missed_hip_roi_coverage']
    assert review_reasons(false_alarm, {'region': 'hip_right', 'hip_roi_coverage': '0'}) == [
        'false_hip_roi_coverage']
    assert review_reasons(false_alarm, {'region': 'hip_right', 'hip_roi_coverage': ''}) == []
