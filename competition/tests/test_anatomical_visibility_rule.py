from experiments.anatomical_visibility_rule import missing_structures, apply_visibility


def test_unavailable_detections_cannot_become_an_absence_violation():
    assert missing_structures({}, 'spine') is None
    assert missing_structures({'anat_lumbar_ordered': 1}, 'spine') is None
    fields = {'anat_lumbar_ordered': 1, 'anat_Th12_present': 1,
              'anat_iliac_left_present': 1, 'anat_iliac_right_present': 0}
    assert missing_structures(fields, 'spine') == ['iliac_right']
    assert missing_structures(fields, 'hip') is None


def test_visibility_cue_adds_a_reviewable_type_without_changing_other_physical_findings():
    fields = {'anat_lumbar_ordered': 1, 'anat_Th12_present': 0,
              'anat_iliac_left_present': 1, 'anat_iliac_right_present': 1}
    original = {'quality': 1, 'violations': ['spine_axis'], 'criterion_states': {
        'spine_axis': {'status': 'fail', 'angle_deg': 6.}, 'spine_coverage': {'status': 'pass'}},
        'review_reasons': [], 'decision_reason': 'criterion_failure'}
    result = apply_visibility(original, fields, 'spine')
    assert result['violations'] == ['spine_coverage', 'spine_axis']
    assert result['criterion_states']['spine_axis'] == original['criterion_states']['spine_axis']
    assert result['criterion_states']['spine_coverage']['clinical_validation'] is False
    assert result['review_reasons'] == ['anatomical_visibility_candidate']
    assert original['violations'] == ['spine_axis']
