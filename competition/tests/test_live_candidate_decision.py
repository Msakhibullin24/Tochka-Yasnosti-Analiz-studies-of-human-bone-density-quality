from experiments.live_candidate_decision import gated_result, ensemble_result


def spec():
    return {'criterion_decoder': {'thresholds': {'spine_axis': .5, 'spine_artifact': .5, 'spine_coverage': .5}},
            'quality_cut': .6, 'preserve_independent': False}


def test_quality_gate_suppresses_independent_types_but_never_suppresses_measured_axis():
    scores = {'spine_axis': .1, 'spine_artifact': .9, 'spine_coverage': .2}
    normal = gated_result('spine', scores, {'spine_abs_angle_deg': 3.}, .3, spec())
    assert normal['quality'] == 0 and not normal['violations']
    physical = gated_result('spine', scores, {'spine_abs_angle_deg': 6.}, .3, spec())
    assert physical['violations'] == ['spine_axis']
    conditional = gated_result('spine', {'spine_axis': .8, 'spine_artifact': .2, 'spine_coverage': .1},
                               {'spine_abs_angle_deg': 3.}, .8, spec())
    assert conditional['quality'] == 1 and conditional['violations'] == ['spine_artifact']
    assert 'conditional_type_prediction_requires_review' in conditional['review_reasons']
    assert conditional['criterion_states']['spine_axis']['status'] == 'pass'


def test_binary_fold_majority_is_not_lost_when_members_disagree_on_type():
    decisions = [{'quality': 1, 'violations': ['hip_position_rotation']},
                 {'quality': 1, 'violations': ['hip_position_rotation']},
                 {'quality': 1, 'violations': ['hip_roi_coverage']},
                 {'quality': 0, 'violations': []}, {'quality': 0, 'violations': []}]
    raw = [{'hip_position_rotation': .4, 'hip_roi_coverage': .3}] * 5
    result, votes = ensemble_result('hip', decisions, raw, .4, {})
    assert max(votes.values()) < .5
    assert result['quality'] == 1 and result['violations'] == ['hip_position_rotation']
    assert result['violation_type_status'] == 'identified'
    assert 'ensemble_type_disagreement_requires_review' in result['review_reasons']
