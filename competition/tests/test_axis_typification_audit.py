import json

import pytest

from audit_axis_typification import counterfactual


def row(score=.7, threshold=.6, measured='pass'):
    return {'decision_version': '5', 'predicted_region': 'spine',
            'quality_score': '.8', 'quality_threshold': '.5', 'quality_pred': '1',
            'violation_type': '', 'decision_reason': 'binary_only_review',
            'criterion_scores': json.dumps({'spine_axis': score}),
            'criterion_thresholds': json.dumps({'spine_axis': threshold}),
            'criterion_states': json.dumps({'spine_axis': {'status': measured},
                                           'spine_artifact': {'status': 'pass'}})}


def test_candidate_uses_frozen_threshold_without_mutating_release():
    source = row()
    result = counterfactual(source, 'learned_axis')
    assert result['violation_type'] == 'Не выравнена ось позвоночника'
    assert source['violation_type'] == ''
    assert result['criterion_thresholds'] == source['criterion_thresholds']
    assert json.loads(result['criterion_states'])['spine_axis']['clinical_validation'] is False
    assert counterfactual(row(score=.59), 'learned_axis')['violation_type'] == ''


def test_candidate_keeps_hip_unchanged_and_distinguishes_axis_policies():
    source = {**row(), 'predicted_region': 'hip_left'}
    assert counterfactual(source, 'learned_axis') == source
    source = row(score=.1, measured='fail')
    assert counterfactual(source, 'learned_axis')['violation_type'] == ''
    assert counterfactual(source, 'measured_or_learned_axis')['violation_type']


def test_candidate_rejects_wrong_version_and_nonfinite_evidence():
    with pytest.raises(ValueError, match='v5'):
        counterfactual({**row(), 'decision_version': '4'}, 'learned_axis')
    with pytest.raises(ValueError, match='evidence'):
        counterfactual(row(score=float('nan')), 'learned_axis')
