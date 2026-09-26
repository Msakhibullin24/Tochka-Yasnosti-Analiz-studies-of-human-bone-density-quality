import json

import pytest

from conditional_typifier import apply_conditional_type
from evaluate_conditional_typifier import paired_candidates


def row():
    return {'source_path': 'image', 'study': 'study', 'fold': '1', 'quality_true': '1',
            'predicted_region': 'spine', 'quality_pred': '1', 'quality_score': '.8',
            'violation_type': '', 'criterion_states': json.dumps({'spine_axis': {
                'basis': 'measured_axis_angle', 'status': 'pass', 'angle_deg': 3.}})}


def distribution():
    return {'combination_scores': {'normal': .6, 'spine_axis': .3, 'spine_artifact': .1}}


def test_alarm_and_geometry_are_preserved_despite_secondary_normal():
    original = row()
    result = apply_conditional_type(original, distribution())
    assert result['quality_pred'] == '1' and result['quality_score'] == '.8'
    assert result['criterion_states'] == original['criterion_states']
    assert result['violation_type'] == 'Присутствуют посторонние предметы'
    evidence = json.loads(result['conditional_type_evidence'])
    assert evidence['conditional_score'] == 1.
    assert evidence['normal_score'] == .6 and evidence['secondary_normal_disagreement']
    assert evidence['clinical_validation'] is False and evidence['requires_review']
    assert original['violation_type'] == ''


def test_existing_types_and_negative_answers_are_unchanged():
    for original in [{**row(), 'quality_pred': '0'}, {**row(), 'violation_type': 'known'}]:
        assert apply_conditional_type(original, distribution()) == original


def test_axis_requires_physical_measurement_above_limit():
    original = row()
    original['criterion_states'] = json.dumps({'spine_axis': {
        'basis': 'measured_axis_angle', 'status': 'fail', 'angle_deg': 7.}})
    assert apply_conditional_type(original, distribution())['violation_type'] == 'Не выравнена ось позвоночника'
    original['criterion_states'] = json.dumps({'spine_axis': {'status': 'fail'}})
    result = apply_conditional_type(original, {'combination_scores': {'normal': .2, 'spine_axis': .8}})
    assert result['quality_pred'] == '1' and result['violation_type'] == ''
    assert json.loads(result['conditional_type_evidence'])['status'] == 'undetermined'


@pytest.mark.parametrize('scores', [
    {'normal': .5, 'hip_roi_coverage': .5}, {'normal': .5, 'spine_axis': float('nan')},
    {'normal': .5, 'spine_axis': -.5}, {'normal': .5, 'spine_axis': .2},
    {'normal': .5, 'spine_axis;spine_axis': .5},
])
def test_invalid_model_distribution_is_rejected(scores):
    with pytest.raises(ValueError):
        apply_conditional_type(row(), {'combination_scores': scores})


def test_pairing_rejects_wrong_folds_and_duplicates():
    original = row()
    secondary = {**original, 'typifier_evidence': json.dumps(distribution())}
    assert paired_candidates([original], [secondary])[0]['quality_pred'] == '1'
    for invalid in [[{**secondary, 'fold': '2'}], [secondary, secondary], [{**secondary, 'source_path': 'other'}]]:
        with pytest.raises(ValueError):
            paired_candidates([original], invalid)


def test_axis_at_limit_and_unknown_anatomy_cannot_be_promoted():
    original = row()
    original['criterion_states'] = json.dumps({'spine_axis': {
        'basis': 'measured_axis_angle', 'status': 'fail', 'angle_deg': 5.}})
    result = apply_conditional_type(original, {'combination_scores': {'normal': .2, 'spine_axis': .8}})
    assert result['violation_type'] == '' and result['quality_pred'] == '1'
    with pytest.raises(ValueError, match='anatomy'):
        apply_conditional_type({**row(), 'predicted_region': 'unknown'}, distribution())
