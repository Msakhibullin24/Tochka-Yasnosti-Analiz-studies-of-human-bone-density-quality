import numpy as np
import pytest

from structured_typifier import StructuredTypifier, target
from evaluate_structured_typifier import apply_candidate, paired_f1_interval
from predict_structured_typifier import describe


def reference(quality='1', axis='1'):
    return {'region': 'spine', 'quality_class': quality, 'spine_axis': axis,
            'spine_coverage': '0', 'spine_artifact': '0'}


def test_reference_filter_excludes_contradictions_and_incomplete_criteria():
    assert target(reference()) == 'spine_axis'
    assert target(reference('0', '0')) == 'normal'
    assert target(reference('1', '0')) is None
    assert target(reference('0', '1')) is None
    assert target(reference(axis='')) is None
    assert target({**reference(), 'region': 'unknown'}) is None


def test_learned_normal_is_separate_evidence_and_never_mutates_baseline():
    row = {'quality_pred': '1', 'violation_type': '', 'predicted_region': 'spine',
           'decision_version': '5'}
    result = apply_candidate(row, {'codes': [], 'basis': 'learned_reference_combination'}, 'untyped_only')
    assert result['quality_pred'] == '0' and result['violation_type'] == ''
    assert result['decision_reason'] == 'experimental_structured_normal'
    assert result['decision_version'] == 'experimental-structured-1'
    assert row['quality_pred'] == '1'
    typed = {**row, 'violation_type': 'Присутствуют посторонние предметы'}
    assert apply_candidate(typed, {'codes': []}, 'untyped_only') == typed
    with pytest.raises(ValueError, match='type'):
        apply_candidate(row, {'codes': ['hip_roi_coverage']}, 'untyped_only')


def test_fitted_classifier_outputs_only_observed_combinations_and_normal():
    model = StructuredTypifier('spine')
    features = {key: 0. for key in model.feature_names}
    model.fit([features]*8,
        np.array([[0., i/100] for i in range(4)]+[[1., i/100] for i in range(4)]),
        ['normal']*4+['spine_axis']*4)
    predictions = model.predict([features, features], [[0., 0.], [1., 0.]])
    assert predictions[0]['codes'] == []
    assert predictions[1]['codes'] == ['spine_axis']
    assert predictions[1]['clinical_validation'] is False
    assert np.isclose(sum(predictions[1]['combination_scores'].values()), 1)
    with pytest.raises(ValueError, match='official'):
        StructuredTypifier('spine').fit([{}]*2, [[0], [1]], ['normal', 'hip_roi_coverage'])


def test_learned_axis_does_not_override_physical_requirement_or_baseline():
    result = {'region': 'spine', 'quality': 1, 'violations': [],
              'features': {'spine_abs_angle_deg': 3.}}
    proposal = describe(result, {'codes': ['spine_axis']})
    assert proposal['candidate_violation_type'] == 'Не выравнена ось позвоночника'
    assert proposal['conflicts'] == ['learned_axis_type_with_measured_angle_at_most_5deg']
    assert proposal['original_verdict_changed'] is False
    assert proposal['original_violation_type'] == ''
    assert describe(result, {'codes': []})['conflicts'] == ['secondary_normal_vs_original_binary_alarm']
    assert result['violations'] == []


def test_paired_comparison_rejects_changed_identity():
    rows = [{'source_path': 'a', 'study': 'study-a', 'quality_true': '1', 'quality_pred': '1'},
            {'source_path': 'b', 'study': 'study-b', 'quality_true': '0', 'quality_pred': '0'}]
    assert paired_f1_interval(rows, rows, repeats=20) == [0., 0.]
    with pytest.raises(ValueError, match='identities'):
        paired_f1_interval(rows, rows[::-1])
