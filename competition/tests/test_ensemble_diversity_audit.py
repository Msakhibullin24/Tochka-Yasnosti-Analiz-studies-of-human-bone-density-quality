import csv
import json

import numpy as np
import pytest

from experiments.ensemble_diversity_audit import audit, summarize


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def cohort(tmp_path):
    labels, rows = [], []
    for i in range(4):
        truth = i % 2
        labels.append({'first_source_path': f'{i}.dcm', 'study_key': f's{i // 2}',
                       'region': 'spine', 'quality_class': truth,
                       'spine_axis': truth, 'spine_artifact': '', 'spine_coverage': 0})
        scores = {'spine_axis': .9 if truth else .1, 'spine_artifact': .1, 'spine_coverage': .1}
        states = {k: {'status': 'pass'} for k in scores}
        rows.append({'index': i, 'repeat': 0, 'fold': i // 2, 'study': f's{i // 2}',
                     'source_path': f'{i}.dcm', 'true_region': 'spine', 'predicted_region': 'spine',
                     'quality_true': truth, 'quality_pred': 0, 'quality_score': .2,
                     'criterion_scores': json.dumps(scores),
                     'criterion_thresholds': json.dumps({k: .5 for k in scores}),
                     'criterion_states': json.dumps(states)})
    reference = tmp_path / 'labels.csv'
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_csv(reference, labels)
    write_csv(a, rows)
    write_csv(b, list(reversed(rows)))
    return reference, {'a': a, 'b': b}, rows


def test_audit_aligns_by_source_and_separates_axis_votes_from_final_rules(cohort):
    labels, experts, _ = cohort
    report = audit(labels, experts)
    axis = report['by_criterion']['spine_axis']
    raw = axis['classifier_threshold_votes']['experts']['a']['metrics_on_evaluated']
    final = axis['final_criterion_decisions']['experts']['a']['metrics_on_evaluated']
    assert raw['tn_fp_fn_tp'] == [2, 0, 0, 2]
    assert final['tn_fp_fn_tp'] == [2, 0, 2, 0]
    assert report['by_criterion']['spine_artifact']['final_criterion_decisions']['reference_n'] == 0
    assert report['weights_fitted'] is False
    assert report['model_affects_decision'] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('field,value,message', [
    ('fold', 5, 'split across folds'),
    ('quality_true', 1, 'truth differs'),
    ('quality_score', 'nan', 'Non-finite'),
    ('predicted_region', 'unknown', 'Unsupported'),
    ('repeat', 1, 'identity'),
    ('index', 9, 'identity'),
    ('quality_pred', 2, 'binary'),
])
def test_invalid_evidence_is_rejected(cohort, field, value, message):
    labels, experts, rows = cohort
    rows[0][field] = value
    write_csv(experts['b'], rows)
    with pytest.raises(ValueError, match=message):
        audit(labels, experts)


def test_aligned_images_with_different_folds_cannot_be_combined(cohort):
    labels, experts, rows = cohort
    for row in rows:
        row['fold'] += 10
    write_csv(experts['b'], rows)
    with pytest.raises(ValueError, match='held-out folds'):
        audit(labels, experts)


@pytest.mark.parametrize('field,payload,message', [
    ('criterion_scores', {'spine_axis': .2}, 'region contract'),
    ('criterion_scores', {'spine_axis': True, 'spine_coverage': .1, 'spine_artifact': .1}, 'Boolean'),
    ('criterion_scores', {'spine_axis': 1.2, 'spine_coverage': .1, 'spine_artifact': .1}, 'outside'),
    ('criterion_states', {k: {'status': 'normal'} for k in ('spine_axis', 'spine_coverage', 'spine_artifact')}, 'state'),
])
def test_malformed_criterion_is_not_interpreted_as_normal(cohort, field, payload, message):
    labels, experts, rows = cohort
    rows[0][field] = json.dumps(payload)
    write_csv(experts['b'], rows)
    with pytest.raises(ValueError, match=message):
        audit(labels, experts)


def test_missing_votes_have_explicit_coverage_and_never_become_negative():
    y = np.array([1, 1, 0, 0])
    report = summarize(y, {'a': np.array([np.nan, 0., 0., 1.]),
                           'b': np.array([1., 1., 0., 0.])})
    assert report['experts']['a']['unavailable_positives'] == 1
    pair = report['pairs']['a__b']
    assert pair['common_n'] == 3
    assert pair['left_fn_detected_by_right'] == 1
    assert pair['left_fp_rejected_by_right'] == 1
    assert pair['both_false_negative'] == 0
    assert pair['error_correlation'] is None
    assert report['all_experts_common']['excluded_unavailable'] == 1


def test_no_common_coverage_remains_explicit_and_json_serializable():
    report = summarize(np.array([1, 0]), {
        'a': np.array([np.nan, np.nan]), 'b': np.array([1., 0.])})
    assert report['experts']['a']['metrics_on_evaluated'] is None
    assert report['all_experts_common']['n'] == 0
    assert report['pairs']['a__b']['error_correlation'] is None
    json.dumps(report, allow_nan=False)


def test_split_region_routing_preserves_missing_criterion_coverage(cohort):
    labels, experts, rows = cohort
    row = rows[1]
    row['predicted_region'] = 'hip_left'
    hip = {'hip_position_rotation': .1, 'hip_roi_coverage': .1}
    row['criterion_scores'] = json.dumps(hip)
    row['criterion_thresholds'] = json.dumps({k: .5 for k in hip})
    row['criterion_states'] = json.dumps({k: {'status': 'pass'} for k in hip})
    for path in experts.values():
        write_csv(path, rows)
    result = audit(labels, experts)['by_criterion']['spine_axis']['final_criterion_decisions']
    assert result['reference_n'] == 4
    assert result['experts']['a']['unavailable_positives'] == 1


def test_duplicate_sources_are_rejected(cohort):
    labels, experts, rows = cohort
    rows[1] = rows[0].copy()
    write_csv(experts['b'], rows)
    with pytest.raises(ValueError, match='exactly once'):
        audit(labels, experts)


def test_fixed_vote_controls_expose_sensitivity_specificity_tradeoff():
    result = summarize(np.array([1, 1, 0, 0]), {
        'a': np.array([1., 0., 1., 0.]),
        'b': np.array([0., 1., 0., 0.]),
        'c': np.array([0., 1., 0., 0.])})
    controls = result['fixed_voting_controls_on_common']
    assert controls['any_positive']['tn_fp_fn_tp'] == [1, 1, 0, 2]
    assert controls['majority']['tn_fp_fn_tp'] == [2, 0, 1, 1]
    assert controls['all_positive']['tn_fp_fn_tp'] == [2, 0, 2, 0]
    assert 'majority' not in summarize(np.array([1]), {
        'a': np.array([1.]), 'b': np.array([0.])})['fixed_voting_controls_on_common']
