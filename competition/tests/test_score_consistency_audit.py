import csv

import numpy as np
import pytest

from experiments.score_consistency_audit import audit, reliability_bins


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_reliability_bins_include_exact_one_and_leave_empty_bins_explicit():
    bins = reliability_bins(np.array([0, 1]), np.array([0.0, 1.0]))
    assert sum(item['n'] for item in bins) == 2
    assert bins[0]['observed_rate'] == 0
    assert bins[-1]['observed_rate'] == 1
    assert bins[1]['mean_score'] is None


def test_score_audit_checks_study_folds_and_keeps_decisions_separate(tmp_path):
    labels = tmp_path / 'labels.csv'
    oof = tmp_path / 'oof.csv'
    _csv(labels, [
        {'first_source_path': 'a.dcm', 'study_key': 'a', 'region': 'spine', 'quality_class': '0'},
        {'first_source_path': 'b.dcm', 'study_key': 'b', 'region': 'hip_left', 'quality_class': '1'},
    ])
    rows = [
        {'source_path': 'a.dcm', 'study': 'a', 'true_region': 'spine', 'quality_true': '0',
         'quality_pred': '1', 'quality_score': '0.2', 'repeat': '0', 'fold': '0'},
        {'source_path': 'b.dcm', 'study': 'b', 'true_region': 'hip_left', 'quality_true': '1',
         'quality_pred': '0', 'quality_score': '0.8', 'repeat': '0', 'fold': '1'},
    ]
    _csv(oof, rows)
    result = audit(oof, labels)
    assert result['roc_auc'] == 1
    assert result['decision_score_crossings_at_0_5']['positive_decision_below_0_5'] == 1
    assert result['decision_score_crossings_at_0_5']['normal_decision_at_least_0_5'] == 1
    _csv(labels, [
        {'first_source_path': 'a.dcm', 'study_key': 'a', 'region': 'spine', 'quality_class': '0'},
        {'first_source_path': 'b.dcm', 'study_key': 'a', 'region': 'hip_left', 'quality_class': '1'},
    ])
    rows[1]['study'] = 'a'
    _csv(oof, rows)
    with pytest.raises(ValueError, match='multiple held-out folds'):
        audit(oof, labels)
