import pytest

from experiments.hip_criterion_audit import LANDMARKS, summarize


def test_summary_separates_candidate_coverage_from_criterion_accuracy():
    records = [
        {'reference': True, 'predicted': False, 'features': {'shaft_abs_angle_deg': 8.},
         'landmarks': {name: name == 'greater_trochanter' for name in LANDMARKS}},
        {'reference': True, 'predicted': True, 'features': {'shaft_abs_angle_deg': 6.},
         'landmarks': {name: False for name in LANDMARKS}},
        {'reference': False, 'predicted': True, 'features': {'shaft_abs_angle_deg': 2.},
         'landmarks': {name: False for name in LANDMARKS}},
        {'reference': False, 'predicted': False, 'features': {'shaft_abs_angle_deg': 1.},
         'landmarks': {name: True for name in LANDMARKS}},
    ]
    result = summarize(records)
    assert result['criterion_confusion_tn_fp_fn_tp'] == [1, 1, 1, 1]
    assert result['landmark_candidate_counts']['greater_trochanter'] == {'positive': 1, 'negative': 1}
    assert result['features']['shaft_abs_angle_deg']['raw_auc'] == 1.0
    assert result['features']['shaft_abs_angle_deg']['positive_median'] == 7.0


def test_summary_requires_both_reference_classes():
    with pytest.raises(ValueError, match='both reference classes'):
        summarize([{'reference': True, 'predicted': True, 'features': {}, 'landmarks': {}}])
