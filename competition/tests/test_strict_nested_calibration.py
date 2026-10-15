import numpy as np
import pytest

from experiments.strict_nested_calibration import grouped_threshold, threshold, train_criterion
from experiments.threshold_stability import fit_predict, probability


def test_outer_label_perturbation_cannot_change_models_cuts_or_predictions():
    rng = np.random.default_rng(12)
    geometry, embedding = rng.normal(size=(48, 3)), rng.normal(size=(48, 5))
    studies = np.repeat([f'study-{i:02}' for i in range(24)], 2)
    y = (geometry[:, 0] + rng.normal(size=48) > 0).astype(float)
    train, test = np.arange(36), np.arange(36, 48)
    first = train_criterion(geometry, embedding, y, studies, train, test,
                            kind='geometry_lr', bags=5)
    altered = y.copy()
    altered[test] = np.nan
    second = train_criterion(geometry, embedding, altered, studies, train, test,
                             kind='geometry_lr', bags=5)
    for key in ('raw', 'platt', 'raw_threshold', 'platt_threshold'):
        np.testing.assert_array_equal(first[key], second[key])
    singleton = train_criterion(geometry, embedding, y, studies, train, test[:1],
                                kind='geometry_lr', bags=5)
    for key in ('raw', 'platt'):
        np.testing.assert_allclose(first[key][:1], singleton[key], rtol=0, atol=1e-12)
    assert first['audit'] == second['audit']
    held = set(studies[test])
    assert not held & set(first['audit']['training_studies'])
    for split in first['audit']['inner_splits']:
        assert not set(split['training_studies']) & set(split['validation_studies'])
        assert not held & (set(split['training_studies']) | set(split['validation_studies']))


@pytest.mark.parametrize('constant', [0, 1])
def test_single_class_training_preserves_the_observed_class(constant):
    x = np.arange(32).reshape(16, 2)
    studies = np.array([f's{i}' for i in range(16)])
    y = np.full(16, float(constant))
    y[-4:] = np.nan
    result = train_criterion(x, x, y, studies, np.arange(12), np.arange(12, 16),
                             kind='geometry_lr', bags=0)
    np.testing.assert_array_equal(result['raw'], np.full(4, constant))
    np.testing.assert_array_equal(result['platt'], np.full(4, constant))
    tr, te, head = fit_predict('lr', x, y, np.arange(12), np.arange(12, 16), 17)
    np.testing.assert_array_equal(tr, np.full(12, constant))
    np.testing.assert_array_equal(te, np.full(4, constant))
    np.testing.assert_array_equal(probability(head, x[-4:]), np.full(4, constant))


def test_threshold_includes_all_negative_and_is_deterministic():
    assert threshold(np.zeros(3, int), np.array([.1, .5, 1.])) > 1
    y, scores, groups = np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), np.array(['a', 'a', 'b', 'b'])
    assert .2 < threshold(y, scores) <= .8
    assert grouped_threshold(y, scores, groups, 10, 17) == grouped_threshold(y, scores, groups, 10, 17)
    with pytest.raises(ValueError, match='nonfinite'):
        threshold(y, np.array([.1, .2, np.nan, .9]))


def test_repeated_study_cannot_cross_outer_split():
    x = np.zeros((8, 2))
    with pytest.raises(ValueError, match='Outer study leakage'):
        train_criterion(x, x, np.arange(8) % 2, np.repeat(['a', 'b'], 4),
                        np.arange(5), np.arange(5, 8))


def test_research_default_matches_release_and_alternative_stays_separate():
    from itertools import product
    from dxaqc.decision import decide as release
    from experiments.research_decision import decide as research
    for group, angle, score, criterion, cut in product(
            ('spine', 'hip'), (None, float('nan'), 0., 5., 5.01, -7.),
            (.1, .9), (.1, .9), (.1, .5, 1.)):
        criteria = {'spine_axis': criterion} if group == 'spine' else {'hip_position_rotation': criterion}
        features = {'spine_abs_angle_deg': angle, 'sat_frac': .02}
        assert release(group, score, criteria, features, cut, {}) == research(group, score, criteria, features, cut, {})
    args = ('spine', .1, {'spine_axis': .1}, {'spine_abs_angle_deg': 7.}, 1., {})
    assert release(*args)['quality'] == 1
    assert research(*args, axis_rule='model')['quality'] == 0
