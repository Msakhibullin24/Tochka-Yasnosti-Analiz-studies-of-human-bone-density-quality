import numpy as np

from experiments.three_iteration_upgrade import (decisions, decode_scores, fit_decoder,
                                                fit_family, predict_family, joint_thresholds, utility)


def sample_state():
    rng = np.random.default_rng(21)
    scores = rng.uniform(.05, .95, (40, 3))
    truth = (scores > [.6, .8, .7]).astype(float)
    angles = np.where(truth[:, 1], 6., 2.)
    target = truth.any(axis=1).astype(int)
    return {'codes': ['spine_coverage', 'spine_axis', 'spine_artifact'],
            'inner': dict(zip(['spine_coverage', 'spine_axis', 'spine_artifact'], scores.T)) |
                     {'quality': .1 + .8 * target},
            'criterion_truth': truth, 'quality_truth': target, 'angles': angles}


def test_training_does_not_read_outer_targets_and_predicts_singletons_identically():
    rng = np.random.default_rng(7)
    geometry = rng.normal(size=(24, 3))
    embeddings = [rng.normal(size=(24, 4))]
    target = (geometry[:, 0] > 0).astype(float)
    train, test = np.arange(16), np.arange(16, 24)
    models = fit_family(geometry, embeddings, target, train)
    altered = target.copy()
    altered[test] = np.nan
    other = fit_family(geometry, embeddings, altered, train)
    first = predict_family(models, geometry, embeddings, test)
    np.testing.assert_array_equal(first, predict_family(other, geometry, embeddings, test))
    for i, row in enumerate(test):
        np.testing.assert_allclose(first[i:i+1], predict_family(models, geometry, embeddings, [row]), atol=1e-12)


def test_all_iterations_have_batch_independent_decoders_and_retain_physical_axis():
    state = sample_state()
    raw = np.column_stack([state['inner'][c] for c in state['codes']])
    direct = state['inner']['quality']
    for iteration in (1, 2, 3):
        spec = fit_decoder(state, iteration)
        scores, quality = decode_scores(spec, state['codes'], raw, direct)
        for i in range(len(raw)):
            s, q = decode_scores(spec, state['codes'], raw[i:i+1], direct[i:i+1])
            np.testing.assert_allclose(s[0], scores[i], atol=1e-12)
            np.testing.assert_allclose(q[0], quality[i], atol=1e-12)
        cuts = [spec['thresholds'][c] for c in state['codes']]
        pred = decisions(scores, cuts, state['codes'], state['angles'])
        np.testing.assert_array_equal(pred[:, 1], state['angles'] > 5)


def test_joint_search_cannot_lower_its_training_objective_or_override_axis():
    state = sample_state()
    scores = np.column_stack([state['inner'][c] for c in state['codes']])
    initial = np.array([.5, .5, .5])
    before = decisions(scores, initial, state['codes'], state['angles'])
    cuts = joint_thresholds(state['quality_truth'], state['criterion_truth'], scores,
                            state['codes'], state['angles'], initial)
    after = decisions(scores, cuts, state['codes'], state['angles'])
    assert utility(state['quality_truth'], state['criterion_truth'], after) >= utility(
        state['quality_truth'], state['criterion_truth'], before)
    np.testing.assert_array_equal(before[:, 1], after[:, 1])


def test_constant_training_class_is_preserved():
    x = np.arange(24).reshape(12, 2)
    for label in (0., 1.):
        target = np.r_[np.full(8, label), np.full(4, np.nan)]
        models = fit_family(x, [x], target, np.arange(8))
        np.testing.assert_array_equal(predict_family(models, x, [x], np.arange(8, 12)), np.full(4, label))
