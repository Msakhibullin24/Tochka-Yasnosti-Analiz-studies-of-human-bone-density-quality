import numpy as np

from experiments.quality_gated_decoder import gated_decisions


def test_axis_is_preserved_when_quality_gate_is_negative_and_not_invented_by_argmax():
    codes = ['spine_coverage', 'spine_axis', 'spine_artifact']
    scores = np.array([[.1, .1, .1], [.3, .99, .4], [.9, .2, .9]])
    selected, fallback = gated_decisions(scores, [.5, .5, .5], codes,
                                         np.array([6., 2., 2.]), np.array([.1, .9, .1]), .5)
    np.testing.assert_array_equal(selected, [[False, True, False], [False, False, True], [False, False, False]])
    np.testing.assert_array_equal(fallback, [False, True, False])


def test_quality_gate_keeps_multiple_types_and_is_independent_of_test_batch():
    codes = ['hip_position_rotation', 'hip_roi_coverage']
    scores = np.array([[.9, .8], [.3, .6]])
    result, fallback = gated_decisions(scores, [.5, .5], codes, np.full(2, np.nan), np.array([.9, .9]), .5)
    np.testing.assert_array_equal(result, [[True, True], [False, True]])
    for i in range(2):
        one, _ = gated_decisions(scores[i:i+1], [.5, .5], codes, np.array([np.nan]), np.array([.9]), .5)
        np.testing.assert_array_equal(one[0], result[i])


def test_preserved_independent_mode_does_not_suppress_an_existing_type():
    codes = ['hip_position_rotation', 'hip_roi_coverage']
    selected, _ = gated_decisions([[.9, .1]], [.5, .5], codes, np.array([np.nan]),
                                  np.array([.1]), .8, preserve_independent=True)
    np.testing.assert_array_equal(selected, [[True, False]])
