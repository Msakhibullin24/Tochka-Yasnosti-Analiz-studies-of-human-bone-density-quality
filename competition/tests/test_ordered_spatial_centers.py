import numpy as np
import pytest

from ordered_spatial_centers import decode_ordered


def test_joint_order_uses_each_levels_scores_without_relabeling_sorted_points():
    values = np.full((1, 4, 10, 10), -20.)
    for level in range(4):
        values[0, level, level+1, level+1] = 10
    values[0, 0, 8, 9] = 11  # independent L1 argmax would be below L4
    decoded = decode_ordered(values)[0]
    np.testing.assert_allclose(decoded, [[.15,.15],[.25,.25],[.35,.35],[.45,.45]])
    assert np.all(np.diff(decoded[:, 1]) > 0)


def test_order_prior_alone_does_not_make_uniform_heatmaps_verified_landmarks():
    result = decode_ordered(np.zeros((2, 4, 8, 8)))
    assert result.shape == (2, 4, 2)
    assert (np.diff(result[:, :, 1], axis=1) > 0).all()
    # This is coordinate decoding only; consumers must keep candidate status.


@pytest.mark.parametrize('values', [np.zeros((1, 4, 3, 8)), np.full((1, 4, 8, 8), np.nan)])
def test_impossible_or_invalid_order_fails(values):
    with pytest.raises(ValueError):
        decode_ordered(values)
