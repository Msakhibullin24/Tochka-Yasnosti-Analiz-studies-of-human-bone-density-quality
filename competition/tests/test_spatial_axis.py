import numpy as np
import pytest

from evaluate_spatial_axis import axis_angle


def test_axis_uses_anisotropic_physical_spacing_and_preserves_sign():
    y = np.array([10., 30., 50., 70.])
    x = np.tan(np.radians(10))*y*1.05/.6
    points = np.column_stack([x, y])
    assert axis_angle(points, .6, 1.05) == pytest.approx(10)
    assert axis_angle(points + [100, 200], .6, 1.05) == pytest.approx(10)
    points[:, 0] *= -1
    assert axis_angle(points, .6, 1.05) == pytest.approx(-10)


@pytest.mark.parametrize('points, sx, sy', [
    (np.zeros((4, 2)), .6, 1.05),
    (np.array([[0, 1], [0, 3], [0, 2], [0, 4]]), .6, 1.05),
    (np.full((4, 2), np.nan), .6, 1.05),
    (np.array([[0, 1], [0, 2], [0, 3], [0, 4]]), 0, 1.05),
    (np.zeros((3, 2)), .6, 1.05),
])
def test_invalid_or_unordered_centers_never_become_a_normal_axis(points, sx, sy):
    with pytest.raises(ValueError):
        axis_angle(points, sx, sy)
