from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from experiments.spine_body_axis_candidates import body_axes


def masks(centers=(70, 76, 82, 88)):
    result = np.zeros((26, 150, 220), bool)
    for channel, x, y in zip((6, 5, 4, 3), centers, (20, 50, 80, 110)):
        result[channel, y-9:y+10, x-14:x+15] = True
    return result


def test_physical_body_axis_uses_both_pixel_scales_and_reflects_direction():
    measured = body_axes(masks(), .6, 1.05)
    expected = np.degrees(np.arctan(6*.6/(30*1.05)))
    assert measured['status'] == 'evaluated'
    for angle in measured['angles_degrees'].values(): assert angle == pytest.approx(expected)
    reflected = body_axes(masks()[..., ::-1], .6, 1.05)
    for method, angle in reflected['angles_degrees'].items():
        assert angle == pytest.approx(-measured['angles_degrees'][method])


def test_curvature_is_not_substituted_for_global_inclination():
    measured = body_axes(masks((60, 80, 80, 60)), .6, 1.05)
    assert max(abs(v) for v in measured['pair_angles_degrees']) > 5
    assert all(abs(v) < 1e-9 for v in measured['angles_degrees'].values())


def test_missing_body_is_unavailable_not_zero_degree_pass():
    source = masks(); source[3] = False
    measured = body_axes(source, .6, 1.05)
    assert measured['status'] == 'unavailable'
    assert 'angles_degrees' not in measured
