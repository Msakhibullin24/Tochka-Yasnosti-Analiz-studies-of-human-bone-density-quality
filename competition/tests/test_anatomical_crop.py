import numpy as np
from experiments.anatomical_crop import hip_crop_bounds


def test_crop_preserves_source_fov_and_uses_physical_units_without_inventing_padding():
    mask = np.zeros((400, 200), bool)
    mask[50:350, 70:130] = True
    assert hip_crop_bounds(mask, .6, 1.05) == [36, 30, 164, 184]
    mask = np.zeros((300, 200), bool)
    mask[:200, :100] = True
    x0, y0, x1, y1 = hip_crop_bounds(mask, .6, 1.05)
    assert (x0, y0) == (0, 0)
    assert x1 <= mask.shape[1] and y1 <= mask.shape[0]
    mask[:] = False
    assert hip_crop_bounds(mask, .6, 1.05) is None
