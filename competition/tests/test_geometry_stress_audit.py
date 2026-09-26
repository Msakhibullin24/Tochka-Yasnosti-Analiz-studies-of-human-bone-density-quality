import numpy as np

from conftest import synthetic_spine
from dxaqc.geometry import measure_image
from experiments.geometry_stress_audit import rotated_physical


def test_physical_rotation_recovers_known_angle_on_anisotropic_grid():
    image = synthetic_spine(0)
    pixel_mm_y, pixel_mm_x = 1.05, 0.6
    base = measure_image(image, 'spine', pixel_mm_y, pixel_mm_x).features['spine_angle_deg']
    changed = rotated_physical(image, 7, pixel_mm_y, pixel_mm_x)
    measured = measure_image(changed, 'spine', pixel_mm_y, pixel_mm_x).features['spine_angle_deg']
    assert np.isfinite(measured)
    assert abs((measured - base) - 7) < 1
