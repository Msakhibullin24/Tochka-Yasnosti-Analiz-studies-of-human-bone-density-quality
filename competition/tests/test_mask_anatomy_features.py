import numpy as np
import pytest

from experiments.mask_anatomy_features import mask_statistics, spine_features


def test_mask_measurements_use_both_pixel_axes_and_exclude_islands():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 10:30] = 1
    mask[0, 90] = 1
    values, center = mask_statistics(mask, mask * 200, .6, 1.05)
    assert values['width_mm'] == pytest.approx(12)
    assert values['height_mm'] == pytest.approx(21)
    assert values['top_mm'] == pytest.approx(21)
    assert values['touch_top'] == 0
    assert center == pytest.approx((19.5, 29.5))


def test_numbered_axis_uses_ordered_lumbar_centers_and_physical_coordinates():
    pixels = np.zeros((200, 100), np.uint8)
    masks = np.zeros((26, 200, 100), bool)
    for channel, y, x in [(6, 30, 30), (5, 60, 35), (4, 90, 40), (3, 120, 45)]:
        masks[channel, y:y+12, x:x+12] = True
    values, angle = spine_features(masks, pixels, .6, 1.05)
    expected = np.degrees(np.arctan((5 * .6) / (30 * 1.05)))
    assert angle == pytest.approx(expected)
    assert values['anat_lumbar_ordered'] == 1
    assert values['anat_Th12_present'] == 0
    masks[5] = 0
    values, angle = spine_features(masks, pixels, .6, 1.05)
    assert angle is None
    assert values['anat_lumbar_ordered'] == 0
    assert 'anat_lumbar_angle_abs_deg' not in values


def test_empty_and_tiny_anatomy_cannot_supply_an_axis():
    masks = np.zeros((26, 100, 100), bool)
    masks[:, 10, 10] = True
    values, angle = spine_features(masks, np.zeros((100, 100), np.uint8), 1., 1.)
    assert angle is None and values['anat_L1_present'] == 0
    with pytest.raises(ValueError):
        spine_features(masks[:10], np.zeros((100, 100), np.uint8), 1., 1.)


def test_degenerate_hip_mask_abstains_from_landmark_geometry():
    from experiments.hip_mask_features import hip_measurements
    masks = np.zeros((23, 100, 100), bool)
    masks[6, 10:90, 20:22] = True
    features, points = hip_measurements(masks, np.zeros((100, 100), np.uint8), .6, 1.05, None)
    assert points is None
    assert features['anat_landmark_input_valid'] == 0
    assert 'anat_troch_top_margin_mm' not in features
