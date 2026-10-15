import numpy as np

from experiments.numbered_spine_roi import propose_spine_rois


def test_spine_density_markers_separate_levels_and_never_invent_missing_levels():
    pixels = np.full((180, 100), 20, np.uint8)
    masks = np.zeros((26, 180, 100), bool)
    for channel, center in [(7, 20), (6, 48), (5, 76), (4, 104), (3, 132), (2, 160)]:
        masks[channel, center - 9:center + 10, 35:66] = True
        pixels[center - 9:center + 10, 35:66] = 180
    result = propose_spine_rois(masks, pixels, .6, 1.05)
    assert len(result['proposals']) == 4
    assert [p['purpose'] for p in result['proposals']] == ['L1', 'L2', 'L3', 'L4']
    assert all(p['mask_fraction_inside'] == 1. for p in result['proposals'])
    for item, upper in zip(result['markers'], (20, 48, 76, 104, 132)):
        assert upper + 9 < item['center'][1] < upper + 19
    masks[5] = False
    missing = propose_spine_rois(masks, pixels, .6, 1.05)
    assert missing['status'] == 'unavailable' and not missing['proposals']
