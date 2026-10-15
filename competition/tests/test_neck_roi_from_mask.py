import numpy as np
import pytest

from experiments.neck_roi_from_mask import neck_roi


def test_proposal_covers_measured_bone_and_tissue_in_physical_coordinates():
    mask = np.zeros((200, 200), bool)
    mask[70:110, 20:170] = True
    result = neck_roi(mask, [140, 90], [90, 90], .6, 1.05)
    assert result['status'] == 'proposed'
    physical = np.array(result['polygon']) * [.6, 1.05]
    assert np.ptp(physical[:, 0]) == pytest.approx(15.)
    assert np.ptp(physical[:, 1]) == pytest.approx(result['cross_section_bone_mm'] + 10)
    assert result['clinical_validation'] is False


def test_missing_bone_or_necessary_source_tissue_cannot_become_a_valid_proposal():
    mask = np.zeros((100, 100), bool)
    assert neck_roi(mask, [80, 50], [50, 50], 1., 1.)['status'] == 'unavailable'
    mask[0:20, 20:90] = True
    assert neck_roi(mask, [80, 10], [50, 10], 1., 1.)['status'] == 'unavailable'
    assert neck_roi(mask, [50, 10], [50, 10], 1., 1.)['reason'] == 'neck_axis_degenerate'
