import numpy as np
import pytest

from experiments.automatic_anatomy_geometry import th12_visibility, hip_roi_relationships, iliac_crests


def spine(visible_rows):
    masks = np.zeros((26, 280, 100), bool)
    for channel, top in ((6, 60), (5, 110), (4, 160), (3, 210)):
        masks[channel, top:top+40, 30:70] = True
    masks[7, :visible_rows, 30:70] = True
    return masks


def hip():
    masks = np.zeros((23, 150, 120), bool)
    masks[6, 10:50, 20:41] = True
    masks[6, 40:140, 30:51] = True
    masks[6, 25:36, 40:100] = True
    points = np.array([[25., 12.], [48., 65.], [90., 30.], [60., 30.]])
    return masks, points


def roi(left=45, right=75, top=15, bottom=45):
    return {'status': 'proposed', 'polygon': [[left, top], [right, top], [right, bottom], [left, bottom]]}


@pytest.mark.parametrize('rows, expected', [(40, 'estimated_pass'), (20, 'estimated_pass'), (19, 'estimated_fail')])
def test_half_visible_body_boundary_is_inclusive_and_uses_physical_neighbor_extent(rows, expected):
    result = th12_visibility(spine(rows), .6, 1.05)
    assert result['outcome'] == expected
    assert result['visible_fraction_estimate'] == pytest.approx(rows/40)
    assert not result['anatomical_body_identity_verified']
    assert not result['range_is_statistical_confidence_interval']


def test_missing_th12_or_reference_does_not_create_a_measured_fraction():
    missing = th12_visibility(spine(0), .6, 1.05)
    assert missing['outcome'] == 'not_detected'
    assert missing['visible_fraction_estimate'] is None
    masks = spine(20)
    masks[[6, 5, 4]] = False
    assert th12_visibility(masks, .6, 1.05)['status'] == 'unavailable'


def test_gt_point_outside_roi_does_not_prove_the_compartment_is_excluded():
    masks, points = hip()
    result = hip_roi_relationships(masks, points, roi(left=38), 1., 1.)
    gt = result['greater_trochanter_compartment']
    assert points[0, 0] < 38 and points[0, 1] < 15
    assert gt['roi_overlap_mm2'] > 0
    assert gt['estimated_compartment_excluded'] is False
    assert not gt['boundary_is_expert_gt_contour']


def test_nonbone_background_alone_cannot_be_reported_as_soft_tissue():
    masks, points = hip()
    result = hip_roi_relationships(masks, points, roi(), .6, 1.05)
    assert not result['soft_tissue_sides']['predicted_soft_tissue_at_both_ends']
    masks[11, 15:25, 45:76] = True
    masks[19, 36:46, 45:76] = True
    result = hip_roi_relationships(masks, points, roi(), .6, 1.05)
    assert result['soft_tissue_sides']['predicted_soft_tissue_at_both_ends']
    assert result['pelvic_overlap']['automatic_failure_threshold'] is None


def test_hip_relationships_are_invariant_under_horizontal_reflection():
    masks, points = hip()
    original = hip_roi_relationships(masks, points, roi(), .6, 1.05)
    points[:, 0] = masks.shape[2]-1-points[:, 0]
    reflected = roi()
    reflected['polygon'] = [[masks.shape[2]-1-x, y] for x, y in reflected['polygon']]
    other = hip_roi_relationships(masks[..., ::-1], points, reflected, .6, 1.05)
    assert {k: v for k, v in original['greater_trochanter_compartment'].items() if k != 'polygons'} == {
        k: v for k, v in other['greater_trochanter_compartment'].items() if k != 'polygons'}
    # The oriented cross-axis normal reverses under reflection; the unordered
    # pair of tissue ends and the both-sides conclusion must remain identical.
    assert original['soft_tissue_sides']['ends'] == other['soft_tissue_sides']['ends'][::-1]
    assert original['soft_tissue_sides']['predicted_soft_tissue_at_both_ends'] == other['soft_tissue_sides']['predicted_soft_tissue_at_both_ends']


def test_empty_iliac_masks_or_border_only_structure_are_explicit():
    masks = np.zeros((23, 100, 100), bool)
    masks[8, 0:20, 0:20] = True
    a, b = iliac_crests(masks)
    assert a['outcome'] == 'boundary_ambiguous'
    assert b['outcome'] == 'not_detected' and not b['points']


def test_unknown_geometry_is_not_replaced_by_invented_roi_or_landmarks():
    masks, points = hip()
    missing = hip_roi_relationships(masks, None, {'status': 'unavailable', 'reason': 'unlocalized'}, .6, 1.05)
    assert missing['status'] == 'unavailable' and missing['reason'] == 'unlocalized'
    with pytest.raises(ValueError, match='pixel spacing'):
        th12_visibility(spine(20), 0., 1.)
