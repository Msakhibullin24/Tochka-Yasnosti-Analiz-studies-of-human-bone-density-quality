import numpy as np

from experiments.roi_candidate_features import measure_roi


def test_roi_relationships_use_physical_coordinates_and_preserve_unavailability():
    femur = np.zeros((100, 100), bool)
    femur[40:61, 20:81] = True
    pelvis = np.zeros_like(femur)
    points = np.array([[15, 45], [20, 60], [70, 50], [50, 50]], float)
    polygon = [[40, 30], [60, 30], [60, 70], [40, 70]]
    features, report = measure_roi({'status': 'proposed', 'polygon': polygon}, femur, pelvis, points, .6, 1.05)
    assert features['anat_roi_gt_inside'] == 0
    assert features['anat_roi_axis_origin_inside'] == 1
    assert features['anat_roi_gt_signed_margin_mm'] == -15
    assert features['anat_roi_perpendicular_error_deg'] == 0
    assert features['anat_roi_minus_edge_bone_fraction'] == 0
    assert features['anat_roi_plus_edge_bone_fraction'] == 0
    assert report['soft_tissue_identity_confirmed'] is False
    absent, report = measure_roi({'status': 'unavailable', 'reason': 'mask_missing'}, femur, pelvis, None, .6, 1.05)
    assert absent == {'anat_roi_proposal_available': 0.}
    assert report['reason'] == 'mask_missing'
