"""Geometric measurements of an automatically proposed neck ROI.

All relationships use predicted masks/points. Non-bone is not certified soft
tissue, a trochanter point is not its whole contour, and unavailable geometry
is not evidence that the source image has an ROI defect. No clinical threshold
is invented for pelvic overlap or non-bone extent.
"""
from __future__ import annotations

import cv2
import numpy as np


def measure_roi(proposal, femur, pelvis, points, sx, sy):
    features = {'anat_roi_proposal_available': float(proposal['status'] == 'proposed')}
    if proposal['status'] != 'proposed':
        return features, {'status': 'unavailable', 'reason': proposal.get('reason'), 'clinical_validation': False}
    femur, pelvis = np.asarray(femur, bool), np.asarray(pelvis, bool)
    polygon = np.asarray(proposal['polygon'], float)
    points = np.asarray(points, float)
    if (femur.ndim != 2 or pelvis.shape != femur.shape or polygon.shape != (4, 2) or points.shape != (4, 2)
            or not np.isfinite(np.r_[polygon.ravel(), points.ravel(), sx, sy]).all() or min(sx, sy) <= 0):
        raise ValueError('Expected calibrated same-frame masks, four landmarks and quadrilateral ROI')
    physical = polygon * [sx, sy]
    axis = (points[2] - points[3]) * [sx, sy]
    length = np.linalg.norm(axis)
    if length < 5:
        return {'anat_roi_proposal_available': 0.}, {'status': 'unavailable', 'reason': 'degenerate_axis', 'clinical_validation': False}
    axis /= length
    normal = np.array([-axis[1], axis[0]])
    origin = points[3] * [sx, sy]
    raster = np.zeros(femur.shape, np.uint8)
    cv2.fillConvexPoly(raster, np.rint(polygon).astype(np.int32), 1)
    inside = raster > 0
    area = int(inside.sum())
    if area < 4:
        return {'anat_roi_proposal_available': 0.}, {'status': 'unavailable', 'reason': 'roi_raster_too_small', 'clinical_validation': False}
    features.update({'anat_roi_area_mm2': area * sx * sy,
        'anat_roi_femur_fraction': float((femur & inside).sum() / area),
        'anat_roi_pelvis_fraction': float((pelvis & inside).sum() / area),
        'anat_roi_nonbone_fraction': float((inside & ~femur & ~pelvis).sum() / area),
        'anat_roi_source_femur_fraction_inside': float((inside & femur).sum() / max(1, femur.sum()))})
    for name, point in [('gt', points[0]), ('head', points[2]), ('axis_origin', points[3])]:
        distance = cv2.pointPolygonTest(physical.astype(np.float32), tuple((point * [sx, sy]).tolist()), True)
        features[f'anat_roi_{name}_inside'] = float(distance >= 0)
        features[f'anat_roi_{name}_signed_margin_mm'] = float(distance)
    relative = (physical - origin) @ np.stack([axis, normal]).T
    extent = relative.max(0) - relative.min(0)
    center = relative.mean(0)
    features.update({'anat_roi_along_axis_mm': float(extent[0]), 'anat_roi_across_axis_mm': float(extent[1]),
                     'anat_roi_center_along_axis_fraction': float(center[0] / length),
                     'anat_roi_center_across_axis_fraction': float(center[1] / length)})
    yy, xx = np.where(inside)
    transverse = (np.column_stack([xx * sx, yy * sy]) - origin) @ normal
    # Fractions on both transverse ends, without calling unlabelled tissue soft tissue.
    for name, selection in [('minus', transverse <= transverse.min() + max(sx, sy)),
                            ('plus', transverse >= transverse.max() - max(sx, sy))]:
        features[f'anat_roi_{name}_edge_bone_fraction'] = float(femur[yy[selection], xx[selection]].mean())
    vectors = np.roll(physical, -1, axis=0) - physical
    norms = np.linalg.norm(vectors, axis=1)
    cosine = np.abs(vectors @ axis) / np.maximum(norms, 1e-10)
    features['anat_roi_perpendicular_error_deg'] = float(np.degrees(np.arcsin(np.clip(cosine.min(), 0, 1))))
    report = {'status': 'evaluated', 'measurement_basis': 'predicted_mask_and_landmark_relationships',
              'features': features, 'clinical_validation': False, 'affects_decision_as_hard_rule': False,
              'whole_greater_trochanter_exclusion_confirmed': False,
              'soft_tissue_identity_confirmed': False}
    return features, report
