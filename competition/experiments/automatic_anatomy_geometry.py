"""Model-derived anatomical geometry; execution is distinct from clinical proof.

The vertebral body is estimated by removing thin processes from named masks.
The GT compartment is a landmark-conditioned partition of the predicted femur,
not an expert contour. Soft-tissue evidence comes from named muscle masks;
background or the complement of bone alone cannot establish soft tissue.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from experiments.mask_anatomy_features import largest_component

MUSCLE_CHANNELS = tuple(range(11, 21))  # TSXR 23-channel map: glutei, autochthon, iliopsoas.


def calibration(sx, sy):
    if not all(np.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError('Finite positive physical pixel spacing is required')
    return np.array([sx, sy], float)


def body_component(mask, sx, sy, radius):
    """A morphology-derived body candidate, without asserting anatomical truth."""
    cleaned = largest_component(mask)
    if radius > 0:
        rx, ry = max(1, int(np.ceil(radius / sx))), max(1, int(np.ceil(radius / sy)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rx+1, 2*ry+1))
        cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_OPEN, kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return largest_component(cleaned)


def th12_visibility(masks, sx, sy):
    spacing = calibration(sx, sy)
    masks = np.asarray(masks, bool)
    if masks.ndim != 3 or masks.shape[0] != 26:
        raise ValueError('Expected 26-channel original-coordinate vertebral masks')
    required = .5
    base = {'method': 'named_mask_central_body_morphology_and_adjacent_height_prior',
            'required_visible_fraction': required, 'clinical_validation': False,
            'anatomical_body_identity_verified': False, 'estimate_not_direct_body_segmentation': True}
    reference = [largest_component(masks[c]) for c in (6, 5, 4, 3)]
    present = [m for m in reference if m.sum() >= max(4, .001*m.size)
               and not m[0].any() and not m[-1].any()]
    if len(present) < 2:
        return {**base, 'status': 'unavailable', 'reason': 'insufficient_named_lumbar_reference'}
    radii = [float(distance_transform_edt(m, sampling=(sy, sx)).max()) for m in present]
    # Fixed technical shape filter; this is not a clinical tolerance.
    radius = .35 * float(np.median(radii))
    centers = np.array([[np.where(m)[1].mean()*sx, np.where(m)[0].mean()*sy] for m in present])
    if np.ptp(centers[:, 1]) <= sy or np.any(np.diff(centers[:, 1]) <= 0):
        return {**base, 'status': 'unavailable', 'reason': 'degenerate_reference_axis'}
    slope = float(np.polyfit(centers[:, 1], centers[:, 0], 1)[0])
    longitudinal = np.array([slope, 1.]) / np.sqrt(1+slope*slope)

    def extent(mask):
        y, x = np.where(mask)
        if not len(x): return None
        u = np.column_stack([x*sx, y*sy]) @ longitudinal
        return float(np.ptp(u) + abs(longitudinal[0])*sx + abs(longitudinal[1])*sy)

    heights = [extent(body_component(m, sx, sy, radius)) for m in present]
    heights = [v for v in heights if v is not None]
    if len(heights) < 2:
        return {**base, 'status': 'unavailable', 'reason': 'body_candidates_not_resolved'}
    named = largest_component(masks[7])
    details = {'reference_body_extents_mm': heights, 'opening_radius_mm': radius,
               'reference_height_mm': float(np.median(heights))}
    if named.sum() < max(4, .001*named.size):
        return {**base, **details, 'status': 'evaluated', 'outcome': 'not_detected',
                'visible_fraction_estimate': None, 'requires_review': True}
    observed = extent(body_component(named, sx, sy, radius))
    clipped = bool(named[0].any())
    if observed is None:
        return {**base, **details, 'status': 'evaluated', 'outcome': 'body_not_resolved',
                'visible_fraction_estimate': None, 'touches_superior_frame': clipped, 'requires_review': True}
    fraction = float(min(1., observed / np.median(heights))) if clipped else 1.
    interval = [float(min(1., observed/max(heights))), float(min(1., observed/min(heights)))] if clipped else [1., 1.]
    tolerance = 32*np.finfo(float).eps  # floating-point boundary, not a clinical tolerance
    outcome = 'estimated_pass' if interval[0] >= required-tolerance else 'estimated_fail' if interval[1] < required-tolerance else 'uncertain'
    return {**base, **details, 'status': 'evaluated', 'outcome': outcome,
            'visible_fraction_estimate': fraction, 'adjacent_height_sensitivity_range': interval,
            'range_is_statistical_confidence_interval': False, 'observed_body_extent_mm': observed,
            'touches_superior_frame': clipped, 'requires_review': outcome != 'estimated_pass'}


def iliac_crests(masks):
    masks = np.asarray(masks, bool)
    if masks.ndim != 3 or masks.shape[0] != 23:
        raise ValueError('Expected 23-channel musculoskeletal masks')
    result = []
    h, w = masks.shape[1:]
    for name, channel in (('iliac_crest_left', 8), ('iliac_crest_right', 9)):
        mask = largest_component(masks[channel]); y, x = np.where(mask)
        points = []
        outcome = 'not_detected'
        if len(x) >= max(4, .001*h*w):
            top = int(y.min()); xx = x[y == top]
            points = [[float(np.median(xx)), float(top)]]
            outcome = 'boundary_ambiguous' if top in (0, h-1) or np.all((xx == 0) | (xx == w-1)) else 'localized'
        result.append({'id': name, 'status': 'evaluated', 'outcome': outcome, 'points': points,
                       'method': 'cranial_extent_of_predicted_named_hemipelvis', 'clinical_validation': False,
                       'requires_review': outcome != 'localized'})
    return result


def hip_roi_relationships(masks, points, proposal, sx, sy):
    spacing = calibration(sx, sy)
    masks = np.asarray(masks, bool)
    if masks.ndim != 3 or masks.shape[0] != 23:
        raise ValueError('Expected 23-channel musculoskeletal masks')
    base = {'clinical_validation': False, 'affects_qc_classifier': False,
            'target': 'automatically_proposed_neck_roi', 'whole_anatomical_gt_verified': False,
            'muscle_channels': list(MUSCLE_CHANNELS)}
    if points is None or proposal.get('status') != 'proposed':
        return {**base, 'status': 'unavailable', 'reason': proposal.get('reason', 'landmarks_unavailable')}
    points, polygon = np.asarray(points, float), np.asarray(proposal['polygon'], float)
    h, w = masks.shape[1:]
    if (points.shape != (4, 2) or polygon.shape != (4, 2) or not np.isfinite(points).all()
            or not np.isfinite(polygon).all() or np.any(polygon < 0) or np.any(polygon >= [w, h])):
        raise ValueError('Finite same-frame landmarks and in-frame quadrilateral ROI required')
    axis = (points[2] - points[3]) * spacing
    length = float(np.linalg.norm(axis))
    if length < 5:
        return {**base, 'status': 'unavailable', 'reason': 'neck_axis_degenerate'}
    axis /= length
    if any(np.linalg.norm((points[i]-points[j])*spacing) < min(sx, sy)/2
           for i in range(4) for j in range(i)):
        return {**base, 'status': 'unavailable', 'reason': 'unresolved_prominence_partition'}
    normal = np.array([-axis[1], axis[0]])
    femur, pelvis = largest_component(masks[6] | masks[7]), largest_component(masks[8] | masks[9])
    if not femur.any():
        return {**base, 'status': 'unavailable', 'reason': 'femur_not_detected'}
    yy, xx = np.mgrid[:h, :w]
    world = np.stack([xx*sx, yy*sy], -1)
    distances = ((world[..., None, :] - points[None, None] * spacing)**2).sum(-1)
    # A landmark-conditioned compartment, not an anatomical GT ground truth.
    gt = femur & (distances[..., 0] <= distances[..., 1:].min(-1))
    raster = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(raster, np.rint(polygon).astype(np.int32), 1)
    inside = raster.astype(bool)
    area = int(inside.sum())
    if area < 4:
        return {**base, 'status': 'unavailable', 'reason': 'degenerate_roi_raster'}
    muscle = masks[list(MUSCLE_CHANNELS)].any(0)
    soft = muscle & ~femur & ~pelvis
    transverse = (world - points[3]*spacing) @ normal
    values = transverse[inside]
    edge_width = max(sx, sy)
    counts = []
    for subset in (inside & (transverse <= values.min()+edge_width),
                   inside & (transverse >= values.max()-edge_width)):
        counts.append({'roi_pixels': int(subset.sum()), 'predicted_muscle_nonbone_pixels': int((subset & soft).sum()),
                       'bone_pixels': int((subset & (femur | pelvis)).sum())})
    gt_count = int(gt.sum()); overlap = int((gt & inside).sum())
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.reshape(-1, 2).astype(float).tolist() for contour in contours if len(contour) >= 3]
    soft_both = all(item['predicted_muscle_nonbone_pixels'] > 0 for item in counts)
    return {**base, 'status': 'evaluated',
            'greater_trochanter_compartment': {'method': 'predicted_femur_partition_by_GT_LT_head_neck_landmarks',
                'area_mm2': gt_count*sx*sy, 'roi_overlap_mm2': overlap*sx*sy,
                'polygons': polygons, 'coordinate_units': 'original_image_pixels',
                'roi_overlap_fraction_of_compartment': overlap/max(1, gt_count),
                'estimated_compartment_excluded': overlap == 0 if gt_count else None,
                'boundary_is_expert_gt_contour': False, 'landmark_point_alone_not_used_for_exclusion': True},
            'soft_tissue_sides': {'method': 'predicted_named_muscles_outside_predicted_femur_and_pelvis',
                'ends': counts, 'predicted_soft_tissue_at_both_ends': soft_both,
                'outcome': 'model_supported' if soft_both else 'not_detected_at_both_ends',
                'missing_muscle_prediction_does_not_prove_missing_soft_tissue': True},
            'pelvic_overlap': {'area_mm2': int((inside & pelvis).sum())*sx*sy,
                'fraction_of_roi': float((inside & pelvis).sum()/area), 'automatic_failure_threshold': None,
                'reason': 'A small ischial overlap is permitted; no numeric tolerance is specified.'}}
