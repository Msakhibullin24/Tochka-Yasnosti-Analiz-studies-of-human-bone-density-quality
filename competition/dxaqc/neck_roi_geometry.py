"""Physical geometry of an explicit neck ROI against candidate anatomy.

Method recommendations §2.7 require soft tissue on both sides, exclusion of the
large trochanter and a transverse neck ROI. They specify no angular tolerance.
We measure the relationships, retaining uncertainty of unverified bone masks.
"""
import math

import numpy as np

from .anatomical_roi import polygon_mask
from .mask_raster import decode_raster


def _mask(item, shape):
    return decode_raster(item['raster'], shape) if 'raster' in item else polygon_mask(item.get('points', item.get('polygon')), shape)


def _principal_axis(mask, sx, sy):
    y, x = np.where(mask)
    if len(x) < 3:
        return None
    points = np.column_stack((x*sx, y*sy))
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    if values[-1] <= 0 or np.isclose(values[-1], values[0], rtol=1e-6, atol=1e-12):
        return None
    return vectors[:, -1]


def neck_geometry(roi, femur, sx, sy, neck_axis_points=None):
    """Return measurements, never a pass/fail from an invented angle tolerance."""
    roi, femur = np.asarray(roi, bool), np.asarray(femur, bool)
    if roi.ndim != 2 or roi.shape != femur.shape or not roi.any() or not femur.any():
        raise ValueError('Compatible nonempty ROI and femur masks are required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (sx, sy)):
        raise ValueError('Positive finite physical scale is required')
    bone = roi & femur
    roi_axis = _principal_axis(roi, sx, sy)
    # Clipping a bone to the assessed ROI biases its principal axis toward that
    # ROI. It must not substitute for a separately localised anatomical neck axis.
    bone_axis = _principal_axis(bone, sx, sy)
    neck_axis = None
    if neck_axis_points is not None:
        points = np.asarray(neck_axis_points, dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all() or (points < 0).any() or (points[:, 0] >= roi.shape[1]).any() or (points[:, 1] >= roi.shape[0]).any():
            raise ValueError('Neck axis requires two finite in-image points')
        direction = (points[1] - points[0]) * [sx, sy]
        length = np.linalg.norm(direction)
        if length <= 0:
            raise ValueError('Neck axis endpoints must differ')
        neck_axis = direction / length
    result = {'bone_pixels_in_roi': int(bone.sum()), 'roi_pixels': int(roi.sum()),
              'roi_to_neck_axis_angle_deg': None, 'deviation_from_perpendicular_deg': None,
              'soft_tissue_sides': None, 'clinical_verdict': 'undetermined',
              'clinical_validation': False, 'basis': 'candidate_femur_mask_within_explicit_neck_roi',
              'limitations': ['The anatomical neck axis must be localised separately; clipped bone PCA is only a descriptive measurement.',
                             'Pixels outside the candidate femur mask may contain pelvic bone; they are not confirmed soft tissue.',
                             'No angular or minimum soft-tissue-width tolerance is specified in the supplied requirements.']}
    if bone_axis is not None and roi_axis is not None:
        result['bone_within_roi_axis_angle_deg'] = float(np.degrees(np.arccos(np.clip(abs(np.dot(roi_axis, bone_axis)), 0, 1))))
    if neck_axis is not None and roi_axis is not None:
        angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(roi_axis, neck_axis)), 0, 1))))
        result.update(roi_to_neck_axis_angle_deg=angle, deviation_from_perpendicular_deg=90.-angle)
    if roi_axis is None or not bone.any():
        result['status'] = 'partial'
        result['reason'] = 'ROI orientation or bone intersection unavailable'
        return result
    yr, xr = np.where(roi); yb, xb = np.where(bone)
    projected_roi = np.column_stack((xr*sx, yr*sy)) @ roi_axis
    projected_bone = np.column_stack((xb*sx, yb*sy)) @ roi_axis
    low, high = float(projected_bone.min()), float(projected_bone.max())
    # Negating the eigenvector swaps the side names without changing the rule.
    result['soft_tissue_sides'] = {
        'negative_axis': {'outside_femur_candidate_pixels': int((projected_roi < low - 1e-8).sum()),
                          'candidate_extent_mm': low - float(projected_roi.min())},
        'positive_axis': {'outside_femur_candidate_pixels': int((projected_roi > high + 1e-8).sum()),
                          'candidate_extent_mm': float(projected_roi.max()) - high},
        'both_sides_have_candidate_pixels': bool((projected_roi < low - 1e-8).any() and (projected_roi > high + 1e-8).any())}
    result['status'] = 'measured' if neck_axis is not None else 'partial'
    if neck_axis is None:
        result['reason'] = 'Independent anatomical neck axis unavailable'
    return result


def assess_neck_rois(source, regions, img):
    rois = [r for r in source.get('rois', []) if r.get('purpose') == 'femoral_neck']
    if not rois:
        return {'status': 'not_applicable', 'checks': []}
    by_name = {r['name']: r for r in regions}
    checks = []
    for roi in rois:
        check = {'roi_id': roi['id'], 'clinical_verdict': 'undetermined', 'clinical_validation': False}
        if img.pixel_mm_source in ('device_default', '', None) or 'femur' not in by_name:
            checks.append({**check, 'status': 'unavailable', 'reason': 'Reliable scale or candidate femur mask unavailable'})
            continue
        try:
            actual = _mask(roi, img.pixels.shape)
            femur = _mask(by_name['femur'], img.pixels.shape)
            check.update(neck_geometry(actual, femur, img.pixel_mm_x or img.pixel_mm, img.pixel_mm,
                                       by_name.get('femoral_neck_axis', {}).get('points')))
            # A point cannot establish whole-structure exclusion. Only explicitly
            # named masks supply an overlap measurement, with their provenance.
            exclusions = {}
            for name in ('greater_trochanter', 'ischium'):
                if name in by_name:
                    mask = _mask(by_name[name], img.pixels.shape)
                    exclusions[name] = {'candidate_intersection_pixels': int((actual & mask).sum()),
                                        'status': 'measured', 'clinical_verdict': 'undetermined'}
                else:
                    exclusions[name] = {'status': 'unavailable', 'reason': 'Named structure mask unavailable'}
            check['structure_exclusions'] = exclusions
        except (ValueError, TypeError, KeyError) as exc:
            check.update(status='unavailable', reason=str(exc))
        checks.append(check)
    return {'status': 'evaluated', 'checks': checks, 'clinical_validation': False}
