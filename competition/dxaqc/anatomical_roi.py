"""Geometry against explicit named reference/candidate masks; purpose is required."""
from __future__ import annotations

import cv2
import numpy as np

from .mask_raster import decode_raster


def polygon_mask(points, shape):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Invalid ROI polygon')
    h, w = shape
    if min(shape) < 1 or (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError('ROI polygon outside raster')
    if not cv2.isContourConvex(points.astype(np.float32)):
        raise ValueError('ROI must be a convex nondegenerate polygon')
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if mask.sum() < 4:
        raise ValueError('Degenerate ROI')
    return mask.astype(bool)


def compare_roi(points, reference, *, purpose, reference_origin):
    """No diagnostic label from IoU: measure inclusion/exclusion relationships."""
    reference = np.asarray(reference, dtype=bool)
    if reference.ndim != 2 or not reference.any() or not purpose or not reference_origin:
        raise ValueError('Named nonempty reference and provenance required')
    roi = polygon_mask(points, reference.shape)
    return compare_masks(roi, reference, purpose=purpose, reference_origin=reference_origin)


def compare_masks(roi, reference, *, purpose, reference_origin):
    roi, reference = np.asarray(roi, bool), np.asarray(reference, bool)
    if roi.shape != reference.shape or roi.ndim != 2 or not roi.any() or not reference.any() or not purpose or not reference_origin:
        raise ValueError('Compatible nonempty named masks required')
    intersection = int((roi & reference).sum())
    union = int((roi | reference).sum())
    return {'purpose': purpose, 'reference_origin': reference_origin,
            'reference_coverage': intersection/int(reference.sum()),
            'outside_reference_fraction': int((roi & ~reference).sum())/int(roi.sum()),
            'iou': intersection/union, 'clinical_validation': False,
            'status': 'measured', 'clinical_verdict': 'undetermined'}


def compare_source_rois(source, regions, shape):
    if source['status'] == 'absent':
        return {'status': 'not_applicable', 'checks': []}
    references = {r['name']: r for r in regions}
    checks = []
    for roi in source['rois']:
        name = 'femoral_neck_roi' if roi['purpose'] == 'femoral_neck' else roi['purpose']
        if name not in references:
            checks.append({'roi_id': roi['id'], 'status': 'unavailable', 'reason': 'Named predicted reference unavailable'})
            continue
        try:
            candidate = references[name]
            reference = (decode_raster(candidate['raster'], shape) if 'raster' in candidate else
                         polygon_mask(cv2.convexHull(np.asarray(candidate['polygon'], np.float32)).reshape(-1, 2), shape))
            actual = decode_raster(roi['raster'], shape) if 'raster' in roi else polygon_mask(roi['points'], shape)
            check = compare_masks(actual, reference, purpose=roi['purpose'], reference_origin='learned_mask_candidate' if 'raster' in candidate else 'learned_polygon_candidate_convex_hull')
            checks.append({'roi_id': roi['id'], **check})
        except ValueError as exc:
            checks.append({'roi_id': roi['id'], 'status': 'unavailable', 'reason': str(exc)})
    return {'status': 'evaluated', 'checks': checks, 'clinical_validation': False}


def propose_roi(reference, *, purpose, reference_origin):
    reference = np.asarray(reference, dtype=bool)
    if reference.ndim != 2 or not reference.any() or not purpose or not reference_origin:
        raise ValueError('Named nonempty reference and provenance required')
    contours, _ = cv2.findContours(reference.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2)
    if len(hull) < 3:
        raise ValueError('Reference cannot form a polygon')
    return {'purpose': purpose, 'polygon': hull.astype(float).tolist(),
            'reference_origin': reference_origin, 'status': 'proposed',
            'requires_confirmation': True, 'clinical_validation': False}
