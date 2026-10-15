"""Construct an inspectable neck ROI from a named axis and femur mask.

Defaults are explicit engineering proposal parameters, not organizer-defined
clinical tolerances. The polygon includes measured bone cross-section and
soft-tissue extensions; missing evidence produces an unavailable proposal.
"""
from __future__ import annotations

import numpy as np


def neck_roi(mask, head, neck, sx, sy, *, along_axis_mm=15., soft_margin_mm=5.):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        return {'status': 'unavailable', 'reason': 'femur_mask_missing'}
    values = np.r_[head, neck, sx, sy, along_axis_mm, soft_margin_mm]
    if not np.isfinite(values).all() or min(sx, sy, along_axis_mm, soft_margin_mm) <= 0:
        raise ValueError('Finite points, positive scale and proposal dimensions required')
    center = np.asarray(neck, float) * [sx, sy]
    axis = (np.asarray(head, float) - neck) * [sx, sy]
    length = np.linalg.norm(axis)
    if length < 5:
        return {'status': 'unavailable', 'reason': 'neck_axis_degenerate'}
    axis = axis / length
    normal = np.array([-axis[1], axis[0]])
    positions = np.arange(-60., 60.001, .25)
    samples = np.rint((center + positions[:, None] * normal) / [sx, sy]).astype(int)
    valid = ((samples >= 0) & (samples < mask.shape[::-1])).all(axis=1)
    inside = np.zeros(len(samples), bool)
    inside[valid] = mask[samples[valid, 1], samples[valid, 0]]
    middle = int(np.argmin(abs(positions)))
    if not inside[middle]:
        return {'status': 'unavailable', 'reason': 'neck_center_outside_candidate_femur'}
    first, last = middle, middle
    while first > 0 and inside[first - 1]:
        first -= 1
    while last + 1 < len(inside) and inside[last + 1]:
        last += 1
    if first == 0 or last == len(inside) - 1 or positions[last] - positions[first] < 3:
        return {'status': 'unavailable', 'reason': 'neck_cross_section_unresolved'}
    lower, upper = positions[first] - soft_margin_mm, positions[last] + soft_margin_mm
    polygon_mm = np.stack([center + s * along_axis_mm / 2 * axis + t * normal
                           for s, t in ((-1, lower), (1, lower), (1, upper), (-1, upper))])
    polygon = polygon_mm / [sx, sy]
    if np.any(polygon < 0) or np.any(polygon >= mask.shape[::-1]):
        return {'status': 'unavailable', 'reason': 'proposal_requires_tissue_outside_source_frame'}
    return {'status': 'proposed', 'purpose': 'femoral_neck', 'polygon': polygon.tolist(),
            'axis_unit_physical': axis.tolist(), 'cross_section_bone_mm': float(positions[last] - positions[first]),
            'soft_tissue_extension_mm_each_side': float(soft_margin_mm), 'along_axis_mm': float(along_axis_mm),
            'parameter_basis': 'fixed_engineering_proposal_not_organizer_tolerance',
            'requires_confirmation': True, 'clinical_validation': False}
