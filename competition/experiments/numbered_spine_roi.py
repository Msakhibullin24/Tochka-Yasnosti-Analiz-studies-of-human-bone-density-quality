"""Numbered spine ROI proposals split at projection-density minima.

Named segmentation supplies the level identity. Density minima supply candidate
intervertebral markers, as described in methodology fig.7; they are not direct
segmentation of intervertebral discs. No pathological vertebrae are excluded.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from experiments.mask_anatomy_features import largest_component


def propose_spine_rois(masks, pixels, sx, sy):
    pixels = np.asarray(pixels)
    if masks.shape != (26, *pixels.shape) or not all(np.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError('Expected calibrated 26-channel original-coordinate masks')
    cleaned, centers = {}, {}
    for name, channel in [('Th12', 7), ('L1', 6), ('L2', 5), ('L3', 4), ('L4', 3), ('L5', 2)]:
        mask = largest_component(masks[channel])
        ys, xs = np.where(mask)
        if len(xs) < max(4, .001 * pixels.size):
            continue
        cleaned[name] = mask
        centers[name] = np.array([xs.mean() * sx, ys.mean() * sy])
    required = [f'L{i}' for i in range(1, 5)]
    if not all(name in centers for name in required):
        return {'status': 'unavailable', 'reason': 'missing_numbered_lumbar_masks', 'proposals': [], 'markers': []}
    points = np.array([centers[n] for n in required])
    if np.any(np.diff(points[:, 1]) <= 0):
        return {'status': 'unavailable', 'reason': 'numbered_order_conflict', 'proposals': [], 'markers': []}
    slope, _ = np.polyfit(points[:, 1], points[:, 0], 1)
    longitudinal = np.array([slope, 1.]) / np.sqrt(1 + slope**2)
    transverse = np.array([longitudinal[1], -longitudinal[0]])
    origin = points.mean(0)
    h, w = pixels.shape
    yy, xx = np.mgrid[:h, :w]
    world = np.stack([xx * sx, yy * sy], -1)
    u, v = (world - origin) @ longitudinal, (world - origin) @ transverse
    center_u = {name: float((point - origin) @ longitudinal) for name, point in centers.items()}
    marker_by_pair, markers = {}, []
    levels = ['Th12', 'L1', 'L2', 'L3', 'L4', 'L5']
    for upper, lower in zip(levels[:-1], levels[1:]):
        if upper not in centers or lower not in centers:
            continue
        gap = center_u[lower] - center_u[upper]
        if gap < 5:
            continue
        # Search the middle 40% between centers; avoid central body-density valleys.
        samples = np.arange(center_u[upper] + .3 * gap, center_u[upper] + .7 * gap, min(sx, sy) / 2)
        widths = []
        for name in (upper, lower):
            widths.append(float(np.quantile(np.abs(v[cleaned[name]]), .75)))
        band = max(3., min(widths) * .6)
        offsets = np.linspace(-band, band, 31)
        coords = origin + samples[:, None, None] * longitudinal + offsets[None, :, None] * transverse
        coordinates = [coords[..., 1] / sy, coords[..., 0] / sx]
        valid = ((coords[..., 0] >= 0) & (coords[..., 0] <= (w - 1) * sx) &
                 (coords[..., 1] >= 0) & (coords[..., 1] <= (h - 1) * sy)).all(1)
        if valid.sum() < 3:
            continue
        density = gaussian_filter1d(map_coordinates(pixels.astype(float), coordinates, order=1, mode='nearest').mean(1), 1.)
        density[~valid] = np.inf
        selection = int(np.argmin(density))
        position = float(samples[selection])
        point = origin + position * longitudinal
        marker_by_pair[upper, lower] = position
        markers.append({'between': [upper, lower], 'center': (point / [sx, sy]).tolist(),
                        'method': 'central_projection_density_minimum',
                        'at_search_boundary': selection in (0, len(samples) - 1),
                        'disc_segmentation_confirmed': False})
    proposals = []
    for i, name in enumerate(required, 1):
        upper, lower = ('Th12' if i == 1 else f'L{i - 1}'), f'L{i + 1}'
        if (upper, name) not in marker_by_pair or (name, lower) not in marker_by_pair:
            continue
        start, end = marker_by_pair[upper, name], marker_by_pair[name, lower]
        transverse_values = v[cleaned[name]]
        left, right = float(transverse_values.min()), float(transverse_values.max())
        polygon = np.array([origin + a * longitudinal + b * transverse for a, b in
                            ((start, left), (start, right), (end, right), (end, left))]) / [sx, sy]
        if np.any(polygon < 0) or np.any(polygon > [w - 1, h - 1]):
            continue
        raster = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(raster, np.rint(polygon).astype(np.int32), 1)
        proposals.append({'purpose': name, 'polygon': polygon.tolist(), 'status': 'proposed',
                          'method': 'named_mask_and_projection_density_minima',
                          'mask_fraction_inside': float((raster.astype(bool) & cleaned[name]).sum() / cleaned[name].sum()),
                          'requires_confirmation': True, 'clinical_validation': False})
    return {'status': 'evaluated', 'proposals': proposals, 'markers': markers,
            'numbered_levels': required, 'axis_degrees': float(np.degrees(np.arctan(slope))),
            'disc_segmentation_confirmed': False, 'clinical_validation': False}
