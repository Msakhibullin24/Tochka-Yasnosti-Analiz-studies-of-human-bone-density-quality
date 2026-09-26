"""Paired transforms; targets describe only the controlled defect, never unknown QC."""
from __future__ import annotations

import cv2
import numpy as np


def _inputs(pixels, masks):
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
        raise ValueError('Expected uint8 grayscale image')
    if any(np.asarray(mask).shape != pixels.shape for mask in masks.values()):
        raise ValueError('Mask/image dimensions differ')
    return pixels


def rotate_pair(pixels, masks, degrees, spacing):
    pixels = _inputs(pixels, masks)
    sx, sy = spacing
    if not all(np.isfinite(v) for v in (degrees, sx, sy)) or min(sx, sy) <= 0:
        raise ValueError('Invalid physical transformation')
    h, w = pixels.shape
    centre = np.array([(w-1)/2, (h-1)/2])
    scale = np.diag([sx, sy])
    rotation = cv2.getRotationMatrix2D((0, 0), float(degrees), 1.)[:, :2]
    linear = np.linalg.solve(scale, rotation @ scale)
    affine = np.column_stack([linear, centre-linear @ centre])
    altered = cv2.warpAffine(pixels, affine, (w, h), flags=cv2.INTER_LINEAR)
    transformed = {name: cv2.warpAffine(np.asarray(mask, np.uint8), affine, (w, h),
                                       flags=cv2.INTER_NEAREST).astype(bool) for name, mask in masks.items()}
    return altered, transformed, {'affine': affine.tolist(), 'relative_degrees': float(degrees),
                                  'target_origin': 'synthetic_geometry', 'qc_targets': {}}


def crop_pair(pixels, masks, box):
    pixels = _inputs(pixels, masks)
    x0, y0, x1, y1 = box
    h, w = pixels.shape
    if not all(isinstance(v, int) for v in box) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Invalid crop box')
    altered = {name: np.asarray(mask, bool)[y0:y1, x0:x1].copy() for name, mask in masks.items()}
    visibility = {name: float(mask.sum()/max(1, np.asarray(masks[name], bool).sum()))
                  for name, mask in altered.items()}
    return pixels[y0:y1, x0:x1].copy(), altered, {
        'crop_xyxy': list(box), 'retained_mask_fraction': visibility,
        'target_origin': 'synthetic_geometry', 'qc_targets': {}}


def rotate_expand(pixels, degrees, spacing):
    pixels = _inputs(pixels, {})
    sx, sy = spacing
    if not all(np.isfinite(v) for v in (degrees, sx, sy)) or min(sx, sy) <= 0:
        raise ValueError('Invalid physical rotation')
    h, w = pixels.shape
    scale = np.diag([sx, sy])
    rotation = cv2.getRotationMatrix2D((0, 0), float(degrees), 1.)[:, :2]
    linear = np.linalg.solve(scale, rotation @ scale)
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]) @ linear.T
    origin = np.floor(corners.min(axis=0))
    size = np.ceil(corners.max(axis=0)-origin).astype(int)+1
    affine = np.column_stack([linear, -origin])
    if int(size.prod()) > 36_000_000:
        raise ValueError('Expanded rotation exceeds image limit')
    return cv2.warpAffine(pixels, affine, tuple(size), flags=cv2.INTER_LINEAR), affine


def artifact_pair(pixels, masks, box):
    pixels = _inputs(pixels, masks)
    _, _, metadata = crop_pair(pixels, masks, box)
    x0, y0, x1, y1 = box
    result = pixels.copy()
    result[y0:y1, x0:x1] = 255
    artifact = np.zeros(pixels.shape, bool); artifact[y0:y1, x0:x1] = True
    return result, {**{k: np.asarray(v, bool).copy() for k, v in masks.items()},
                    'injected_artifact': artifact}, {
        **metadata, 'target_origin': 'synthetic_injected_object',
        'qc_targets': {'spine_artifact': 1}, 'implant_ground_truth': False}


def transform_polygon(points, affine):
    points, affine = np.asarray(points, dtype=float), np.asarray(affine, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or affine.shape != (2, 3):
        raise ValueError('Invalid polygon or affine')
    if not np.isfinite(points).all() or not np.isfinite(affine).all():
        raise ValueError('Nonfinite polygon transform')
    return (np.c_[points, np.ones(len(points))] @ affine.T).tolist()
