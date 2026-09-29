"""Independent review geometry; anatomical contours may be concave.

This does not relax the runtime's convex measurement-ROI contract.
"""
import math

import cv2
import numpy as np


def axis_points(points, shape):
    if (not isinstance(points, list) or len(points) != 2
            or any(not isinstance(p, list) or len(p) != 2 for p in points)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for p in points for v in p)):
        raise ValueError('Axis requires two finite points')
    values = np.asarray(points, float)
    h, w = shape
    if (values < 0).any() or (values[:, 0] >= w).any() or (values[:, 1] >= h).any() or np.array_equal(values[0], values[1]):
        raise ValueError('Axis endpoints must differ and lie inside the source image')
    return values


def contour_mask(points, shape):
    if (not isinstance(points, list) or not 3 <= len(points) <= 128
            or any(not isinstance(p, list) or len(p) != 2 for p in points)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for p in points for v in p)):
        raise ValueError('Contour requires 3..128 finite vertices')
    p = np.asarray(points, float)
    h, w = shape
    if (p < 0).any() or (p[:, 0] >= w).any() or (p[:, 1] >= h).any():
        raise ValueError('Contour lies outside source image')
    if len(set(map(tuple, p))) != len(p):
        raise ValueError('Contour contains repeated vertices')
    def cross(a, b, c):
        return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    def on(a, b, c):
        return abs(cross(a,b,c)) <= 1e-8 and all(min(a[k],b[k])-1e-8 <= c[k] <= max(a[k],b[k])+1e-8 for k in (0,1))
    def intersects(a,b,c,d):
        x,y,z,t = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
        return (x*y < 0 and z*t < 0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)
    for i, a in enumerate(p):
        b = p[(i+1)%len(p)]
        for j in range(i+1,len(p)):
            if j == i+1 or (i == 0 and j == len(p)-1):
                continue
            if intersects(a,b,p[j],p[(j+1)%len(p)]):
                raise ValueError('Contour self-intersects')
        if abs(cross(p[i-1], a, b)) <= 1e-8 and on(p[i-1], a, b):
            raise ValueError('Contour doubles back')
    if abs(cv2.contourArea(p.astype(np.float32))) <= 1e-8:
        raise ValueError('Degenerate contour')
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 1)
    if mask.sum() < 4:
        raise ValueError('Degenerate rasterized contour')
    return mask.astype(bool)
