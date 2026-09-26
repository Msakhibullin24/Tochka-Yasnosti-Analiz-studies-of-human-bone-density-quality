"""Bounded exact binary masks in original-image coordinates."""
import numpy as np

MAX_PIXELS = 36_000_000
MAX_RUNS = 65_536


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError('Raster coordinates must be integers')
    return int(value)


def encode_raster(mask, origin=(0, 0)):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or mask.size > MAX_PIXELS or not np.isin(mask, (0, 1)).all():
        raise ValueError('Invalid binary mask')
    if len(origin) != 2:
        raise ValueError('Invalid raster origin')
    origin = [_integer(v) for v in origin]
    flat = mask.astype(np.int8).reshape(-1)
    edges = np.diff(np.r_[0, flat, 0])
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if len(starts) > MAX_RUNS:
        raise ValueError('Raster run count exceeds limit')
    return {'shape': list(mask.shape), 'origin': origin, 'runs': np.column_stack((starts, stops)).tolist()}


def decode_raster(raster, shape):
    try:
        if len(shape) != 2 or len(raster['shape']) != 2 or len(raster['origin']) != 2:
            raise ValueError('Invalid raster dimensions')
        h, w = [_integer(v) for v in shape]
        rh, rw = [_integer(v) for v in raster['shape']]
        y, x = [_integer(v) for v in raster['origin']]
        if min(h, w, rh, rw) < 1 or h*w > MAX_PIXELS or rh*rw > MAX_PIXELS:
            raise ValueError('Raster dimensions exceed limits')
        runs = raster['runs']
        if not isinstance(runs, (list, tuple)) or len(runs) > MAX_RUNS:
            raise ValueError('Invalid raster runs')
        checked, previous = [], 0
        for pair in runs:
            if len(pair) != 2:
                raise ValueError('Invalid raster run')
            start, end = [_integer(v) for v in pair]
            if not 0 <= previous <= start < end <= rh*rw:
                raise ValueError('Overlapping or out-of-bounds raster run')
            checked.append((start, end)); previous = end
        raw = np.zeros((rh, rw), bool)
        for start, end in checked:
            raw.ravel()[start:end] = True
        result = np.zeros((h, w), bool)
        y0, y1 = max(0, y), min(h, y+rh)
        x0, x1 = max(0, x), min(w, x+rw)
        if y0 < y1 and x0 < x1:
            result[y0:y1, x0:x1] = raw[y0-y:y1-y, x0-x:x1-x]
        return result
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid raster representation') from exc
