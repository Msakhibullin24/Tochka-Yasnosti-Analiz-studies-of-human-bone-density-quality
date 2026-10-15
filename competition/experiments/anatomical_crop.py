"""Fixed, target-free crop around the proximal 120 mm of a predicted femur."""
import numpy as np

POLICY = 'hip_proximal120mm_pad20mm_v1'


def hip_crop_bounds(femur, sx, sy):
    mask = np.asarray(femur, bool)
    if mask.ndim != 2 or not all(np.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError('Calibrated 2D mask required')
    yy, xx = np.where(mask)
    if (len(xx) < .05 * mask.size or not len(xx) or (xx.max() - xx.min() + 1) * sx < 25
            or (yy.max() - yy.min() + 1) * sy < 70):
        return None
    ytop = int(yy.min())
    proximal = xx[yy <= ytop + np.ceil(120 / sy)]
    height, width = mask.shape
    return [max(0, int(np.floor(proximal.min() - 20 / sx))),
            max(0, int(np.floor(ytop - 20 / sy))),
            min(width, int(np.ceil(proximal.max() + 1 + 20 / sx))),
            min(height, int(np.ceil(ytop + 140 / sy)))]
