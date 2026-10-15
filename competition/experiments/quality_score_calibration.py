"""Training-only monotone probability calibration of the fixed QC ranking score."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

MODES = ('shared', 'region_intercept', 'region_slope', 'region_axis')


def design(scores, groups, angles, mode):
    scores = np.asarray(scores, float).reshape(-1)
    groups = np.asarray(groups)
    angles = np.asarray(angles, float)
    if mode not in MODES or groups.shape != scores.shape or angles.shape != scores.shape:
        raise ValueError('Calibration feature layout differs')
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Finite ranking scores in [0,1] required')
    if not np.isin(groups, ('spine', 'hip')).all(): raise ValueError('Unknown anatomical group')
    clipped = np.clip(scores, 1e-6, 1-1e-6)
    x = np.log(clipped/(1-clipped))
    hip = (groups == 'hip').astype(float)
    columns = [x]
    if mode != 'shared': columns.append(hip)
    if mode == 'region_slope': columns.append(hip*x)
    if mode == 'region_axis': columns.append(((hip == 0) & np.isfinite(angles) & (abs(angles) > 5)).astype(float))
    return np.column_stack(columns)


def fit_calibration(scores, groups, angles, targets, mode):
    x = design(scores, groups, angles, mode)
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=17).fit(x, targets)
    slopes = [model.coef_[0, 0]]
    if mode == 'region_slope': slopes.append(model.coef_[0, 0] + model.coef_[0, 2])
    if min(slopes) <= 0:
        raise ValueError('Calibration is not monotone within anatomical group and physical-axis state')
    return {'mode': mode, 'model': model, 'monotone_in_base_score': True, 'C': 1., 'clip': 1e-6}


def calibrate(calibration, scores, groups, angles):
    x = design(scores, groups, angles, calibration['mode'])
    return calibration['model'].predict_proba(x)[:, 1]


def calibrated_cut(calibration, cutoff, group, angle):
    if cutoff > 1 or cutoff < 0: return float(cutoff)
    return float(calibrate(calibration, [cutoff], [group], [angle])[0])
