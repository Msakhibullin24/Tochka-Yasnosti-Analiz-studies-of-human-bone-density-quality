"""Explainable geometric measurements for DXA quality control.

All functions take an 8-bit grayscale image (bone bright on dark) and the pixel size in mm.
Hip images must be passed in canonical orientation: femoral shaft on the image left,
pelvis on the right (left hips are mirrored by the caller).
Every measurement returns plain floats so it can feed both rules and a classifier,
plus overlay primitives used for the visual explanation series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_PIXEL_MM = 0.607  # Legacy uncertain fallback for uncalibrated inputs; NOT the organiser V2 scale


@dataclass
class Measurement:
    features: dict[str, float]
    overlay: dict[str, list] = field(default_factory=dict)


def measure_image(pixels: np.ndarray, region: str, pixel_mm_y: float,
                  pixel_mm_x: float | None = None) -> Measurement:
    """Measure in an isotropic physical grid; return points in input pixel coordinates.

    The caller supplies canonical hip orientation. CNN inputs are not resampled.
    """
    sx = pixel_mm_y if pixel_mm_x is None else pixel_mm_x
    if not (np.isfinite(sx) and np.isfinite(pixel_mm_y) and sx > 0 and pixel_mm_y > 0):
        raise ValueError("invalid physical pixel spacing")
    width = max(1, round(pixels.shape[1] * sx / pixel_mm_y))
    if width * pixels.shape[0] > 36_000_000:
        raise ValueError("physical grid exceeds image size limit")
    scale = width / pixels.shape[1]
    physical = pixels if width == pixels.shape[1] else cv2.resize(
        pixels, (width, pixels.shape[0]), interpolation=cv2.INTER_LINEAR)
    result = (measure_spine if region == "spine" else measure_hip)(physical, pixel_mm_y)
    if scale != 1:
        result.overlay = {key: [((x + 0.5) / scale - 0.5, y) for x, y in points]
                          for key, points in result.overlay.items()}
    return result


def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)


def _otsu(img: np.ndarray) -> float:
    t, _ = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _robust_line(y: np.ndarray, x: np.ndarray, w: np.ndarray | None = None, iters: int = 6) -> tuple[float, float]:
    """Fit x = a*y + b with iteratively reweighted least squares (Tukey-like)."""
    w = np.ones_like(x, dtype=np.float64) if w is None else w.astype(np.float64).copy()
    a = b = 0.0
    for _ in range(iters):
        if w.sum() <= 0:
            break
        a, b = np.polyfit(y, x, 1, w=np.sqrt(w + 1e-9))
        r = x - (a * y + b)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = np.where(np.abs(r) < 2.5 * s, 1.0, 0.05)
    return float(a), float(b)


# --------------------------------------------------------------------------------------
# Spine
# --------------------------------------------------------------------------------------

def spine_centerline(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the vertebral column centre per row with a dynamic-programming ridge tracker."""
    h, w = img.shape
    s = _smooth(img, 2.0)
    band = max(9, int(round(w * 0.2)) | 1)
    # Horizontal box filter ~ vertebral width: the column is the widest bright band.
    ridge = cv2.blur(s, (band, 1))
    ridge = cv2.GaussianBlur(ridge, (0, 0), sigmaX=1.0, sigmaY=4.0)
    cost = -(ridge / (ridge.max() + 1e-6))
    # Prior: column is near the horizontal centre.
    xs = np.arange(w)
    cost = cost + 0.15 * ((xs - w / 2) / (w / 2)) ** 2
    max_step, penalty = 2, 0.02
    acc = cost.copy()
    back = np.zeros((h, w), np.int8)
    for y in range(1, h):
        best = np.full(w, np.inf, np.float32)
        arg = np.zeros(w, np.int8)
        for d in range(-max_step, max_step + 1):
            shifted = np.roll(acc[y - 1], d)
            if d > 0:
                shifted[:d] = np.inf
            elif d < 0:
                shifted[d:] = np.inf
            cand = shifted + penalty * abs(d)
            better = cand < best
            best[better] = cand[better]
            arg[better] = d
        acc[y] += best
        back[y] = arg
    x = np.zeros(h, np.int32)
    x[-1] = int(np.argmin(acc[-1]))
    for y in range(h - 1, 0, -1):
        x[y - 1] = x[y] - back[y, x[y]]
    ys = np.arange(h)
    strength = ridge[ys, x]
    return ys, x.astype(np.float64), strength


def measure_spine(img: np.ndarray, pixel_mm: float = DEFAULT_PIXEL_MM) -> Measurement:
    h, w = img.shape
    ys, xc, strength = spine_centerline(img)
    # Use the lumbar part: skip the top 8% (T12/ribs) and the bottom 18% (sacrum/pelvis).
    lo, hi = int(h * 0.08), int(h * 0.82)
    a, b = _robust_line(ys[lo:hi].astype(float), xc[lo:hi], strength[lo:hi])
    angle = float(np.degrees(np.arctan(a)))
    resid = xc[lo:hi] - (a * ys[lo:hi] + b)
    # Piecewise angles catch a tilted segment inside a curved (scoliotic) spine.
    seg_angles = []
    n_seg = 3
    edges = np.linspace(lo, hi, n_seg + 1).astype(int)
    for i in range(n_seg):
        sa, _ = _robust_line(ys[edges[i]:edges[i + 1]].astype(float), xc[edges[i]:edges[i + 1]])
        seg_angles.append(float(np.degrees(np.arctan(sa))))
    centre_offset_mm = float((np.median(xc[lo:hi]) - w / 2) * pixel_mm)

    s = _smooth(img, 1.5)
    thr = max(_otsu(img), 40.0)
    half_band = int(round(28 / pixel_mm / 2)) + 6  # vertebral body ~ 40-50 mm wide incl. processes
    # Iliac crests: bone lateral to the column in the lowest part of the image.
    crest = {}
    bottom = slice(int(h * 0.80), h)
    for side, cols in (("left", slice(0, max(1, int(np.median(xc[bottom]) - half_band * 1.6)))),
                       ("right", slice(min(w - 1, int(np.median(xc[bottom]) + half_band * 1.6)), w))):
        zone = s[bottom, cols]
        crest[side] = float((zone > thr).mean()) if zone.size else 0.0
    # Height of the highest crest pixel above the lower image border (mm).
    lateral = np.ones((h, w), bool)
    for y in range(h):
        lateral[y, max(0, int(xc[y] - half_band * 1.6)):min(w, int(xc[y] + half_band * 1.6))] = False
    lower = (s > thr) & lateral
    lower[: int(h * 0.55)] = False
    rows_with = np.where(lower.sum(1) >= 4)[0]
    crest_height_mm = float((h - rows_with.min()) * pixel_mm) if rows_with.size else 0.0
    # Rib / T12 evidence: lateral structure in the top of the image.
    top_zone = s[: int(h * 0.15)]
    top_lat = top_zone[lateral[: int(h * 0.15)]]
    rib_signal = float(np.percentile(top_lat, 90)) if top_lat.size else 0.0

    # Artefacts: saturated blobs and thin very bright structures (white top-hat).
    sat = (img >= 250).astype(np.uint8)
    n_sat, _, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    sat_areas = stats[1:, cv2.CC_STAT_AREA] if n_sat > 1 else np.array([0])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    th_lat = tophat[lateral]
    th_col = tophat[~lateral]
    col_vals = img[~lateral]
    # Underwires / clips sit in the upper part of the frame: measure that zone separately.
    top_rows = int(h * 0.15)
    th_top = tophat[:top_rows][lateral[:top_rows]]
    features = {
        "spine_angle_deg": angle,
        "spine_abs_angle_deg": abs(angle),
        "spine_seg_max_abs_angle_deg": float(np.max(np.abs(seg_angles))),
        "spine_seg_angle_range_deg": float(np.max(seg_angles) - np.min(seg_angles)),
        "spine_curve_rms_mm": float(np.sqrt(np.mean(resid ** 2)) * pixel_mm),
        "spine_curve_max_mm": float(np.max(np.abs(resid)) * pixel_mm),
        "spine_centre_offset_mm": centre_offset_mm,
        "spine_abs_centre_offset_mm": abs(centre_offset_mm),
        "crest_left_frac": crest["left"],
        "crest_right_frac": crest["right"],
        "crest_min_frac": min(crest.values()),
        "crest_height_mm": crest_height_mm,
        "rib_signal": rib_signal,
        "image_height_mm": h * pixel_mm,
        "sat_pixels": float(sat.sum()),
        "sat_max_blob": float(sat_areas.max()),
        "tophat_lat_p999": float(np.percentile(th_lat, 99.9)) if th_lat.size else 0.0,
        "tophat_lat_p99": float(np.percentile(th_lat, 99)) if th_lat.size else 0.0,
        "tophat_col_p999": float(np.percentile(th_col, 99.9)) if th_col.size else 0.0,
        "tophat_lat_strong_frac": float((th_lat > 60).mean()) if th_lat.size else 0.0,
        "tophat_top_lat_p999": float(np.percentile(th_top, 99.9)) if th_top.size else 0.0,
        "tophat_top_lat_p99": float(np.percentile(th_top, 99)) if th_top.size else 0.0,
        "tophat_top_strong_frac": float((th_top > 60).mean()) if th_top.size else 0.0,
        "column_p99": float(np.percentile(col_vals, 99)) if col_vals.size else 0.0,
        "column_mean": float(col_vals.mean()) if col_vals.size else 0.0,
        "lateral_mean": float(img[lateral].mean()),
    }
    overlay = {
        "centerline": [(float(x), float(y)) for y, x in zip(ys[::4], xc[::4])],
        "axis": [(float(a * lo + b), float(lo)), (float(a * hi + b), float(hi))],
    }
    return Measurement(features, overlay)


# --------------------------------------------------------------------------------------
# Hip (canonical orientation: shaft left, pelvis right)
# --------------------------------------------------------------------------------------

def measure_hip(img: np.ndarray, pixel_mm: float = DEFAULT_PIXEL_MM) -> Measurement:
    h, w = img.shape
    s = _smooth(img, 1.5)
    thr = max(_otsu(img) * 0.8, 30.0)
    bone = (s > thr).astype(np.uint8)
    bone = cv2.morphologyEx(bone, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Femur contour: start from the widest bone run at the bottom (the shaft) and track it upward
    # by overlap, so the pelvis / ischium never get confused with the femur.
    def runs(y):
        d = np.diff(np.concatenate([[0], bone[y], [0]]))
        return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

    lat, med = np.full(h, np.nan), np.full(h, np.nan)
    prev = None
    min_w = max(6, int(8 / pixel_mm))
    for y in range(h - 1, -1, -1):
        cand = [r for r in runs(y) if r[1] - r[0] >= min_w]
        if prev is None:
            cand = [r for r in cand if r[0] < w * 0.75]
            if not cand:
                if y < h * 0.7:
                    break
                continue
            prev = max(cand, key=lambda r: r[1] - r[0])
        else:
            over = [r for r in cand if min(r[1], prev[1]) - max(r[0], prev[0]) > 0]
            if not over:
                break
            # The femur is the lateral-most overlapping structure in canonical orientation.
            prev = min(over, key=lambda r: r[0])
        lat[y], med[y] = prev
    rows = np.where(~np.isnan(lat))[0]
    feats: dict[str, float] = {}
    overlay: dict[str, list] = {}
    shaft_rows = rows[rows >= int(h * 0.80)]
    if shaft_rows.size >= 8:
        mid = (lat[shaft_rows] + med[shaft_rows]) / 2
        a, b = _robust_line(shaft_rows.astype(float), mid)
        feats["shaft_angle_deg"] = float(np.degrees(np.arctan(a)))
        feats["shaft_width_mm"] = float(np.median(med[shaft_rows] - lat[shaft_rows]) * pixel_mm)
        # Medial cortex line from the distal shaft, extrapolated upward: the lesser trochanter
        # is the medial bulge beyond this line below the femoral neck.
        # Reference = shaft axis shifted to the medial cortex (robust to the symmetric shaft flare).
        am, bm = a, b + float(np.median(med[shaft_rows] - mid))
        search = rows[(rows >= int(h * 0.40)) & (rows < int(h * 0.85))]
        prot = med[search] - (am * search + bm)
        # The neck/head merge medially at the top of the search window; stop at the first row
        # (bottom-up) where protrusion exceeds a shaft width (that is the neck, not the trochanter).
        limit = feats["shaft_width_mm"] / pixel_mm * 0.6
        keep = []
        for yy, p in zip(search[::-1], prot[::-1]):
            if p > limit:
                break
            keep.append((yy, p))
        if len(keep) >= 6:
            kp = np.array(keep, dtype=np.float64)  # ordered bottom-up: shaft -> neck start
            ky, kprot = kp[:, 0], cv2.GaussianBlur(kp[:, 1].reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
            # Chord from the shaft to the start of the neck. A visible lesser trochanter bulges
            # medially beyond the chord; an over-rotated femur gives a smooth concave arc under it.
            chord = kprot[0] + (kprot[-1] - kprot[0]) * (ky - ky[0]) / (ky[-1] - ky[0] - 1e-9)
            bulge = kprot - chord
            feats["lesser_troch_protrusion_mm"] = float(max(bulge.max(), 0) * pixel_mm)
            feats["lesser_troch_area_mm2"] = float(np.clip(bulge, 0, None).sum() * pixel_mm ** 2)
            feats["medial_concavity_mm"] = float(max(-bulge.min(), 0) * pixel_mm)
            feats["neck_start_height_mm"] = float((ky[0] - ky[-1]) * pixel_mm)
            # Local prominence: peak followed (upwards) by a notch before the neck.
            ipk = int(np.argmax(bulge))
            notch = float(kprot[ipk] - kprot[ipk:].min())
            feats["lesser_troch_notch_mm"] = notch * pixel_mm
            ymax = int(ky[ipk])
            if feats["lesser_troch_protrusion_mm"] >= 1.0:  # no marker when the trochanter is not visible
                overlay["lesser_trochanter"] = [(float(med[ymax]), float(ymax))]
        overlay["shaft_axis"] = [(float(a * shaft_rows.min() + b), float(shaft_rows.min())),
                                 (float(a * (h - 1) + b), float(h - 1))]
    # Greater trochanter top: follow the lateral femur contour upward from its most lateral point;
    # the trochanter ends where the contour turns medially by more than a shaft width or the run ends.
    if rows.size:
        up = rows[rows < int(h * 0.75)]
        if up.size:
            y_lat = int(up[np.nanargmin(lat[up])])
            y_gt = y_lat
            for y in range(y_lat, -1, -1):
                if np.isnan(lat[y]) or lat[y] > lat[y_lat] + feats.get("shaft_width_mm", 26.0) / pixel_mm:
                    break
                y_gt = y
            feats["troch_top_margin_mm"] = float(y_gt * pixel_mm)
            feats["lateral_margin_mm"] = float(lat[y_lat] * pixel_mm)
            overlay["greater_trochanter_top"] = [(float(lat[y_gt] + 4), float(y_gt))]
            overlay["greater_trochanter_lateral"] = [(float(lat[y_lat]), float(y_lat))]
    # Ischium bottom: lowest bone pixel in the medial (right) 35% of the image.
    right = bone[:, int(w * 0.65):]
    rrows = np.where(right.sum(1) >= 3)[0]
    if rrows.size:
        feats["ischium_bottom_margin_mm"] = float((h - 1 - rrows.max()) * pixel_mm)
        feats["ischium_touches_bottom"] = float(rrows.max() >= h - 3)
        overlay["ischium_bottom"] = [(float(int(w * 0.65) + np.where(right[rrows.max()])[0].mean()), float(rrows.max()))]
    feats["bone_top_rows_frac"] = float(bone[:4].mean())
    feats["bone_frac"] = float(bone.mean())
    feats["image_height_mm"] = h * pixel_mm
    feats["image_width_mm"] = w * pixel_mm
    feats["sat_pixels"] = float((img >= 250).sum())
    feats["sat_frac"] = float((img >= 250).mean())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    feats["tophat_p999"] = float(np.percentile(tophat, 99.9))
    # Blank rectangular blocks (GE export masks part of the pelvis) are normal; report only.
    feats["zero_frac"] = float((img == 0).mean())
    for k in ("shaft_angle_deg", "shaft_width_mm", "lesser_troch_protrusion_mm", "lesser_troch_area_mm2",
              "medial_concavity_mm", "neck_start_height_mm", "lesser_troch_notch_mm",
              "lateral_margin_mm", "troch_top_margin_mm", "ischium_bottom_margin_mm", "ischium_touches_bottom"):
        feats.setdefault(k, float("nan"))
    feats["shaft_abs_angle_deg"] = abs(feats["shaft_angle_deg"])
    return Measurement(feats, overlay)
