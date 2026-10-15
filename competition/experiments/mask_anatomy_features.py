"""Pixel/physical statistics from independently pretrained anatomical masks.

The builder never reads QC targets. Missing anatomy remains missing; masks are
candidate measurements, not confirmed landmarks or visibility ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from source_integrity import inspect_sources
from gpu_research.common import fingerprint


def largest_component(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    return cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def mask_statistics(mask, pixels, sx, sy):
    mask = largest_component(mask)
    h, w = pixels.shape
    if mask.shape != pixels.shape:
        raise ValueError('Mask/image dimensions differ')
    ys, xs = np.where(mask)
    area = len(xs)
    if area < max(4, .001 * h * w):
        return {'present': 0.}, None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    center = (float(xs.mean()), float(ys.mean()))
    result = {'present': 1., 'area_fraction': area / (h * w), 'area_mm2': area * sx * sy,
              'center_x': center[0] / w, 'center_y': center[1] / h,
              'width_mm': float((x1 - x0) * sx), 'height_mm': float((y1 - y0) * sy),
              'top_mm': float(y0 * sy), 'bottom_mm': float((h - y1) * sy),
              'left_mm': float(x0 * sx), 'right_mm': float((w - x1) * sx),
              'touch_top': float(y0 == 0), 'touch_bottom': float(y1 == h),
              'touch_left': float(x0 == 0), 'touch_right': float(x1 == w),
              'intensity_mean': float(pixels[mask].mean() / 255.),
              'intensity_std': float(pixels[mask].std() / 255.)}
    return result, center


def spine_features(masks, pixels, sx, sy):
    if masks.shape != (26, *pixels.shape):
        raise ValueError('Expected 26 independent TSXR channels')
    result, centers = {}, {}
    # Published non-background channel indices: sacrum=0, S1=1, L5=2, ..., T12=7.
    for name, channel in [('sacrum', 0), ('L5', 2), ('L4', 3), ('L3', 4),
                          ('L2', 5), ('L1', 6), ('Th12', 7), ('Th11', 8)]:
        stats, center = mask_statistics(masks[channel], pixels, sx, sy)
        result.update({f'anat_{name}_{k}': v for k, v in stats.items()})
        if center is not None:
            centers[name] = center
    lumbar = [centers.get(f'L{i}') for i in range(1, 5)]
    ordered = bool(all(p is not None for p in lumbar) and
                   np.all(np.diff(np.asarray(lumbar)[:, 1]) > 0))
    result['anat_lumbar_ordered'] = float(ordered)
    angle = None
    if ordered:
        points = np.asarray(lumbar) * [sx, sy]
        a, b = np.polyfit(points[:, 1], points[:, 0], 1)
        angle = float(np.degrees(np.arctan(a)))
        result['anat_lumbar_angle_abs_deg'] = abs(angle)
        result['anat_lumbar_line_residual_mm'] = float(np.sqrt(np.mean((points[:, 0] - (a * points[:, 1] + b))**2)))
        for i in range(3):
            delta = points[i+1] - points[i]
            result[f'anat_lumbar_pair_{i+1}_angle_abs_deg'] = float(abs(np.degrees(np.arctan2(delta[0], delta[1]))))
            result[f'anat_lumbar_pair_{i+1}_height_mm'] = float(delta[1])
    return result, angle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output file')
    probe = json.loads((args.probe / 'probe.json').read_text())
    label_hash = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    if probe.get('labels_sha256') != label_hash:
        raise ValueError('Probe label identity is missing or stale; re-extract masks')
    source = inspect_sources([('organiser', args.labels, args.dataset)])
    references = {r['pixel_sha256']: r for r in probe['results'] if r['polarity'] == 'bone-bright'}
    table = pd.read_csv(args.labels)
    entries = {}
    for row in table.loc[table.region.eq('spine')].itertuples():
        image = read_dxa(args.dataset / row.first_source_path)
        reference = references[row.pixel_sha256]
        source_hash = hashlib.sha256((args.dataset / row.first_source_path).read_bytes()).hexdigest()
        path = args.probe / f'{row.pixel_sha256[:16]}-bone-bright.npy'
        if (image.pixel_sha256 != reference['decoded_pixel_sha256'] or
                source_hash != reference['source_file_sha256'] or
                hashlib.sha256(path.read_bytes()).hexdigest() != reference['mask_sha256']):
            raise ValueError('Probe source or mask identity mismatch')
        masks = np.load(path)
        features, angle = spine_features(masks, image.pixels, image.pixel_mm_x, image.pixel_mm)
        entries[row.pixel_sha256] = {'features': features, 'numbered_axis_degrees': angle}
    columns = sorted({k for e in entries.values() for k in e['features']})
    report = {'schema_version': 1, 'labels_sha256': label_hash,
              'source_fingerprint': fingerprint(source['entries']),
              'checkpoint_sha256': probe['checkpoint_sha256'], 'clinical_validation': False,
              'training_on_organizer_labels': False, 'entries': entries,
              'columns_by_group': {'spine': columns, 'hip': []}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'images': len(entries), 'features': len(columns)}))


if __name__ == '__main__':
    main()
