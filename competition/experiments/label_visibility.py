"""Are the remaining defect labels visible in the pixels at all?

spine_axis was closed with one question: train an exactly-supervised regressor on
synthetic rotations and ask whether the labelled positives actually show the
defect. They did not, and the criterion was unrecoverable.

This applies the same test to the other four image-level criteria, using oracle
measurements that do not depend on any learned model:

  margins    - distance in millimetres from the frame edge to the nearest bone
               pixel, per side. Drives spine_coverage and hip_roi_coverage.
  occupancy  - fraction of bone area, and bone area inside the outer band of the
               frame. Drives coverage again, from the opposite direction.
  brightness - share of saturated pixels and of top-hat response, i.e. metal,
               clips, zips and other foreign objects. Drives spine_artifact and
               hip_metal_implant.
  texture    - local contrast energy, the photometric handle on noise and
               streaking.

A criterion is only worth modelling if at least one of its measurements separates
the labels. This reports that directly, per criterion, with bootstrap intervals,
and does not fit anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.dicom_io import read_any  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from gpu_research.common import dump  # noqa: E402


def bone_mask(pixels, threshold):
    return pixels >= threshold


def measurements(pixels, pixel_mm_x, pixel_mm_y, band_fraction):
    """Oracle geometry and photometry for one frame. No labels are used."""
    h, w = pixels.shape
    bone = bone_mask(pixels, int(np.percentile(pixels, 92)))
    values = np.where(bone, pixels, 0).astype(np.int32)
    # Distance from each edge to the nearest bone pixel, in millimetres.
    rows_any = bone.any(axis=1)
    cols_any = bone.any(axis=0)
    if not rows_any.any() or not cols_any.any():
        margins = {k: np.nan for k in ('top_mm', 'bottom_mm', 'left_mm', 'right_mm')}
    else:
        top_row = int(np.argmax(rows_any))
        bottom_row = int(h - 1 - np.argmax(rows_any[::-1]))
        left_col = int(np.argmax(cols_any))
        right_col = int(w - 1 - np.argmax(cols_any[::-1]))
        margins = {'top_mm': top_row * pixel_mm_y, 'bottom_mm': (h - 1 - bottom_row) * pixel_mm_y,
                   'left_mm': left_col * pixel_mm_x, 'right_mm': (w - 1 - right_col) * pixel_mm_x}
    band = max(1, int(round(min(h, w) * band_fraction)))
    inner = bone[band:h - band, band:w - band]
    outer = int(bone.sum() - inner.sum())
    total = max(int(bone.sum()), 1)

    saturated = float((pixels >= 250).mean())
    bright = float((pixels >= 235).mean())
    # Top-hat response: bright compact structures that are not part of the bone
    # field. Foreign objects are exactly this.
    from scipy.ndimage import grey_closing, grey_opening
    closed = grey_closing(pixels, size=max(3, int(round(min(h, w) * 0.02)) | 1))
    tophat = (closed - pixels).astype(np.int32)
    tophat_strong = float((tophat >= 25).mean())
    tophat_max = int(tophat.max())
    opened = grey_opening(pixels, size=max(3, int(round(min(h, w) * 0.02)) | 1))
    blackhat = (pixels - opened).astype(np.int32)
    blackhat_strong = float((blackhat >= 25).mean())

    dy = np.diff(pixels.astype(np.float32), axis=1)
    dx = np.diff(pixels.astype(np.float32), axis=0)
    texture = float(np.sqrt((dx ** 2).mean() + (dy ** 2).mean()))

    out = dict(margins)
    out.update({'bone_fraction': total / float(h * w),
                'bone_in_outer_band_fraction': outer / total,
                'saturated_fraction': saturated, 'bright_fraction': bright,
                'tophat_strong_fraction': tophat_strong, 'tophat_max': tophat_max,
                'blackhat_strong_fraction': blackhat_strong, 'texture_energy': texture,
                'height_mm': h * pixel_mm_y, 'width_mm': w * pixel_mm_x})
    out['min_margin_mm'] = float(np.nanmin(list(margins.values())))
    return out


def auc(y, score):
    """Rank AUC with tie handling, plus a bootstrap interval."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    point = float(roc_auc_score(y[ok], score[ok]))
    rng = np.random.default_rng(17)
    draws = []
    for _ in range(1000):
        pick = rng.integers(0, ok.sum(), ok.sum())
        yy = y[ok][pick]
        if len(np.unique(yy)) < 2:
            continue
        draws.append(roc_auc_score(yy, score[ok][pick]))
    return {'auc': point, 'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            'defined_resamples': len(draws)}


def run(args):
    from sklearn.metrics import roc_auc_score
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}
    rows = []
    for i, row in enumerate(labels.itertuples()):
        image = read_any(args.dataset / row.first_source_path)
        if image.pixel_sha256 != audited[row.first_source_path]:
            raise ValueError('Source pixels changed after audit')
        record = measurements(image.pixels, image.pixel_mm_x, image.pixel_mm, args.band_fraction)
        record['path'] = row.first_source_path
        record['region'] = row.region
        record['quality_class'] = row.quality_class
        for criterion in ('spine_coverage', 'spine_axis', 'spine_artifact',
                          'hip_position_rotation', 'hip_roi_coverage'):
            record[criterion] = pd.to_numeric(getattr(row, criterion, np.nan), errors='coerce')
        rows.append(record)
        if (i + 1) % 50 == 0:
            print(f'measured {i + 1}/{len(labels)}', flush=True)
    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'oracle_measurements.csv', index=False)

    numeric = [c for c in table.columns if c not in
               ('path', 'region', 'quality_class', 'spine_coverage', 'spine_axis',
                'spine_artifact', 'hip_position_rotation', 'hip_roi_coverage')]
    report = {'protocol': __doc__, 'rows': int(len(table)),
              'criteria': {}, 'best_measurement_per_criterion': {}}
    for criterion in ('spine_coverage', 'hip_roi_coverage', 'spine_artifact', 'hip_position_rotation'):
        target = table[criterion].to_numpy(float)
        mask = np.isfinite(target) & table['region'].isin(
            ('spine',) if criterion.startswith('spine') else ('hip_left', 'hip_right')).to_numpy()
        if mask.sum() < 10:
            continue
        y = (target[mask] == 1).astype(int)
        per_measurement = {}
        for column in numeric:
            score = table[column].to_numpy(float)[mask]
            # Report the better of the two directions; visibility is direction-free.
            forward = auc(y, score)
            if forward is None:
                continue
            reverse = auc(y, -score)
            per_measurement[column] = {'auc': forward['auc'], 'ci95': forward['ci95'],
                                       'auc_flipped': reverse['auc'],
                                       'best_auc': max(forward['auc'], reverse['auc']),
                                       'separates': bool(min(forward['auc'], reverse['auc']) > 0.6
                                                          or max(forward['auc'], reverse['auc']) > 0.75)}
        ranked = sorted(per_measurement.items(), key=lambda kv: -kv[1]['best_auc'])
        report['criteria'][criterion] = {
            'n': int(mask.sum()), 'positives': int(y.sum()),
            'per_measurement': per_measurement}
        report['best_measurement_per_criterion'][criterion] = {
            'measurement': ranked[0][0], 'best_auc': ranked[0][1]['best_auc'],
            'ci95_forward': ranked[0][1]['ci95'],
            'any_measurement_separates': any(v['separates'] for v in per_measurement.values())}

    # Directly test whether the shipped geometry block already covers the margins.
    for criterion in ('spine_coverage', 'hip_roi_coverage'):
        if criterion not in report['criteria']:
            continue
        block = ('min_margin_mm', 'left_mm', 'right_mm')
        present = {k: report['criteria'][criterion]['per_measurement'][k]['best_auc']
                   for k in block if k in report['criteria'][criterion]['per_measurement']}
        report['criteria'][criterion]['margin_subset_auc'] = present

    dump(args.output / 'visibility_report.json', report)
    print('\n=== is the label visible in the pixels? ===')
    for criterion, entry in report['best_measurement_per_criterion'].items():
        print(f"{criterion:24} n={report['criteria'][criterion]['n']:3} pos={report['criteria'][criterion]['positives']:3} "
              f"best={entry['measurement']:22} AUC={entry['best_auc']:.3f} "
              f"separates={entry['any_measurement_separates']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--band-fraction', type=float, default=0.08)
    run(parser.parse_args())