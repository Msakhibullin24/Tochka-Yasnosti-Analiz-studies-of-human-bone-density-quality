"""Measure hip geometry stability under mild image-processing perturbations.

This is a technical invariance audit, not a clinical validation or retraining.
Per-image paths are written only to the optional local case file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dxaqc.anatomy import detect_landmarks
from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import measure_image
from dxaqc.model import VIOLATION_LABEL

VARIANTS = ('dim', 'bright', 'resize')
POINTS = ('greater_trochanter', 'lesser_trochanter', 'femoral_neck', 'ischium')
FEATURES = ('shaft_abs_angle_deg', 'lesser_troch_protrusion_mm')


def perturb(pixels: np.ndarray, name: str) -> np.ndarray:
    if name == 'dim':
        return np.clip(pixels.astype(np.float32) * .9 + 3, 0, 255).astype(np.uint8)
    if name == 'bright':
        return np.clip(pixels.astype(np.float32) * 1.1 - 3, 0, 255).astype(np.uint8)
    if name == 'resize':
        h, w = pixels.shape
        small = cv2.resize(pixels, (max(1, round(w*.9)), max(1, round(h*.9))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f'unknown perturbation: {name}')


def evidence(pixels, region, pixel_mm_y, pixel_mm_x):
    canonical = np.ascontiguousarray(pixels[:, ::-1]) if region == 'hip_left' else pixels
    measurement = measure_image(canonical, region, pixel_mm_y, pixel_mm_x)
    anatomy = detect_landmarks(pixels, region, measurement.overlay, pixel_mm_y, pixel_mm_x)
    return measurement.features, {item['name']: item['points'] for item in anatomy['landmarks']}


def compare(base, altered, pixel_mm_y, pixel_mm_x):
    bf, bp = base
    af, ap = altered
    result = {}
    for name in FEATURES:
        a, b = bf[name], af[name]
        result[name + '_delta'] = abs(float(a-b)) if np.isfinite(a) and np.isfinite(b) else None
    for name in POINTS:
        first, second = bp[name], ap[name]
        result[name + '_status_flip'] = bool(first) != bool(second)
        result[name + '_shift_mm'] = (float(np.linalg.norm((np.array(first[0])-second[0]) *
                                                          [pixel_mm_x, pixel_mm_y]))
                                      if first and second else None)
    return result


def summarize(cases):
    result = {}
    for variant in VARIANTS:
        result[variant] = {}
        for group in ('all', 'tp', 'fn', 'fp', 'tn'):
            items = [c['variants'][variant] for c in cases if group == 'all' or c['group'] == group]
            block = {'images': len(items)}
            for name in FEATURES:
                vals = [item[name + '_delta'] for item in items if item[name + '_delta'] is not None]
                block[name + '_delta_p95'] = float(np.percentile(vals, 95)) if vals else None
                block[name + '_delta_max'] = float(max(vals)) if vals else None
                block[name + '_unavailable'] = len(items) - len(vals)
            for name in POINTS:
                vals = [item[name + '_shift_mm'] for item in items if item[name + '_shift_mm'] is not None]
                block[name + '_status_flips'] = sum(item[name + '_status_flip'] for item in items)
                block[name + '_shift_p95_mm'] = float(np.percentile(vals, 95)) if vals else None
                block[name + '_shift_max_mm'] = float(max(vals)) if vals else None
                block[name + '_review_over_5mm_or_flip'] = sum(
                    item[name + '_status_flip'] or
                    (item[name + '_shift_mm'] is not None and item[name + '_shift_mm'] > 5)
                    for item in items)
                block[name + '_paired_candidates'] = len(vals)
            result[variant][group] = block
    return result


def audit(oof_path, labels_path, dataset, repeat):
    with labels_path.open(newline='', encoding='utf-8-sig') as stream:
        labels = {row['first_source_path']: row for row in csv.DictReader(stream)
                  if row['region'].startswith('hip') and row['hip_position_rotation'] in ('0', '1')}
    with oof_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = [row for row in csv.DictReader(stream) if int(row['repeat']) == repeat
                and row['source_path'] in labels]
    if len({row['source_path'] for row in rows}) != len(rows):
        raise ValueError('duplicate source path within OOF repeat')
    cases = []
    for row in rows:
        label = labels[row['source_path']]
        image = read_dxa(dataset / row['source_path'])
        sx = image.pixel_mm_x or image.pixel_mm
        base = evidence(image.pixels, label['region'], image.pixel_mm, sx)
        reference = label['hip_position_rotation'] == '1'
        predicted = VIOLATION_LABEL['hip_position_rotation'] in row['violation_type'].split(';')
        group = ('tp' if predicted else 'fn') if reference else ('fp' if predicted else 'tn')
        cases.append({'source_path': row['source_path'], 'group': group,
                      'variants': {name: compare(base, evidence(perturb(image.pixels, name),
                                                               label['region'], image.pixel_mm, sx),
                                                 image.pixel_mm, sx) for name in VARIANTS}})
    report = {'scope': 'technical invariance on inspected organiser development images',
              'transformations': {'dim': '0.9*x+3', 'bright': '1.1*x-3',
                                  'resize': '90% area resize then restore original size'},
              'review_distance_mm': 5,
              'repeat': repeat, 'images': len(cases),
              'oof_sha256': hashlib.sha256(oof_path.read_bytes()).hexdigest(),
              'labels_sha256': hashlib.sha256(labels_path.read_bytes()).hexdigest(),
              'summary': summarize(cases),
              'limitations': 'feature/point consistency only; no landmark ground truth or clinical validation'}
    return report, cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oof', 'labels', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='local per-image error file; do not commit medical paths')
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args()
    report, cases = audit(args.oof, args.labels, args.dataset, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if args.cases:
        args.cases.parent.mkdir(parents=True, exist_ok=True)
        args.cases.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'images': report['images'], 'groups': {
        group: report['summary']['bright'][group]['images'] for group in ('tp', 'fn', 'fp', 'tn')}}))
