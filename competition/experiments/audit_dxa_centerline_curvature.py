"""Evaluate a published cumulative centerline-curvature idea on AP spine DXA.

Research only. The reference paper uses full-body DXA and its cumulative
displacement is not the vertebral angle specified by the local requirements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import spine_centerline


PAPER = 'https://www.nature.com/articles/s41746-026-02540-6'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y: np.ndarray, scores: np.ndarray) -> dict:
    return {'roc_auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'threshold_5_deg_proxy': {
                'tp': int(((y == 1) & (scores > 5)).sum()),
                'fp': int(((y == 0) & (scores > 5)).sum()),
                'fn': int(((y == 1) & (scores <= 5)).sum()),
                'tn': int(((y == 0) & (scores <= 5)).sum())}}


def curvature_proxy(pixels: np.ndarray, pixel_mm_y: float, pixel_mm_x: float) -> float:
    """Sum lateral centerline changes after removing endpoint tilt, as an angle proxy."""
    width = max(1, round(pixels.shape[1] * pixel_mm_x / pixel_mm_y))
    physical = (cv2.resize(pixels, (width, pixels.shape[0]), interpolation=cv2.INTER_LINEAR)
                if width != pixels.shape[1] else pixels)
    ys, xs, _ = spine_centerline(physical)
    lo, hi = int(len(ys) * .08), int(len(ys) * .82)
    sample_y = np.linspace(lo, hi - 1, 20).round().astype(int)
    sample_x = np.array([np.median(xs[max(lo, y - 4):min(hi, y + 5)]) for y in sample_y])
    deviation = sample_x - np.linspace(sample_x[0], sample_x[-1], len(sample_x))
    travel_mm = float(np.abs(np.diff(deviation)).sum() * pixel_mm_y)
    height_mm = float((sample_y[-1] - sample_y[0]) * pixel_mm_y)
    return math.degrees(math.atan2(travel_mm, height_mm))


def audit(labels: Path, dataset: Path, baseline: Path) -> dict:
    with labels.open(newline='', encoding='utf-8-sig') as stream:
        references = [row for row in csv.DictReader(stream)
                      if row['region'] == 'spine' and row['spine_axis'] in ('0', '1')]
    baseline_report = json.loads(baseline.read_text())
    indexed = {row['source_path']: row for row in baseline_report['cases']}
    if len(references) != 99 or len(indexed) != 99:
        raise ValueError('Unexpected organizer spine cohort')
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Known malformed organizer DICOM UIDs.
        for reference in references:
            relative = reference['first_source_path']
            source = (dataset/relative).resolve()
            if not source.is_relative_to(dataset.resolve()):
                raise ValueError('Source escapes dataset root')
            image = read_dxa(source)
            old = indexed[relative]
            if (image.pixel_sha256 != old['pixel_sha256']
                    or image.pixels.shape != (int(reference['rows']), int(reference['columns']))
                    or int(reference['spine_axis']) != old['axis_violation']):
                raise ValueError('Pixels or labels differ from pinned baseline')
            proxy = curvature_proxy(image.pixels, image.pixel_mm,
                                    image.pixel_mm_x or image.pixel_mm)
            cases.append({'source_sha256': sha256(source),
                          'study_sha256': hashlib.sha256(reference['study_key'].encode()).hexdigest(),
                          'target': int(reference['spine_axis']),
                          'published_method_proxy_deg': round(proxy, 6),
                          'current_global_angle_deg': abs(float(old['original_angle_deg'])),
                          'current_segment_max_deg': abs(float(old['segment_max_angle_deg']))})
    y = np.array([row['target'] for row in cases])
    if int(y.sum()) != 10 or len({r['study_sha256'] for r in cases}) != 99:
        raise ValueError('Unexpected positives or duplicate study')
    scores = np.array([row['published_method_proxy_deg'] for row in cases])
    global_scores = np.array([row['current_global_angle_deg'] for row in cases])
    segment_scores = np.array([row['current_segment_max_deg'] for row in cases])
    return {'protocol': '99 organizer AP spine DXA, same pixels and binary labels; fixed published-style 20-point cumulative lateral displacement',
            'paper': PAPER,
            'paper_modality': 'full-body DXA; target modality is AP lumbar DXA',
            'images': len(cases), 'positive_labels': int(y.sum()),
            'candidate': metric(y, scores),
            'current_global_angle': metric(y, global_scores),
            'current_segment_max': metric(y, segment_scores),
            'labels_sha256': sha256(labels), 'baseline_report_sha256': sha256(baseline),
            'code_sha256': sha256(Path(__file__)),
            'cases': cases, 'clinical_validation': False, 'release_modified': False,
            'requirements_complete': False,
            'limitations': [
                'The paper uses full-body DXA, not the local AP lumbar acquisition.',
                'The cumulative-displacement angle proxy is not a measured vertebral axis angle.',
                'Binary organizer axis labels contain no continuous landmark-angle ground truth.',
                'Organizer cases were used repeatedly during development; this is not independent validation.',
            ]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new report path')
    result = audit(args.labels, args.dataset, args.baseline)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('images', 'positive_labels', 'candidate',
                     'current_global_angle', 'current_segment_max')}, ensure_ascii=False, indent=2))
