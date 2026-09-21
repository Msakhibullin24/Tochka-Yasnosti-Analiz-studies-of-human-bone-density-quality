"""Exploratory hip positioning audit on one existing study-held-out OOF repeat.

No model is fitted or modified here. Univariate AUCs describe the inspected
development set and must not be used as a new independent performance claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dxaqc.anatomy import detect_landmarks
from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import measure_image
from dxaqc.model import VIOLATION_LABEL


FEATURES = ('shaft_abs_angle_deg', 'lesser_troch_protrusion_mm',
            'lesser_troch_notch_mm', 'medial_concavity_mm',
            'shaft_width_mm', 'troch_top_margin_mm', 'lateral_margin_mm')
LANDMARKS = ('greater_trochanter', 'lesser_trochanter', 'femoral_neck', 'ischium')


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError('no labelled hip predictions')
    y = np.array([r['reference'] for r in records], dtype=bool)
    p = np.array([r['predicted'] for r in records], dtype=bool)
    if len(np.unique(y)) != 2:
        raise ValueError('both reference classes are required')
    result = {
        'images': len(records), 'reference_positives': int(y.sum()),
        'criterion_confusion_tn_fp_fn_tp': [int((~y & ~p).sum()), int((~y & p).sum()),
                                           int((y & ~p).sum()), int((y & p).sum())],
        'landmark_candidate_counts': {
            name: {'positive': sum(r['reference'] and r['landmarks'][name] for r in records),
                   'negative': sum(not r['reference'] and r['landmarks'][name] for r in records)}
            for name in LANDMARKS},
        'features': {},
    }
    for name in FEATURES:
        values = np.array([r['features'].get(name, float('nan')) for r in records], dtype=float)
        valid = np.isfinite(values)
        if len(np.unique(y[valid])) != 2:
            result['features'][name] = {'valid': int(valid.sum()), 'raw_auc': None}
            continue
        result['features'][name] = {
            'valid': int(valid.sum()),
            'raw_auc': float(roc_auc_score(y[valid], values[valid])),
            'positive_median': float(np.median(values[valid & y])),
            'negative_median': float(np.median(values[valid & ~y])),
        }
    return result


def audit(oof_path: Path, labels_path: Path, dataset: Path, repeat: int) -> dict:
    with labels_path.open(newline='', encoding='utf-8-sig') as stream:
        labels = {r['first_source_path']: r for r in csv.DictReader(stream)
                  if r['region'].startswith('hip') and r['hip_position_rotation'] in ('0', '1')}
    with oof_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['repeat']) == repeat
                and r['source_path'] in labels]
    if len({r['source_path'] for r in rows}) != len(rows):
        raise ValueError('duplicate source path within OOF repeat')
    records = []
    for row in rows:
        label = labels[row['source_path']]
        image = read_dxa(dataset / row['source_path'])
        pixels = np.ascontiguousarray(image.pixels[:, ::-1]) if label['region'] == 'hip_left' else image.pixels
        measurement = measure_image(pixels, label['region'], image.pixel_mm, image.pixel_mm_x)
        anatomy = detect_landmarks(image.pixels, label['region'], measurement.overlay,
                                   image.pixel_mm, image.pixel_mm_x)
        candidates = {item['name']: item['status'] == 'candidate' for item in anatomy['landmarks']}
        records.append({
            'reference': label['hip_position_rotation'] == '1',
            'predicted': VIOLATION_LABEL['hip_position_rotation'] in row['violation_type'].split(';'),
            'features': measurement.features,
            'landmarks': candidates,
        })
    result = summarize(records)
    result.update({
        'scope': 'exploratory univariate analysis on inspected organiser development images',
        'repeat': repeat, 'oof_sha256': hashlib.sha256(oof_path.read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        'limitations': 'candidate availability is not landmark accuracy; no independent clinical reference',
    })
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oof', 'labels', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args()
    report = audit(args.oof, args.labels, args.dataset, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'images': report['images'],
                      'confusion': report['criterion_confusion_tn_fp_fn_tp']}))
