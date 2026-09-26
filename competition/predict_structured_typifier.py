"""Produce inspectable secondary hypotheses for unresolved released predictions.

No CSV verdict, source label, or physician answer is overwritten. A learned axis
label conflicting with the measured 5-degree rule is reported as a conflict.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import joblib

from dxaqc.decision import AXIS_LIMIT_DEG
from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import embed
from dxaqc.model import CRITERIA, group_of, official_violation_type
from dxaqc.pipeline import Analyzer
from evaluate_organizer_dataset import sha256
from structured_typifier import StructuredTypifier


def describe(result, prediction):
    codes = prediction['codes']
    if not set(codes) <= CRITERIA[group_of(result['region'])].keys():
        raise ValueError('Candidate contains a wrong-region or nonofficial criterion')
    angle = result['features'].get('spine_abs_angle_deg')
    conflicts = []
    if 'spine_axis' in codes:
        if not isinstance(angle, (int, float)) or not math.isfinite(angle):
            conflicts.append('axis_measurement_unavailable')
        elif angle <= AXIS_LIMIT_DEG:
            conflicts.append('learned_axis_type_with_measured_angle_at_most_5deg')
    if not codes and result['quality'] == 1:
        conflicts.append('secondary_normal_vs_original_binary_alarm')
    return {'candidate_quality_class': int(bool(codes)),
            'candidate_violation_type': official_violation_type(codes),
            'prediction': prediction, 'measured_axis_deg': angle,
            'axis_limit_deg': AXIS_LIMIT_DEG, 'conflicts': conflicts,
            'original_quality_class': result['quality'],
            'original_violation_type': official_violation_type(result['violations']),
            'original_verdict_changed': False, 'clinical_validation': False}


def run(dataset, results, models, output):
    if output.exists():
        raise ValueError('Choose a new candidate output file')
    analyzer = Analyzer()
    loaded, hashes = {}, {}
    for group in ('spine', 'hip'):
        path = models/f'logistic_{group}.joblib'
        model = joblib.load(path)
        if not isinstance(model, StructuredTypifier) or model.group != group or model.family != 'logistic':
            raise ValueError('Incompatible secondary model')
        loaded[group], hashes[group] = model, sha256(path)
    with results.open(encoding='utf-8-sig', newline='') as stream:
        selected = [row for row in csv.DictReader(stream)
                    if row.get('processing_status') == 'Success' and row.get('quality_class') == '1'
                    and not row.get('violation_type')]
    records, cache = [], {}
    for row in selected:
        path = (dataset/row['path_to_file']).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError('Result path escapes dataset')
        image = read_dxa(path)
        if image.pixel_sha256 not in cache:
            result = analyzer.analyze(image)
            # Reject stale baseline reports before proposing a correction.
            if result['quality'] != 1 or result['violations']:
                raise ValueError('Current inference differs from the unresolved baseline report')
            group = group_of(result['region'])
            pixels = image.pixels[:, ::-1].copy() if result['region']=='hip_left' else image.pixels
            prediction = loaded[group].predict([result['features']], embed(pixels)[None])[0]
            cache[image.pixel_sha256] = describe(result, prediction)
        records.append({'path_to_file': row['path_to_file'], 'pixel_sha256': image.pixel_sha256,
                        **cache[image.pixel_sha256]})
    report = {'scope': 'secondary hypotheses only; released verdicts unchanged',
              'results_sha256': sha256(results), 'model_sha256': hashes,
              'images': len(records), 'unique_pixels': len(cache),
              'candidate_typed': sum(bool(r['candidate_violation_type']) for r in records),
              'candidate_normal': sum(not r['candidate_violation_type'] for r in records),
              'conflicts': sum(bool(r['conflicts']) for r in records), 'cases': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'results', 'models', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*', category=UserWarning)
        report = run(args.dataset, args.results, args.models, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
