"""Verify a display-only landmark refinement leaves frozen QC decisions identical.

Uses target-free input paths and fresh raw inference twice per image. This is
an implementation check, not an estimate of model accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.live_anatomy_deployment import load_candidate
from dxaqc.dicom_io import read_dxa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    candidate = load_candidate(args.config)
    model = candidate.hip_display_model
    if model is None:
        raise ValueError('An explicitly configured display refinement is required')
    checks = []
    with args.manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        image = read_dxa(args.dataset / row['first_source_path'])
        candidate.hip_display_model = None
        try:
            primary = candidate.analyze(image)
        finally:
            candidate.hip_display_model = model
        refined = candidate.analyze(image)
        for field in ('quality', 'score', 'violations', 'criterion_states', 'criterion_thresholds', 'criteria'):
            if primary[field] != refined[field]:
                raise AssertionError('Display refinement changed frozen QC field: ' + field)
        before = primary['learned_anatomy']['neck_roi_assessment']
        after = refined['learned_anatomy']['neck_roi_assessment']
        checks.append({'decoded_pixel_sha256': image.pixel_sha256, 'qc_identical': True,
                       'primary_roi_status': before['status'], 'primary_roi_reason': before.get('reason'),
                       'refined_roi_status': after['status'], 'refined_roi_reason': after.get('reason'),
                       'qc_input_landmark_model_sha256': candidate.hip_model_hashes['landmarks'],
                       'display_landmark_model_sha256': candidate.hip_model_hashes['display_landmarks']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'images': len(checks), 'all_qc_identical': all(c['qc_identical'] for c in checks),
        'checks': checks, 'accuracy_evaluation': False,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    print(json.dumps({'images': len(checks), 'all_qc_identical': True}), flush=True)


if __name__ == '__main__':
    main()
