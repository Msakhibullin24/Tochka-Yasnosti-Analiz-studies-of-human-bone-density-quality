"""Evaluate frozen numbered centers on BUU2024, excluding the entire BUU400 cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_buu_transfer import references
from evaluate_organizer_dataset import sha256
from predict_spatial_anatomy import load, predict
from train_buu_anatomy import numbered_metrics


def exclusion_reason(path, pixels, known_names, known_pixels):
    # Both AP and lateral records from BUU400 are excluded, including its test patients.
    if path.stem[:-1] in known_names:
        return 'existing_patient_filename'
    if hashlib.sha256(pixels.tobytes()).hexdigest() in known_pixels:
        return 'existing_pixels'
    return None


def run(root, model_path, output):
    if output.exists():
        raise ValueError('Choose a new report path')
    torch.set_num_threads(4)
    original = json.loads(Path('competition/reports/buu_frozen_transfer_2026_09_26.json').read_text())['cases']
    names = {Path(r['file']).stem[:-1] for r in original}
    pixels_seen = {r['pixel_sha256'] for r in original}
    model = load(model_path)
    cases, excluded = [], []
    for i, path in enumerate(sorted(root.rglob('AP/*.jpg'))):
        pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError(f'Unreadable image: {path}')
        reason = exclusion_reason(path, pixels, names, pixels_seen)
        if reason:
            excluded.append({'file': str(path.relative_to(root)), 'reason': reason})
            continue
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest in pixels_seen:
            raise ValueError('Duplicate evaluation pixels')
        pixels_seen.add(digest)
        centers, heights = references(path.with_suffix('.csv'), pixels.shape[1], pixels.shape[0])
        predicted = predict(model, pixels)
        cases.append({'file': str(path.relative_to(root)), 'patient_id': path.stem[:-1],
                      'source_sha256': sha256(path), 'pixel_sha256': digest,
                      'annotation_sha256': sha256(path.with_suffix('.csv')),
                      'reference': centers.tolist(), 'prediction': predicted.tolist(),
                      'body_heights': heights.tolist()})
        if len(cases) % 100 == 0:
            print(f'Evaluated {len(cases)} new AP images; excluded {len(excluded)}', flush=True)
    if not cases:
        raise ValueError('No independent images after exclusions')
    report = {'protocol': 'Frozen BUU400 model; entire original cohort excluded by source patient filename and pixel hash; no fit or threshold selection.',
              'clinical_validation': False, 'release_modified': False,
              'model_sha256': sha256(model_path), 'code_sha256': sha256(Path(__file__)),
              'metrics': numbered_metrics(np.array([r['prediction'] for r in cases]),
                                          np.array([r['reference'] for r in cases]),
                                          np.array([r['body_heights'] for r in cases])),
              'cases': cases, 'excluded': excluded,
              'limitations': ['Same institution, not unseen-site evaluation.',
                              'Filename-based patient identity; resized or renamed near-duplicates require further audit.',
                              'Plain radiographs, not DXA; no masks or clinical QC labels.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'model', 'output'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.model, a.output)
