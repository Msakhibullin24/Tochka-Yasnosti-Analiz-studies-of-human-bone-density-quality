"""Encode proximal hip crops with audited frozen image encoders.

Preserve audited global vectors for spine and unlocalised hips. No QC target
values select the crop. Full global encoders remain separate classifier inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from gpu_research.common import digest, fingerprint, load_encoder, image_input, representations, load_feature_bundle
from source_integrity import inspect_sources
from experiments.transfer_camera_projection import selected_masks
from experiments.anatomical_crop import hip_crop_bounds, POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'probe', 'mirror_probe', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    parser.add_argument('--global-features', type=Path, nargs=2, required=True)
    parser.add_argument('--encoders', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new crop feature directory')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(4)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    source_id = fingerprint(inspect_sources([('organiser', args.labels, args.dataset)])['entries'])
    routing = pd.read_csv(args.baseline, usecols=['index', 'source_path', 'predicted_region']).sort_values('index')
    if routing.source_path.tolist() != labels.first_source_path.tolist():
        raise ValueError('Routing/source image order differs')
    probes = [json.loads((p / 'probe.json').read_text()) for p in (args.probe, args.mirror_probe)]
    if (any(p['labels_sha256'] != digest(args.labels) for p in probes) or not probes[1]['input_horizontally_flipped']
            or probes[0]['checkpoint_sha256'] != probes[1]['checkpoint_sha256']):
        raise ValueError('Anatomical probe identities differ')
    refs = [{r['pixel_sha256']: r for r in p['results'] if r['polarity'] == 'bone-bright'} for p in probes]
    cropped = {}
    for i, row in enumerate(labels.itertuples()):
        region = routing.iloc[i].predicted_region
        if not region.startswith('hip'):
            continue
        path = args.dataset / row.first_source_path
        image = read_dxa(path)
        femur, _ = selected_masks(args.probe, args.mirror_probe, refs[0][row.pixel_sha256], refs[1][row.pixel_sha256], row, image, path)
        bounds = hip_crop_bounds(femur, image.pixel_mm_x, image.pixel_mm)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        pixels = image.pixels[y0:y1, x0:x1]
        if region == 'hip_left':
            pixels = np.ascontiguousarray(pixels[:, ::-1])
        cropped[i] = {'pixels': pixels, 'bounds': bounds, 'decoded_pixel_sha256': image.pixel_sha256}
    args.output.mkdir(parents=True)
    outputs = []
    for original_path, directory in zip(args.global_features, args.encoders, strict=True):
        global_vectors, metadata = load_feature_bundle(original_path, labels, source_id, digest(args.labels))
        for filename, expected in metadata['weights'].items():
            if digest(directory / filename) != expected:
                raise ValueError('Frozen encoder checkpoint differs')
        kind = json.loads((directory / 'config.json').read_text())['model_type']
        processor, encoder = load_encoder(directory, kind, 'cuda', metadata['precision'])
        result = np.asarray(global_vectors).astype(np.float32).copy()
        with torch.inference_mode():
            for i, case in cropped.items():
                inputs, _ = image_input(processor, case['pixels'], metadata['size'], kind)
                vector, _ = representations(encoder, inputs, kind, 'mean_global')
                result[i] = vector.float().cpu().numpy()[0]
        name = metadata['model']
        feature_path = args.output / (name + '.npy')
        np.save(feature_path, result)
        audit = copy.deepcopy(metadata)
        audit.update(features_sha256=digest(feature_path), derived_from_features_sha256=digest(original_path),
                     crop_policy=POLICY, source_fingerprint=source_id, labels_sha256=digest(args.labels),
                     crop_geometry_code_sha256=digest(Path(__file__).with_name('anatomical_crop.py')),
                     code_sha256=digest(Path(__file__)), anatomy_checkpoint_sha256=probes[0]['checkpoint_sha256'],
                     cropped_images=len(cropped), copied_global_images=len(labels) - len(cropped),
                     qc_target_values_used=False, clinical_validation=False)
        feature_path.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n')
        outputs.append({'path': str(feature_path), 'sha256': digest(feature_path), 'shape': list(result.shape)})
        del encoder
        torch.cuda.empty_cache()
        print(json.dumps(outputs[-1]), flush=True)
    (args.output / 'crop_audit.json').write_text(json.dumps({'policy': POLICY, 'outputs': outputs,
        'cropped_images': len(cropped), 'source_fingerprint': source_id, 'qc_target_values_used': False,
        'bounds_by_index': {str(i): {k: v for k, v in case.items() if k != 'pixels'} for i, case in cropped.items()},
        'clinical_validation': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
