"""Transfer synthetic camera recognition to checked DXA mask inputs.

No quality targets are used. Scores estimate synthetic camera endpoints;
no clinical projection ground truth or in-domain accuracy is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from experiments.mask_anatomy_features import largest_component
from experiments.pelvis_camera_projection import normalized_channels
from experiments.train_camera_projection import ProjectionNet
from gpu_research.common import fingerprint
from source_integrity import inspect_sources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_masks(probe, mirror_probe, reference, mirrored, row, image, source_path):
    stacks = []
    for directory, ref in ((probe, reference), (mirror_probe, mirrored)):
        mask_path = directory / f'{row.pixel_sha256[:16]}-bone-bright.npy'
        if (image.pixel_sha256 != ref['decoded_pixel_sha256'] or
                digest(source_path) != ref['source_file_sha256'] or digest(mask_path) != ref['mask_sha256']):
            raise ValueError('Current source pixel, file or mask identity differs')
        masks = np.load(mask_path)
        if masks.shape != (23, *image.pixels.shape):
            raise ValueError('Expected 23 original-coordinate musculoskeletal masks')
        stacks.append(masks)
    selected = []
    for a, b in ((6, 7), (8, 9)):
        candidates = [largest_component(m[a] | m[b]) for m in stacks]
        selected.append(max(candidates, key=lambda m: int(m.sum())))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'probe', 'mirror_probe', 'model', 'dataset', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new transfer directory')
    report = json.loads(args.features.read_text())
    label_hash = digest(args.labels)
    if (report['labels_sha256'] != label_hash or report['source_fingerprint'] !=
            fingerprint(inspect_sources([('organiser', args.labels, args.dataset)])['entries'])):
        raise ValueError('Feature source identities differ')
    probes = [json.loads((p / 'probe.json').read_text()) for p in (args.probe, args.mirror_probe)]
    if (any(p['labels_sha256'] != label_hash or p['checkpoint_sha256'] != report['hip_checkpoint_sha256'] for p in probes)
            or not probes[1]['input_horizontally_flipped']):
        raise ValueError('Anatomy probe identity differs')
    refs = [{r['pixel_sha256']: r for r in p['results'] if r['polarity'] == 'bone-bright'} for p in probes]
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    if checkpoint.get('channels') != ['femur', 'ipsilateral_pelvis']:
        raise ValueError('Unexpected camera model inputs')
    torch.set_num_threads(4)
    model = ProjectionNet().cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    rows = pd.read_csv(args.labels, usecols=['first_source_path', 'pixel_sha256', 'region'])
    results = []
    for row in rows.itertuples():
        if not row.region.startswith('hip') or row.pixel_sha256 not in report['entries']:
            continue
        features = report['entries'][row.pixel_sha256]['features']
        # A segmentation failure is neither a projection class nor a QC defect.
        if features.get('anat_landmark_input_valid') != 1:
            features['anat_projection_input_valid'] = 0.
            results.append({'historical_pixel_sha256': row.pixel_sha256, 'status': 'unavailable'})
            continue
        path = args.dataset / row.first_source_path
        image = read_dxa(path)
        femur, pelvis = selected_masks(args.probe, args.mirror_probe, refs[0][row.pixel_sha256],
                                       refs[1][row.pixel_sha256], row, image, path)
        channels = normalized_channels(femur.astype(np.uint8), pelvis.astype(np.uint8), image.pixel_mm_x, image.pixel_mm)
        if row.region == 'hip_right':
            channels = channels[..., ::-1].copy()
        with torch.inference_mode():
            scores = model(torch.tensor(channels[None, :2].copy(), dtype=torch.float32, device='cuda')).softmax(1)[0]
        probability = float(scores[1])
        features.update({'anat_projection_input_valid': 1., 'anat_projection_lateral_probability': probability,
                         'anat_projection_confidence': max(probability, 1 - probability)})
        results.append({'historical_pixel_sha256': row.pixel_sha256, 'decoded_pixel_sha256': image.pixel_sha256,
                        'status': 'estimated', 'lateral_endpoint_probability': probability,
                        'synthetic_endpoint_prediction': 'lateral' if probability >= .5 else 'frontal'})
    report['columns_by_group']['hip'] = sorted({k for r in rows.itertuples() if r.region.startswith('hip') and r.pixel_sha256 in report['entries']
                                               for k in report['entries'][r.pixel_sha256]['features']})
    report['projection_model_sha256'] = digest(args.model)
    report['projection_code_sha256'] = digest(Path(__file__))
    args.output.mkdir(parents=True)
    (args.output / 'features.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    summary = {'protocol': __doc__, 'images': len(results),
               'available': sum(r['status'] == 'estimated' for r in results),
               'predicted_lateral_endpoints': sum(r.get('synthetic_endpoint_prediction') == 'lateral' for r in results),
               'model_sha256': digest(args.model), 'input_features_sha256': digest(args.features),
               'code_sha256': digest(Path(__file__)), 'clinical_validation': False,
               'projection_accuracy_established': False, 'automatic_promotions': 0, 'results': results}
    (args.output / 'evaluation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
