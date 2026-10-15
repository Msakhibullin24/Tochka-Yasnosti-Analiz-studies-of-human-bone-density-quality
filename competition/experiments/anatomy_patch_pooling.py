"""Pool frozen DINO patch tokens by independently predicted anatomical masks.

Three fixed regions: bone, a ten-mm surrounding ring, and remaining image
context. Original global features are retained. Targets never enter pooling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_research.common import digest, fingerprint, load_feature_bundle
from source_integrity import inspect_sources
from experiments.mask_anatomy_features import largest_component


def project_mask(mask, transform, size, side):
    mask = np.asarray(mask, np.float32)
    if transform['mirrored']:
        mask = mask[:, ::-1]
    canvas = np.zeros((size, size), np.float32)
    x, y, width, height = [transform[k] for k in ('x', 'y', 'width', 'height')]
    canvas[y:y+height, x:x+width] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA).ravel()


def weighted_pool(tokens, weights):
    tokens, weights = np.asarray(tokens, np.float32), np.asarray(weights, np.float32)
    if tokens.ndim != 2 or weights.shape != (len(tokens),) or not np.isfinite(tokens).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Finite patch tokens and aligned nonnegative weights required')
    return (weights @ tokens / weights.sum()) if weights.sum() > 0 else np.zeros(tokens.shape[1], np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dense', 'dataset', 'spine_probe', 'hip_probe', 'mirror_probe', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new feature directory')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    source = inspect_sources([('organiser', args.labels, args.dataset)])
    source_id = fingerprint(source['entries'])
    globals_, meta = load_feature_bundle(args.dense / 'features.npy', labels, source_id, digest(args.labels))
    if not meta.get('dense_features'):
        raise ValueError('Dense patch cache required')
    probes = {k: json.loads((p / 'probe.json').read_text()) for k, p in
              [('spine', args.spine_probe), ('hip', args.hip_probe), ('mirror', args.mirror_probe)]}
    for p in probes.values():
        if p['labels_sha256'] != digest(args.labels):
            raise ValueError('Anatomical source table differs')
    records = {k: {r['pixel_sha256']: r for r in p['results'] if r['polarity'] == 'bone-bright'} for k, p in probes.items()}
    vectors = []
    for i, row in enumerate(labels.itertuples()):
        group = 'spine' if row.region == 'spine' else 'hip'
        directory = args.spine_probe if group == 'spine' else args.hip_probe
        ref = records[group][row.pixel_sha256]
        mask_path = directory / f'{row.pixel_sha256[:16]}-bone-bright.npy'
        transform = meta['transforms'][i]
        if transform['pixel_sha256'] != ref['decoded_pixel_sha256'] or digest(mask_path) != ref['mask_sha256']:
            raise ValueError('Dense cache and mask pixel identities differ')
        masks = np.load(mask_path)
        if group == 'spine':
            bone = np.stack([largest_component(m) for m in masks[3:8]]).any(axis=0)
        else:
            mirror = records['mirror'][row.pixel_sha256]
            other_path = args.mirror_probe / f'{row.pixel_sha256[:16]}-bone-bright.npy'
            if mirror['decoded_pixel_sha256'] != ref['decoded_pixel_sha256'] or digest(other_path) != mirror['mask_sha256']:
                raise ValueError('Mirrored anatomical identity differs')
            other = np.load(other_path)
            first = largest_component(masks[6:8].any(axis=0))
            second = largest_component(other[6:8].any(axis=0))
            bone = second if second.sum() > first.sum() else first
        sx, sy = transform['pixel_mm_x'], transform['pixel_mm_y']
        kx, ky = 2 * int(np.ceil(10 / sx)) + 1, 2 * int(np.ceil(10 / sy)) + 1
        dilated = cv2.dilate(bone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))).astype(bool)
        masks_to_pool = (bone, dilated & ~bone, ~dilated)
        path = args.dense / f'patches_{i:03d}.npy'
        if digest(path) != meta['dense_sha256'][path.name]:
            raise ValueError('Dense patch checksum differs')
        tokens = np.load(path).astype(np.float32)
        side = int(np.sqrt(len(tokens)))
        if side * side != len(tokens):
            raise ValueError('Expected a square patch grid')
        pools = [weighted_pool(tokens, project_mask(m, transform, meta['size'], side)) for m in masks_to_pool]
        vectors.append(np.r_[globals_[i], *pools].astype(np.float32))
    args.output.mkdir(parents=True)
    out = args.output / 'features.npy'
    values = np.stack(vectors)
    np.save(out, values)
    report = {**meta, 'pooling': 'global_plus_anatomical_bone_ring_context',
              'shape': list(values.shape), 'features_sha256': digest(out),
              'code_sha256': digest(Path(__file__)), 'mask_sources': {k: p['checkpoint_sha256'] for k, p in probes.items()},
              'clinical_validation': False, 'model_affects_decision': False}
    report.pop('dense_features', None)
    report.pop('dense_sha256', None)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'shape': list(values.shape), 'features_sha256': digest(out)}), flush=True)


if __name__ == '__main__':
    main()
