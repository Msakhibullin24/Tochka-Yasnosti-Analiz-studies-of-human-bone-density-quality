"""Multi-layer and multi-scale frozen features.

Two gaps in the original extraction, both suspected of costing signal:

1. Only the final transformer block was used. In ViTs the last block is the most
   task-specific and the least spatially precise; middle blocks carry finer
   localisation. Concatenating a spread of depths gives the head access to both.
2. A single input size was used. Multi-scale averaging is variance reduction and
   costs nothing - unlike flip/rotation TTA, which destroys laterality and the
   spine-axis signal and is therefore forbidden here.

Writes the same features.npy / features.json contract as run.py so the nested
screen consumes it unchanged. Pooling per selected layer is CLS concatenated with
the mean patch token, matching the existing global-mean convention.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import (CATALOG, checkpoint_identity, declared_patch_size, digest, dump,
                                 fingerprint, image_input, load_encoder)
from dxaqc.dicom_io import read_any
from source_integrity import inspect_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=CATALOG)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', required=True,
                        help='Hidden-state indices to pool; -1 means the final block')
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16')
    parser.add_argument('--pooling', choices=('global', 'global-mean'), default='global-mean')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    entry = CATALOG[args.model]
    patch = declared_patch_size(args.model_dir, entry['kind'])
    for size in args.sizes:
        if not 32 <= size <= 2048 or size % patch:
            raise ValueError(f'Input size must be divisible by patch size {patch} and within 32..2048')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    source_fingerprint = fingerprint(audit['entries'])
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}
    identity = checkpoint_identity(args.model_dir, entry['kind'])
    processor, encoder = load_encoder(args.model_dir, entry['kind'], args.device, args.precision)
    encoder.requires_grad_(False)
    config = json.loads((args.model_dir / 'config.json').read_text())
    depth = int(config.get('num_hidden_layers') or len(getattr(encoder, 'model', encoder).layer))
    registers = int(getattr(encoder.config, 'num_register_tokens', 0))
    # output_hidden_states holds depth + 1 entries indexed 0..depth, so the final
    # block is index `depth`, not depth + 1.
    layers = [depth if n == -1 else n for n in args.layers]
    for n in layers:
        if not 0 < n <= depth:
            raise ValueError(f'Layer {n} is outside 1..{depth}')

    args.output.mkdir(parents=True)
    per_image = []
    transforms = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(labels.itertuples()):
            image = read_any(args.dataset / row.first_source_path)
            if image.pixel_sha256 != audited[row.first_source_path]:
                raise ValueError('Source pixels changed after audit')
            pixels = np.ascontiguousarray(image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels)
            scale_vectors = []
            for size in args.sizes:
                values, transform = image_input(processor, pixels, size, entry['kind'])
                out = encoder(pixel_values=values.to(device=args.device, dtype=next(encoder.parameters()).dtype),
                              output_hidden_states=True)
                states = out.hidden_states
                if states is None or max(layers) >= len(states):
                    raise ValueError(f'Model returned {0 if states is None else len(states)} hidden states; '
                                     f'cannot read layer {max(layers)}')
                picked = []
                for n in layers:
                    hidden = states[n]
                    if entry['kind'] == 'siglip':
                        vector = hidden[:, 0] if args.pooling == 'global' else torch.cat(
                            (hidden[:, 0], hidden[:, 1:].mean(1)), dim=1)
                    else:
                        patches = hidden[:, 1 + registers:]
                        vector = hidden[:, 0] if args.pooling == 'global' else torch.cat(
                            (hidden[:, 0], patches.mean(1)), dim=1)
                    if not torch.isfinite(vector).all():
                        raise ValueError('Nonfinite vision output')
                    picked.append(vector.float())
                scale_vectors.append(torch.cat(picked, dim=1))
                transforms.append({'path': row.first_source_path, 'size': size, **transform})
            per_image.append(torch.stack(scale_vectors).mean(0)[0].cpu().numpy())
            if (i + 1) % 25 == 0 or i == 0:
                print(f'{args.model} {i + 1}/{len(labels)}', flush=True)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    values = np.stack(per_image).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite feature matrix')
    path = args.output / 'features.npy'
    np.save(path, values, allow_pickle=False)
    dump(path.with_suffix('.json'), {
        'schema_version': 1, 'model': args.model, 'repo': entry['repo'],
        'revision': json.loads((args.model_dir / 'research_manifest.json').read_text()).get('revision', 'unrecorded')
        if (args.model_dir / 'research_manifest.json').exists() else 'unrecorded',
        **identity, 'labels_sha256': digest(args.labels), 'source_fingerprint': source_fingerprint,
        'paths': labels.first_source_path.astype(str).tolist(), 'features_sha256': digest(path),
        'precision': args.precision, 'sizes': args.sizes, 'layers': layers, 'layer_depth': depth,
        'pooling': 'mean over scales of concat(CLS, patch-mean) per selected layer',
        'preprocessing': 'pixel-aspect full-frame letterbox, bone-bright, left hip mirrored using known label',
        'transforms': transforms, 'encoder_finetuned': False, 'dense_features': False,
        'smoke_only': False, 'dimension': int(values.shape[1]),
        'elapsed_seconds': time.perf_counter() - start,
        'peak_vram_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
        'rationale': __doc__.strip().splitlines()[0],
        'clinical_validation': False, 'model_affects_decision': False})


if __name__ == '__main__':
    main()
