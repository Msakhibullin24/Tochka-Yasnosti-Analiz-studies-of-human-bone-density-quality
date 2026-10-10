"""Frozen encoder features for the Chinese osteoporosis DXA corpus.

Deliberately separate from the probe so the expensive part runs once and the cheap part can be
revisited. Every encoder listed in the catalogue is loaded through the shared loader, which
refuses to let a processor crop or resize, so the whole DXA frame survives into the encoder.

The corpus has no QC labels, so this script produces features only. What they are for is
decided by the probe, which is the honest order: measure whether these representations carry
DXA-relevant signal at all before spending compute on self-supervised training over them.
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
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import CATALOG, digest, dump, image_input, load_encoder, representations  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def load_gray(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert('L'))
    if a.ndim != 2 or a.shape[0] < 32 or a.shape[1] < 32:
        raise ValueError(f'implausible raster {path} {a.shape}')
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='extraction root')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--folders', nargs='+', default=['LumbarP', 'LumbarL', 'Hip'])
    parser.add_argument('--per-patient', type=int, default=2, help='cap images per patient per folder')
    parser.add_argument('--pooling', default='mean_global', choices=['global', 'mean_global'])
    parser.add_argument('--precision', default='fp16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)

    spec = CATALOG[args.model]
    inv = pd.read_csv(args.inventory)
    inv = inv[inv.parsed & inv.folder.isin(args.folders)].copy()
    # One patient's AP and lateral views must not be split across folds later, and a patient with
    # many repeat scans would otherwise dominate. Cap per patient per folder.
    inv = inv.sort_values('path').groupby(['patient', 'folder'], as_index=False).head(args.per_patient)
    inv = inv.sort_values(['folder', 'patient', 'path']).reset_index(drop=True)
    if args.max_images:
        inv = inv.iloc[:args.max_images].reset_index(drop=True)
    print(f'{len(inv)} images, {inv.patient.nunique()} patients, folders {sorted(inv.folder.unique())}')

    processor, encoder = load_encoder(spec['local_dir'], spec['kind'], args.device, args.precision)
    feats, kept = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i, row in inv.iterrows():
            p = args.root / row['path']
            try:
                pixels = load_gray(p)
                values, _ = image_input(processor, pixels, spec['size'], spec['kind'])
                vector, _ = representations(encoder, values, spec['kind'], args.pooling)
                feats.append(vector[0].float().cpu().numpy())
                kept.append(row['path'])
            except Exception as exc:  # noqa: BLE001
                print(f'skip {row["path"]}: {type(exc).__name__}: {exc}')
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f'{i + 1}/{len(inv)} {el:.0f}s {el / (i + 1) * 1000:.0f}ms/img', flush=True)

    X = np.stack(feats).astype(np.float32)
    meta = inv[inv.path.isin(set(kept))].reset_index(drop=True)
    assert len(meta) == X.shape[0], (len(meta), X.shape)
    out = args.output / f'features_{args.model}.npy'
    np.save(out, X)
    meta.to_csv(args.output / f'meta_{args.model}.csv', index=False)
    dump(out.with_suffix('.json'), {
        'protocol': __doc__, 'model': args.model, 'repo': spec['repo'], 'pooling': args.pooling,
        'precision': args.precision, 'size': spec['size'], 'shape': list(X.shape),
        'images': int(X.shape[0]), 'patients': int(meta.patient.nunique()),
        'folders': {k: int(v) for k, v in meta.folder.value_counts().items()},
        'weights_sha256': spec.get('weights_sha256'), 'has_qc_labels': False,
        'note': 'Unlabelled corpus. No QC ground truth exists here; see inventory_summary.json.'})
    print(f'wrote {out} {X.shape} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
