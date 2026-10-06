"""Patch-local pooling features from a dense DINOv3 forward pass.

arXiv 2606.11606 reports that CLS / global-pooled embeddings sit at the chance
floor for small localised perturbations (AUC 0.500-0.524) while patch-local
pooling on the same forward pass recovers AUC near 1.0. Positioning defects and
foreign objects are exactly that kind of small localised signal, so this script
re-derives the feature vector from the patch grid instead of the pooled token.

Blocks, all computed from the frozen float16 patch cache:
  1. multi-scale grid means at 1x1, 2x2, 4x4 and 8x8 - spatially localised
     descriptors, coarse scales act as position buckets;
  2. local-deviation maps - per-patch L2 distance from the image mean, pooled by
     mean/max over the same grids, i.e. "how unusual is the most unusual region";
  3. top-k patch norms by L2 distance from the image mean;
  4. the existing pooled/CLS vector, concatenated last as a control so the screen
     can attribute any change to the patch blocks rather than to the encoder.

Writes features.npy plus the companion JSON expected by
gpu_research.common.load_feature_bundle, so the nested screen can consume it
without modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GRIDS = (1, 2, 4, 8)
TOP_K = 16


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def grid_mean(grid, cells):
    """Mean over a cells x cells partition of a (H, W, D) grid."""
    h, w, d = grid.shape
    if h % cells or w % cells:
        raise ValueError(f'{cells}x{cells} does not divide the {h}x{w} patch grid')
    return grid.reshape(cells, h // cells, cells, w // cells, d).mean(axis=(1, 3)).reshape(cells * cells, d)


def build(tokens):
    """tokens: (P, D) float16 patch grid in raster order, P a perfect square."""
    n, d = tokens.shape
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f'{n} patches is not a square grid')
    grid = tokens.reshape(side, side, d).astype(np.float32)

    mean = grid.mean(axis=(0, 1))
    deviation = np.linalg.norm(grid - mean, axis=2)

    blocks, names = [], []
    for cells in GRIDS:
        blocks.append(grid_mean(grid, cells).reshape(-1))
        names.append(f'gridmean_{cells}x{cells}')
        if cells > 1:
            dmap = deviation.reshape(side, side)
            blocks.append(np.concatenate([grid_mean(dmap[..., None], cells).reshape(-1),
                                         deviation.reshape(cells, side // cells, cells, side // cells)
                                         .max(axis=(1, 3)).reshape(-1)]))
            names.append(f'localdev_mean_max_{cells}x{cells}')
    order = np.argsort(deviation.reshape(-1))[::-1][:min(TOP_K, n)]
    blocks.append(deviation.reshape(-1)[order])
    names.append(f'top{TOP_K}_patch_deviation')
    blocks.append(np.array([deviation.mean(), deviation.max(), deviation.std(),
                            deviation.max() - deviation.mean()], np.float32))
    names.append('deviation_summary')
    return np.concatenate(blocks).astype(np.float32), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense', type=Path, required=True, help='Directory with patches_*.npy and features.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.dense / 'features.json'
    meta = json.loads(source.read_text())
    if not meta.get('dense_features'):
        raise ValueError('Source directory was not produced with --dense')
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    names = None
    vectors = []
    for i in range(len(meta['paths'])):
        tokens = np.load(args.dense / f'patches_{i:03d}.npy', allow_pickle=False)
        vector, names = build(tokens)
        vectors.append(vector)
    values = np.stack(vectors)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite patch-local features')
    path = args.output / 'features.npy'
    np.save(path, values, allow_pickle=False)
    out = dict(meta)
    out.update({'pooling': 'patch-local-multi-scale', 'source_pooling': meta['pooling'],
                'patch_grid_side': int(round(meta['paths'].__len__() ** 0)) or None,
                'blocks': names, 'dimension': int(values.shape[1]),
                'features_sha256': digest(path), 'smoke_only': False,
                'rationale': 'arXiv 2606.11606: CLS/global pooling is at chance for small localised defects'})
    out.pop('dense_sha256', None)
    out.pop('dense_features', None)
    path.with_suffix('.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(f'{values.shape} written to {path}')
    print('blocks:', names)


if __name__ == '__main__':
    main()
