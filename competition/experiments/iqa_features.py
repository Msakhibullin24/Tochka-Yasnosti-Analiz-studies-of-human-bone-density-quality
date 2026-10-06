"""Frozen no-reference image-quality scores as auxiliary features.

pyiqa ships 100+ trained no-reference IQA metrics with public weights. None of
them was trained on projection radiographs and none of them measures positioning:
BRISQUE/NIQE/PIQE read MSCN/GMSD statistics, so they see noise, contrast,
sharpness and spatial complexity but are blind to a spine tilted 6 degrees or a
femur under-rotated.

That blindness is exactly why they may still help. They add an orthogonal
noise/sharpness/texture axis that the hand-crafted geometry block cannot
express, and they cannot overfit: nothing here is fitted to the 249 labels.

Per the RSNA data-limited-strategies guidance, no geometric augmentation is
applied and no flip/rotation TTA is used - laterality and rotation carry the
labels. The DXA frame is used as decoded, at native resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.dicom_io import read_any  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402

DEFAULT_METRICS = ('brisque', 'niqe', 'piqe', 'musiq', 'maniqa', 'topiq_nr',
                   'clipiqa', 'liqe', 'unique', 'uranker', 'hyperiqa', 'wadiqam_nr')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    import pyiqa
    import torch

    if args.output.exists():
        raise ValueError('Choose a new output directory')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}

    available = set(pyiqa.list_models())
    # Full-reference metrics need a pristine reference image, which a QC screen
    # does not have. Reject them up front instead of failing 249 images later.
    metrics = [m for m in args.metrics if m in available and not m.endswith('_fr')]
    missing = [m for m in args.metrics if m not in available or m.endswith('_fr')]
    if missing:
        print(f'skipping unavailable metrics: {missing}', flush=True)
    if not metrics:
        raise ValueError('None of the requested metrics exist in this pyiqa build')

    images = []
    for row in labels.itertuples():
        image = read_any(args.dataset / row.first_source_path)
        if image.pixel_sha256 != audited[row.first_source_path]:
            raise ValueError('Source pixels changed after audit')
        images.append(image.pixels)

    columns, values = {}, []
    for name in metrics:
        try:
            metric = pyiqa.create_metric(name, device=args.device)
        except Exception as error:  # a metric needing optional deps must not kill the run
            print(f'{name}: unavailable ({type(error).__name__}: {error})', flush=True)
            continue
        scores = []
        for i, pixels in enumerate(images):
            # Several pyiqa archs (MUSIQ, SWIN backbones) require 3 channels.
            gray = torch.from_numpy(np.ascontiguousarray(pixels)).float().div(255)
            tensor = gray.unsqueeze(0).repeat(args.channels, 1, 1).unsqueeze(0)
            if args.device == 'cuda':
                tensor = tensor.cuda()
            with torch.inference_mode():
                value = float(metric(tensor).flatten()[0].item())
            if not np.isfinite(value):
                raise ValueError(f'{name}: nonfinite score on image {i}')
            scores.append(value)
        columns[name] = np.asarray(scores, np.float32)
        values.append(scores)
        print(f'{name}: mean {np.mean(scores):.4f} std {np.std(scores):.4f}', flush=True)
        del metric
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    table = pd.DataFrame(columns)
    if table.shape[1] < 2:
        raise ValueError('Fewer than two metrics were computed')
    args.output.mkdir(parents=True)
    path = args.output / 'features.npy'
    np.save(path, table.to_numpy(np.float32), allow_pickle=False)
    (args.output / 'features.csv').write_text(table.to_csv(index=False))
    # Per-metric z-scores keep every axis on a comparable scale before PCA.
    scaled = (table - table.mean()) / table.std(ddof=0).replace(0, 1)
    np.save(args.output / 'features_z.npy', scaled.to_numpy(np.float32), allow_pickle=False)
    meta = {'schema_version': 1, 'model': 'pyiqa-frozen-iqa',
            'repo': 'chaofengc/IQA-PyTorch-Weights', 'revision': 'pyiqa-' + pyiqa.__version__,
            'weights_sha256': {}, 'labels_sha256': digest(args.labels),
            'source_fingerprint': fingerprint, 'paths': labels.first_source_path.astype(str).tolist(),
            'features_sha256': digest(path), 'metrics': metrics,
            'pooling': 'per-image scalar', 'encoder_finetuned': False, 'smoke_only': False,
            'rationale': __doc__.strip().splitlines()[0],
            'limitations': ['Trained on natural-image distortion corpora (KonIQ, SPAQ, LIVE, TID, KADID), not projection radiographs.',
                            'Cannot measure positioning: blind to rotation, laterality and coverage.',
                            'Uncalibrated across vendors; interpretable as relative scores only.'],
            'clinical_validation': False, 'model_affects_decision': False}
    (args.output / 'features.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(f'{table.shape} written to {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metrics', nargs='+', default=list(DEFAULT_METRICS))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--channels', type=int, default=3, help='Repeat the grayscale frame N times')
    run(parser.parse_args())
