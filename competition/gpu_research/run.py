"""GPU research CLI: hardware check, isolated download and full-cohort frozen features."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_research.common import (CATALOG, HERE, checkpoint_identity, digest, dump, fingerprint,
                                 image_input, load_encoder, representations)


def doctor(output):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check NVIDIA driver and CUDA PyTorch wheels')
    properties = torch.cuda.get_device_properties(0)
    torch.manual_seed(17)
    layer = torch.nn.Linear(32, 8, device='cuda')
    optimizer = torch.optim.AdamW(layer.parameters())
    for precision in ('fp32', 'bf16'):
        if precision == 'bf16' and not torch.cuda.is_bf16_supported():
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            loss = layer(torch.randn(4, 32, device='cuda')).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in layer.parameters()):
            raise RuntimeError('GPU forward/backward produced nonfinite values')
        optimizer.step()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    report = {'gpu': properties.name, 'capability': list(torch.cuda.get_device_capability()),
              'free_vram_bytes': free, 'total_vram_bytes': total,
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'bf16_supported': torch.cuda.is_bf16_supported(), 'optimizer_smoke_passed': True,
              'packages': {p: importlib.metadata.version(p) for p in ('torchvision', 'transformers', 'numpy', 'scikit-learn')},
              'clinical_validation': False}
    dump(output, report)
    print(json.dumps(report, indent=2))


def download(name, root):
    from huggingface_hub import HfApi, snapshot_download
    entry = CATALOG[name]
    info = HfApi().model_info(entry['repo'], files_metadata=True)
    # Account must already have accepted any owner terms; credentials are never logged.
    weight_bytes = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith('.safetensors'))
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < weight_bytes * 1.2 + 2 * 1024**3:
        raise RuntimeError('Insufficient disk for chosen checkpoint and download overhead')
    directory = root / name
    snapshot_download(repo_id=entry['repo'], revision=info.sha, local_dir=directory,
                      allow_patterns=['*.json', '*.safetensors', '*.model', 'README.md', 'LICENSE*'], max_workers=2)
    identity = checkpoint_identity(directory, entry['kind'])
    for filename, checksum in identity['weights'].items():
        metadata = directory / '.cache/huggingface/download' / (filename + '.metadata')
        if metadata.exists():
            lines = metadata.read_text().splitlines()
            etag = lines[1].strip('"') if len(lines) > 1 else ''
            if len(etag) == 64 and etag != checksum:
                raise ValueError('Downloaded weight differs from publisher LFS SHA-256')
    dump(directory / 'research_manifest.json', {'repo': entry['repo'], 'revision': info.sha, **identity})
    print(f'Verified {name}: {directory}')


def extract(args):
    import numpy as np
    import pandas as pd
    import torch
    from dxaqc.dicom_io import read_any
    from source_integrity import inspect_sources
    entry = CATALOG[args.model]
    size = args.size or entry['size']
    patch = 14 if entry['kind'] == 'siglip' else 16
    if not 32 <= size <= 2048 or size % patch:
        raise ValueError(f'Input size must be 32..2048 and divisible by patch size {patch}')
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    source_fp = fingerprint(audit['entries'])
    audited = {row['path']: row['pixel_sha256_current'] for row in audit['entries']}
    identity = checkpoint_identity(args.model_dir, entry['kind'])
    processor, encoder = load_encoder(args.model_dir, entry['kind'], args.device, args.precision)
    encoder.requires_grad_(False)
    args.output.mkdir(parents=True)
    vectors, transforms, dense_sha = [], [], {}
    begin = time.perf_counter()
    indices = range(min(1, len(labels))) if args.smoke else range(len(labels))
    with torch.inference_mode():
        for i in indices:
            row = labels.iloc[i]
            image = read_any(args.dataset / row.first_source_path)
            if image.pixel_sha256 != audited[row.first_source_path]:
                raise ValueError('Source pixels changed after audit')
            mirrored = row.region == 'hip_left'
            pixels = np.ascontiguousarray(image.pixels[:, ::-1] if mirrored else image.pixels)
            values, transform = image_input(processor, pixels, size, entry['kind'])
            vector, patches = representations(encoder, values, entry['kind'], args.pooling)
            vectors.append(vector[0].float().cpu().numpy())
            transforms.append({'path': row.first_source_path, 'mirrored': mirrored,
                               'pixel_sha256': image.pixel_sha256,
                               'pixel_mm_x': image.pixel_mm_x, 'pixel_mm_y': image.pixel_mm, **transform})
            if args.dense:
                tokens = patches[0].float().cpu().numpy().astype(np.float16)
                if len(tokens) != (size // patch)**2 or not np.isfinite(tokens).all():
                    raise ValueError('Dense patch grid or float16 cache is invalid')
                patch_path = args.output / f'patches_{i:03d}.npy'
                np.save(patch_path, tokens, allow_pickle=False)
                dense_sha[patch_path.name] = digest(patch_path)
            if (i + 1) % 10 == 0 or i == 0:
                print(f'{args.model} encoded {i+1}/{len(labels)}', flush=True)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    path = args.output / 'features.npy'
    np.save(path, np.stack(vectors), allow_pickle=False)
    revision_file = args.model_dir / 'research_manifest.json'
    revision = json.loads(revision_file.read_text()).get('revision') if revision_file.exists() else 'unrecorded'
    meta = {'schema_version': 1, 'model': args.model, 'repo': entry['repo'], 'revision': revision,
            **identity, 'labels_sha256': digest(args.labels), 'source_fingerprint': source_fp,
            'paths': labels.first_source_path.iloc[list(indices)].tolist(), 'features_sha256': digest(path),
            'precision': args.precision, 'size': size, 'pooling': args.pooling,
            'preprocessing': 'pixel-aspect full-frame letterbox, bone-bright, left hip mirrored using known label',
            'transforms': transforms, 'encoder_finetuned': False,
            'dense_features': args.dense, 'dense_sha256': dense_sha, 'smoke_only': args.smoke,
            'elapsed_seconds': time.perf_counter() - begin,
            'peak_vram_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
            'code_sha256': {'run': digest(__file__), 'common': digest(Path(__file__).with_name('common.py'))},
            'clinical_validation': False, 'model_affects_decision': False}
    dump(path.with_suffix('.json'), meta)
    print(f'Features saved: {path}; smoke_only={args.smoke}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('doctor')
    check.add_argument('--output', type=Path, required=True)
    fetch = commands.add_parser('download')
    fetch.add_argument('--model', choices=CATALOG, required=True)
    fetch.add_argument('--root', type=Path, default=HERE.parent / 'data/gpu-models')
    features = commands.add_parser('extract')
    features.add_argument('--model', choices=CATALOG, required=True)
    features.add_argument('--model-dir', type=Path, required=True)
    features.add_argument('--dataset', type=Path, required=True)
    features.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    features.add_argument('--output', type=Path, required=True)
    features.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    features.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16')
    features.add_argument('--pooling', choices=('global', 'global-mean'), default='global-mean')
    features.add_argument('--size', type=int)
    features.add_argument('--dense', action='store_true')
    features.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'doctor':
        doctor(args.output)
    elif args.command == 'download':
        download(args.model, args.root)
    else:
        extract(args)


if __name__ == '__main__':
    main()
