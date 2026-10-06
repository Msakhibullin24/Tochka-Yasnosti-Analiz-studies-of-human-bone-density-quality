"""Identity, full-frame preprocessing and locally loaded vision encoders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
CATALOG = json.loads((Path(__file__).parent / 'models.json').read_text())['models']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(entries):
    return hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in entries],
        ensure_ascii=False).encode()).hexdigest()


def letterbox(pixels, size):
    """Preserve the complete pixel frame and aspect; physical geometry stays in source coordinates."""
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or not 32 <= size <= 2048:
        raise ValueError('Expected uint8 grayscale and size 32..2048')
    h, w = pixels.shape
    if min(h, w) < 1:
        raise ValueError('Empty image')
    scale = min(size / w, size / h)
    nw, nh = min(size, max(1, round(w * scale))), min(size, max(1, round(h * scale)))
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas = np.zeros((size, size), np.uint8)
    canvas[y:y+nh, x:x+nw] = cv2.resize(pixels, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.repeat(canvas[:, :, None], 3, axis=2), {'x': x, 'y': y, 'width': nw, 'height': nh,
                                                     'source_width': w, 'source_height': h}


def checkpoint_identity(directory, kind):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    if config.get('model_type') != kind:
        raise ValueError('Model architecture differs from selected catalog entry')
    weights = sorted(directory.glob('*.safetensors'))
    if not weights:
        raise ValueError('No local safetensors weights')
    index = directory / 'model.safetensors.index.json'
    if index.exists() and set(json.loads(index.read_text())['weight_map'].values()) != {p.name for p in weights}:
        raise ValueError('Incomplete sharded model')
    files = {p.name: digest(p) for p in weights}
    return {'weights': files, 'config_sha256': digest(directory / 'config.json'),
            'processor_sha256': digest(directory / 'preprocessor_config.json'),
            'weights_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def declared_patch_size(model_dir, kind):
    """Read patch_size from the checkpoint config; fall back to the family default.

    DINOv2 and SigLIP use 14, DINOv3 uses 16, so this must not be hard-coded.
    Tolerates a missing config so synthetic test encoders still run.
    """
    config = Path(model_dir) / 'config.json'
    if config.exists():
        declared = json.loads(config.read_text()).get('patch_size')
        if declared:
            return int(declared)
    return 14 if kind in ('siglip', 'dinov2') else 16


def load_encoder(directory, kind, device, precision):
    from transformers import AutoImageProcessor, AutoModel
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent CPU fallback')
    if precision == 'bf16' and device == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unsupported; explicitly select fp32/fp16 for extraction')
    if device == 'cpu' and precision != 'fp32':
        raise ValueError('CPU validation uses fp32')
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
    model = AutoModel.from_pretrained(directory, local_files_only=True, dtype=dtype)
    # Text tower is unused. Keep only the vision tower for MedSigLIP.
    encoder = model.vision_model if kind == 'siglip' else model
    encoder.to(device).eval()
    return processor, encoder


def image_input(processor, pixels, size, kind):
    rgb, transform = letterbox(pixels, size)
    # Never let a processor crop or resample: the full DXA frame must survive.
    kwargs = {'do_resize': False, 'do_center_crop': False}
    values = processor(images=rgb, return_tensors='pt', **kwargs)['pixel_values']
    if values.shape != (1, 3, size, size):
        raise ValueError('Processor cropped or resized the complete DXA frame')
    return values, transform


def representations(encoder, values, kind, pooling):
    parameter = next(encoder.parameters())
    kwargs = {'pixel_values': values.to(device=parameter.device, dtype=parameter.dtype)}
    if kind == 'siglip':
        kwargs['interpolate_pos_encoding'] = True
    output = encoder(**kwargs)
    hidden = output.last_hidden_state
    if kind == 'siglip':
        patches = hidden
        pooled = output.pooler_output
    else:
        registers = int(getattr(encoder.config, 'num_register_tokens', 0))
        patches = hidden[:, 1 + registers:]
        pooled = hidden[:, 0]
    vector = pooled if pooling == 'global' else torch.cat((pooled, patches.mean(1)), dim=1)
    if not torch.isfinite(vector).all() or not torch.isfinite(patches).all():
        raise ValueError('Nonfinite vision output')
    return vector, patches


def load_feature_bundle(path, labels, source_fingerprint, labels_sha):
    """Reject reordered, stale, partial or tampered features before training."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    values = np.load(path, allow_pickle=False)
    expected = labels.first_source_path.astype(str).tolist()
    if (meta.get('schema_version') != 1 or meta.get('paths') != expected
            or meta.get('labels_sha256') != labels_sha
            or meta.get('source_fingerprint') != source_fingerprint
            or meta.get('features_sha256') != digest(path)):
        raise ValueError('Feature identities, labels, source pixels or checksum differ')
    if values.ndim != 2 or values.shape[0] != len(labels) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Incomplete or nonfinite feature matrix')
    if meta.get('encoder_finetuned') is not False:
        raise ValueError('Shared cross-fold features must use a frozen external encoder')
    return values, meta
