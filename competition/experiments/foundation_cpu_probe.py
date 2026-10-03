"""Explicitly separate INT8 CPU research embeddings from the FP32 foundation probe.

The classifier still uses study-held-out folds. Quantization applies only to
frozen vision Linear layers; no clinical or release decision is changed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import warnings

import torch
from transformers import AutoImageProcessor, AutoModel

import dxa_hip_foundation_probe as probe


def main():
    args = sys.argv[1:]
    image_size = 448
    if '--image-size' in args:
        position = args.index('--image-size')
        image_size = int(args[position + 1])
        if image_size not in (224, 448):
            raise ValueError('Supported research sizes: 224, 448')
        del sys.argv[position + 1:position + 3]
        args = sys.argv[1:]
    if '--device' in args and args[args.index('--device') + 1] != 'cpu':
        raise ValueError('This research wrapper supports CPU only')
    if '--cache-dir' not in args:
        raise ValueError('Supply a separate --cache-dir for INT8 embeddings')
    # Do not let lower-precision features masquerade as the FP32 cache.
    cache = Path(args[args.index('--cache-dir') + 1]) / f'dynamic-int8-vision-linear-{image_size}-v1'
    sys.argv[sys.argv.index('--cache-dir') + 1] = str(cache)
    load_model = AutoModel.from_pretrained
    load_processor = AutoImageProcessor.from_pretrained
    backend = 'x86' if 'x86' in torch.backends.quantized.supported_engines else 'fbgemm'
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError('No supported x86 dynamic quantization backend')
    torch.backends.quantized.engine = backend

    def load(*a, **kw):
        model = load_model(*a, **kw)
        if not hasattr(model, 'vision_model'):
            raise ValueError('Expected a SigLIP model with an independent vision tower')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            model.vision_model = torch.ao.quantization.quantize_dynamic(
                model.vision_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        image_features = model.get_image_features
        def features(**values):
            return image_features(**values, interpolate_pos_encoding=image_size != 448)
        model.get_image_features = features
        return model

    def processor(*a, **kw):
        instance = load_processor(*a, **kw)
        instance.size = {'height': image_size, 'width': image_size}
        return instance

    started = time.perf_counter()
    AutoModel.from_pretrained = load
    AutoImageProcessor.from_pretrained = processor
    try:
        probe.main()
    finally:
        AutoModel.from_pretrained = load_model
        AutoImageProcessor.from_pretrained = load_processor
    if '--smoke' not in args:
        output = Path(args[args.index('--output') + 1])
        report = json.loads(output.read_text())
        report.update(encoder_arithmetic='CPU dynamic INT8 vision Linear; remaining operators FP32',
                      quantization_backend=backend, elapsed_seconds=time.perf_counter() - started,
                      input_image_size=image_size,
                      interpolated_position_embeddings=image_size != 448,
                      quantization_code_sha256=probe.digest(Path(__file__)))
        report['limitations'].append('INT8 encoder is a separate exploratory candidate; no equivalence to FP32 is claimed.')
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'encoder_arithmetic': 'dynamic_int8_vision_linear',
                      'input_image_size': image_size,
                      'elapsed_seconds': time.perf_counter() - started}), flush=True)


if __name__ == '__main__':
    main()
