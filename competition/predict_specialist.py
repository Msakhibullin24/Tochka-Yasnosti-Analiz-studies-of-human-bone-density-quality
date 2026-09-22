"""Offline shadow QC for one DICOM/raster; optional class-specific Grad-CAM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dxaqc.dicom_io import read_any
from dxaqc.specialist_qc import (OUTPUTS, SpecialistQC, ImageQC, create_encoder,
                               digest, image_tensor)


def class_cam(candidate: SpecialistQC, pixels: np.ndarray, weights: Path, name: str) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from safetensors.torch import load_file

    meta = candidate.metadata
    if meta.get('views', 'full') != 'full':
        raise ValueError('CAM is not yet defined for the two-view model')
    if digest(weights) != meta['backbone_sha256']:
        raise ValueError('CAM backbone checksum mismatch')
    head_path = candidate.directory / 'head.safetensors'
    if digest(head_path) != meta['head_sha256']:
        raise ValueError('CAM head checksum mismatch')
    encoder = create_encoder(meta['backbone'], weights)
    if meta.get('encoder_sha256'):
        trained = candidate.directory/'encoder.safetensors'
        if digest(trained) != meta['encoder_sha256']:
            raise ValueError('Fine-tuned encoder checksum mismatch')
        encoder.load_state_dict(load_file(str(trained)), strict=True)
    head = torch.nn.Linear(encoder.num_features, len(OUTPUTS))
    head.load_state_dict(load_file(str(head_path)), strict=True)
    model = ImageQC(encoder, head).eval()
    layer = encoder.stages[-1] if meta['backbone'] == 'convnextv2_tiny' else encoder.conv_head
    x = image_tensor(pixels, meta['input_size'])[None]
    with torch.no_grad():
        if not torch.allclose(model(x), candidate.model(x), atol=1e-4, rtol=1e-4):
            raise ValueError('CAM model differs from deployed candidate')
    with GradCAM(model=model, target_layers=[layer]) as cam:
        activation = cam(input_tensor=x, targets=[ClassifierOutputTarget(OUTPUTS.index(name))])[0]
    # Undo exactly the letterbox used for inference, preserving original orientation.
    h, w = pixels.shape
    size = meta['input_size']
    scale = size / max(h, w)
    nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
    y, x0 = (size-nh)//2, (size-nw)//2
    return cv2.resize(activation[y:y+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--region', choices=['spine', 'hip_left', 'hip_right'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cam', choices=OUTPUTS)
    parser.add_argument('--weights', type=Path, help='Local original backbone safetensors; required for CAM')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    if args.cam and not args.weights:
        parser.error('--cam requires --weights')
    torch.set_num_threads(4)
    img = read_any(args.input)
    model = SpecialistQC(args.model)
    result = model.predict(img.pixels, args.region)
    if args.cam and result['predictions'][args.cam]['status'] in ('not_applicable', 'undetermined'):
        parser.error('CAM requested for an unsupported or uncalibrated class')
    args.output.mkdir(parents=True)
    if args.cam:
        heatmap = class_cam(model, img.pixels, args.weights, args.cam)
        np.save(args.output/'cam.npy', heatmap)
        color = cv2.applyColorMap((heatmap.clip(0,1)*255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img.pixels, cv2.COLOR_GRAY2BGR), .65, color, .35, 0)
        cv2.imencode('.png', overlay)[1].tofile(str(args.output/'cam.png'))
        result['explanation'] = {'method': 'GradCAM', 'class': args.cam,
                                 'meaning': 'model_attribution_not_defect_localization',
                                 'coordinates': 'original_image', 'width': img.pixels.shape[1],
                                 'height': img.pixels.shape[0]}
    (args.output/'prediction.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
