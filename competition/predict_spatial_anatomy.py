"""Offline numbered-center candidate. Coordinates are pixels; no DXA validity claimed."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import WEIGHTS_SHA256
from evaluate_organizer_dataset import sha256
from train_spatial_anatomy import SIZE, GRID, head, decode, spatial_embedding


def load(path):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if (bundle.get('encoder_sha256') != WEIGHTS_SHA256
            or bundle.get('levels') != ['L1', 'L2', 'L3', 'L4', 'L5']
            or bundle.get('input_size') != SIZE or bundle.get('heatmap_grid') != GRID
            or bundle.get('clinical_validation') is not False):
        raise ValueError('Incompatible spatial anatomy candidate')
    model = head()
    model.load_state_dict(bundle['state_dict'], strict=True)
    return model.eval()


def predict(model, pixels, ordered=False):
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or pixels.size == 0:
        raise ValueError('Expected a nonempty uint8 grayscale image')
    with torch.inference_mode():
        logits = model(torch.from_numpy(spatial_embedding(pixels).astype(np.float32))[None])
    if ordered:
        from ordered_spatial_centers import decode_ordered
        centers = decode_ordered(logits.numpy())[0]
    else:
        centers = decode(logits.numpy())[0]
    return centers * np.array([pixels.shape[1], pixels.shape[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--ordered', action='store_true', help='Joint order-prior decoding of L1-L4 only; still unverified candidates')
    args = parser.parse_args()
    torch.set_num_threads(4)
    pixels = (read_dxa(args.input).pixels if args.input.suffix.lower() == '.dcm'
              else cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE))
    centers = predict(load(args.model), pixels, ordered=args.ordered)
    print(json.dumps({'status': 'candidate', 'verified': False, 'clinical_validation': False,
                      'model_sha256': sha256(args.model), 'coordinate_units': 'pixels',
                      'decoding': 'joint_order_prior' if args.ordered else 'independent_heatmaps',
                      'centers': {f'L{i+1}': p.tolist() for i, p in enumerate(centers)}}, indent=2))
