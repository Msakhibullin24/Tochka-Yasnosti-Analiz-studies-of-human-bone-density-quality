"""Offline architectural smoke tests for downloaded checkpoints (not accuracy tests)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

from dxaqc.specialist_qc import create_encoder, digest


def shape_model(output_nodes: int) -> nn.Module:
    """Reconstruct upstream RegressionModel_Transformer without import-time downloads.

    Architecture: EmmanuelleB985/DXA-to-3D model.py, MIT, copyright 2024
    Emmanuelle Bourigault. See docs/competition/DXA_TO_3D_LICENSE.txt.
    Output order/preprocessing are NOT established by this compatibility test.
    """
    from torchvision.models import resnet50
    from bottleneck_transformer_pytorch import BottleStack

    class ShapeRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
            self.layer = BottleStack(dim=2048, fmap_size=7, dim_out=2048, proj_factor=4,
                                     downsample=False, heads=4, dim_head=128, rel_pos_emb=True,
                                     activation=nn.ReLU())
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten(1)
            self.linear = nn.Linear(2048, output_nodes)

        def forward(self, x):
            return self.linear(self.flatten(self.pool(self.layer(self.model(x)))))

    return ShapeRegressor().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('data/specialists'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    catalog = json.loads((Path(__file__).resolve().parents[1]/'docs/competition/specialist_sources.json').read_text())
    artifacts = {a['id']: a for a in catalog['artifacts']}
    results = {}
    for name, artifact_id in [('convnextv2_tiny', 'convnextv2_tiny_weights'),
                              ('tf_efficientnet_b4', 'efficientnet_b4_weights'),
                              ('dxa_to_3d', 'dxa_to_3d_weights')]:
        spec = artifacts[artifact_id]
        path = args.assets/spec['path']
        try:
            if digest(path) != spec['sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            if name == 'dxa_to_3d':
                state = torch.load(path, map_location='cpu', weights_only=True)
                model = shape_model(state['linear.weight'].shape[0])
                model.load_state_dict(state, strict=True)
            else:
                model = create_encoder(name, path)
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 224, 224))
            if not torch.isfinite(output).all():
                raise ValueError('Non-finite output')
            results[name] = {'status': 'architecture_verified', 'output_shape': list(output.shape),
                             'checkpoint_sha256': spec['sha256'], 'clinical_validation': False}
            del model
        except Exception as exc:
            results[name] = {'status': 'error', 'error': str(exc)}
    results['limitations'] = {
        'dxa_to_3d': 'Raw regression output only; preprocessing and curve order unverified. No Th12/L1-L4 labels.',
        'yolo26x': 'Downloaded COCO detection/segmentation weights; no DXA implant training.',
    }
    os.environ.setdefault('YOLO_CONFIG_DIR', str((args.assets/'ultralytics-config').resolve()))
    for artifact_id, task in [('yolo26x_weights', 'detect'), ('yolo26x_seg_weights', 'segment'), ('yolo26x_pose_weights', 'pose')]:
        try:
            import numpy as np
            from ultralytics import YOLO
            spec = artifacts[artifact_id]
            path = args.assets/spec['path']
            if digest(path) != spec['sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            detector = YOLO(str(path))
            expected_classes = 1 if task == 'pose' else 80
            if detector.task != task or len(detector.names) != expected_classes:
                raise ValueError('Unexpected generic detector vocabulary')
            output = detector.predict(np.zeros((224, 224, 3), dtype=np.uint8),
                                      device='cpu', imgsz=224, verbose=False, save=False)[0]
            results[artifact_id] = {'status': 'architecture_verified_generic_coco', 'task': task,
                                    'classes': expected_classes, 'detections_on_blank': len(output.boxes),
                                    'checkpoint_sha256': spec['sha256'],
                                    'dxa_artefact_model': False}
        except Exception as exc:
            results[artifact_id] = {'status': 'error', 'error': str(exc)}
    (args.assets/'checkpoint-checks.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
    raise SystemExit(int(any(r.get('status') == 'error' for r in results.values())))


if __name__ == '__main__':
    main()
