"""Experimental image QC. Offline TorchScript inference; no downloads at runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .embedding import letterbox
from .model import CRITERIA, group_of

OUTPUTS = ('quality', *CRITERIA['spine'], *CRITERIA['hip'])
SCHEMA_VERSION = 1
PREPROCESS_VERSION = 'grayscale-letterbox-imagenet-v1'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_tensor(pixels: np.ndarray, size: int) -> torch.Tensor:
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
        raise ValueError('Expected non-empty uint8 grayscale image')
    x = torch.from_numpy(letterbox(pixels, size).copy()).float().div_(255)
    x = x.unsqueeze(0).repeat(3, 1, 1)
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, kind: str = 'bce') -> torch.Tensor:
    """NaN means unlabelled/not applicable; it contributes neither loss nor gradient.

    ASL uses negative probability clipping and asymmetric focusing (gamma-=4,
    gamma+=0), following the published ASL formulation. Reduction is by known label.
    """
    if (logits.shape != targets.shape or kind not in ('bce', 'asl')
            or not torch.isfinite(logits).all() or torch.isinf(targets).any()):
        raise ValueError('Invalid loss inputs')
    known = torch.isfinite(targets)
    if torch.any(known & (targets != 0) & (targets != 1)):
        raise ValueError('Targets must be binary or NaN')
    y = torch.nan_to_num(targets, nan=0.)
    if kind == 'bce':
        terms = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    else:
        positive = torch.sigmoid(logits)
        negative = (1 - positive + .05).clamp(max=1)
        terms = -(y * positive.clamp(min=1e-8).log() + (1-y) * negative.clamp(min=1e-8).log())
        with torch.no_grad():
            weight = (1 - positive*y - negative*(1-y)).pow(4*(1-y))
        terms = terms * weight
    return torch.where(known, terms, torch.zeros_like(terms)).sum() / known.sum().clamp(min=1)


class ImageQC(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MultiViewQC(nn.Module):
    """Full frame and fixed central 70% view; this crop is not an anatomical detector."""
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder, self.head = encoder, head

    @staticmethod
    def center_view(x: torch.Tensor) -> torch.Tensor:
        margin = x.shape[-1] * 15 // 100
        return F.interpolate(x[:, :, margin:-margin, margin:-margin], size=x.shape[-2:],
                             mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(x), self.encoder(self.center_view(x))], dim=1))


def create_encoder(name: str, weights: Path) -> nn.Module:
    import timm
    from safetensors.torch import load_file

    if name not in ('convnextv2_tiny', 'tf_efficientnet_b4'):
        raise ValueError('Unsupported specialist backbone')
    # Load the complete ImageNet classifier strictly before removing its head.
    model = timm.create_model(name, pretrained=False)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.reset_classifier(0)
    return model.eval()


class SpecialistQC:
    """Shadow prediction only: never changes the accepted product decision."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'model.json').read_text())
        m = self.metadata
        if (m.get('schema_version') != SCHEMA_VERSION or m.get('outputs') != list(OUTPUTS)
                or m.get('preprocess_version') != PREPROCESS_VERSION or m.get('mode') != 'shadow'):
            raise ValueError('Unsupported specialist model contract')
        if not isinstance(m.get('input_size'), int) or not 32 <= m['input_size'] <= 1024:
            raise ValueError('Invalid specialist input size')
        if set(m.get('thresholds', {})) != set(OUTPUTS):
            raise ValueError('Missing specialist thresholds')
        for value in m['thresholds'].values():
            if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Invalid specialist threshold')
        path = self.directory / 'model.ts'
        if digest(path) != m.get('model_sha256'):
            raise ValueError('Specialist checksum mismatch')
        self.model = torch.jit.load(str(path), map_location='cpu').eval()

    def predict(self, pixels: np.ndarray, region: str) -> dict:
        if region not in ('spine', 'hip_left', 'hip_right'):
            raise ValueError('Unsupported specialist region')
        scope = self.metadata.get('region_scope', 'all')
        if scope not in ('all', group_of(region)):
            raise ValueError('Specialist was trained for a different anatomical region')
        x = image_tensor(pixels, self.metadata['input_size']).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(x)
            if tuple(logits.shape) != (1, len(OUTPUTS)) or not torch.isfinite(logits).all():
                raise ValueError('Invalid specialist output')
            scores = torch.sigmoid(logits)[0].tolist()
        applicable = {'quality', *CRITERIA[group_of(region)]}
        predictions = {}
        for name, score in zip(OUTPUTS, scores):
            threshold = self.metadata['thresholds'][name]
            status = ('not_applicable' if name not in applicable else 'undetermined' if threshold is None
                      else 'fail' if score >= threshold else 'pass')
            predictions[name] = {'score': score if name in applicable else None,
                                 'threshold': threshold if name in applicable else None, 'status': status}
        return {'mode': 'shadow', 'clinical_validation': False, 'affects_decision': False,
                'model_sha256': self.metadata['model_sha256'], 'predictions': predictions}


class RegionalSpecialists:
    """Verified regional candidates; select exactly one model for the routed region."""
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        path = self.directory/'suite.json'
        spec = json.loads(path.read_text())
        if spec.get('schema_version') != 1 or set(spec.get('members',{})) != {'spine','hip'}:
            raise ValueError('Invalid regional specialist suite')
        self.models = {}
        for region, entry in spec['members'].items():
            child = (self.directory/entry['path']).resolve()
            if not child.is_relative_to(self.directory) or child == self.directory:
                raise ValueError('Specialist member path escapes suite')
            if digest(child/'model.json') != entry['metadata_sha256']:
                raise ValueError('Specialist member metadata checksum mismatch')
            model = SpecialistQC(child)
            if model.metadata.get('region_scope') != region:
                raise ValueError('Specialist region mismatch')
            self.models[region] = model
        self.metadata = {'mode':'shadow','model_sha256':digest(path),'kind':'regional_suite'}

    def predict(self, pixels: np.ndarray, region: str) -> dict:
        if region not in ('spine','hip_left','hip_right'):
            raise ValueError('Unsupported specialist region')
        result = self.models[group_of(region)].predict(pixels,region)
        result['region_scope'] = group_of(region)
        result['suite_sha256'] = self.metadata['model_sha256']
        return result


def load_specialist(directory: Path):
    return RegionalSpecialists(directory) if (Path(directory)/'suite.json').is_file() else SpecialistQC(directory)
