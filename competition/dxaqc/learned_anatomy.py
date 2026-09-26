"""Versioned automatic mask candidates, independent of clinical QC decisions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mask_raster import encode_raster
from .embedding import WEIGHTS_SHA256, _model, _MEAN, _STD
import json

CLASSES = ['background'] + [f'Th{i}' for i in range(1, 13)] + [f'L{i}' for i in range(1, 6)]
HIP_CLASSES = ['femur', 'hemipelvis', 'femoral_neck_roi']


def numbered_axis(regions, sx, sy):
    if not all(np.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError('Invalid axis calibration')
    by_name = {r['name']: r for r in regions}
    centers = [by_name[f'L{i}']['center'] for i in range(1, 5) if f'L{i}' in by_name]
    if len(centers) < 3:
        return None
    points = np.asarray(centers, dtype=float)*[sx, sy]
    if not np.isfinite(points).all() or np.any(np.diff(points[:, 1]) <= 0):
        return None
    return float(np.degrees(np.arctan(np.polyfit(points[:, 1], points[:, 0], 1)[0])))


def roi_proposals(regions):
    proposals = []
    for region in regions:
        if region['name'] not in ('L1', 'L2', 'L3', 'L4', 'femoral_neck_roi'):
            continue
        polygon = cv2.convexHull(np.asarray(region['polygon'], np.float32)).reshape(-1, 2)
        proposals.append({'purpose': region['name'], 'polygon': polygon.astype(float).tolist(),
                          'status': 'proposed', 'requires_confirmation': True,
                          'reference_origin': 'learned_mask', 'clinical_validation': False})
    return proposals


def spatial_features(pixels):
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
        raise ValueError('Expected a uint8 grayscale raster')
    image = cv2.resize(pixels, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    x = torch.from_numpy(((np.repeat(image[None], 3, axis=0)-_MEAN)/_STD)[None])
    backbone = _model()
    with torch.inference_mode():
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        x = backbone.layer1(x)
        maps = []
        for layer in (backbone.layer2, backbone.layer3, backbone.layer4):
            x = layer(x)
            maps.append(F.interpolate(x, size=(40, 40), mode='bilinear', align_corners=False))
    return torch.cat(maps, dim=1)[0].numpy().astype(np.float16)


def mask_head(channels=len(CLASSES)):
    return nn.Sequential(nn.Conv2d(896, 32, 1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, channels, 1))


def decode_masks(logits, shape):
    values = np.asarray(logits)
    if values.shape != (len(CLASSES), 40, 40) or not np.isfinite(values).all():
        raise ValueError('Invalid anatomical logits')
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError('Invalid output raster')
    index = values.argmax(axis=0).astype(np.uint8)
    return cv2.resize(index, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def mask_regions(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0 or not np.isin(labels, np.arange(len(CLASSES))).all():
        raise ValueError('Invalid named label raster')
    regions = []
    for value, name in enumerate(CLASSES[1:], 1):
        mask = (labels == value).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            continue
        largest = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = components == largest
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 4:
            continue
        polygon = cv2.approxPolyDP(contour, .01 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        ys, xs = np.where(component)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()+1), int(ys.min()), int(ys.max()+1)
        regions.append({'name': name, 'polygon': polygon.astype(float).tolist(),
                        'center': [float(xs.mean()), float(ys.mean())],
                        'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                        'status': 'candidate', 'verified': False, 'pixels': int(component.sum()),
                        'raster': encode_raster(component[y0:y1, x0:x1], (y0, x0))})
    return regions


class LearnedAnatomy:
    def __init__(self, path):
        path = Path(path)
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if (bundle.get('schema_version') != 1 or bundle.get('classes') != CLASSES
                or bundle.get('encoder_sha256') != WEIGHTS_SHA256
                or bundle.get('input_size') != 320 or bundle.get('grid') != 40
                or bundle.get('clinical_validation') is not False):
            raise ValueError('Incompatible automatic anatomy bundle')
        self.model = mask_head()
        self.model.load_state_dict(bundle['state_dict'], strict=True)
        self.model.eval()
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def predict(self, pixels, region):
        if region != 'spine':
            return {'status': 'not_applicable', 'regions': [], 'affects_decision': False}
        pixels = np.asarray(pixels)
        if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
            raise ValueError('Expected a uint8 grayscale raster')
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(spatial_features(pixels).astype(np.float32))[None])[0].numpy()
        regions = mask_regions(decode_masks(logits, pixels.shape))
        return {'status': 'evaluated', 'method': 'learned_named_mask',
                'model_sha256': self.sha256, 'coordinate_units': 'original_image_pixels',
                'regions': regions, 'roi_proposals': roi_proposals(regions),
                'target_semantics': 'vertebral quadrilaterals and named DXA ROI masks',
                'clinical_validation': False, 'affects_decision': False}


class LearnedHipAnatomy:
    def __init__(self, path):
        path = Path(path)
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if (bundle.get('schema_version') != 1 or bundle.get('classes') != HIP_CLASSES
                or bundle.get('encoder_sha256') != WEIGHTS_SHA256
                or bundle.get('input_size') != 320 or bundle.get('grid') != 40
                or bundle.get('clinical_validation') is not False):
            raise ValueError('Incompatible hip anatomy bundle')
        self.model = mask_head(len(HIP_CLASSES))
        self.model.load_state_dict(bundle['state_dict'], strict=True)
        self.model.eval()
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def predict(self, pixels, region):
        if region not in ('hip_left', 'hip_right'):
            return {'status': 'not_applicable', 'regions': [], 'affects_decision': False}
        pixels = np.asarray(pixels)
        if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
            raise ValueError('Expected a uint8 grayscale raster')
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(spatial_features(pixels).astype(np.float32))[None])[0]
            masks = (torch.sigmoid(logits).numpy() >= .5).astype(np.uint8)
        regions = []
        for index, name in enumerate(HIP_CLASSES):
            mask = cv2.resize(masks[index], (pixels.shape[1], pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
            for item in mask_regions(mask):
                item['name'] = name
                regions.append(item)
        return {'status': 'evaluated', 'method': 'learned_multilabel_hip_masks',
                'model_sha256': self.sha256, 'coordinate_units': 'original_image_pixels',
                'regions': regions, 'roi_proposals': roi_proposals(regions),
                'clinical_validation': False, 'affects_decision': False,
                'target_semantics': 'cadaver bone masks and named vendor neck ROI; no verified trochanter labels'}


class AnatomyPortfolio:
    """A pinned directory manifest, or the original single spine checkpoint."""
    def __init__(self, path):
        path = Path(path)
        if path.suffix != '.json':
            self.models = {'spine': LearnedAnatomy(path)}
            return
        manifest = json.loads(path.read_text())
        if manifest.get('schema_version') != 1 or not manifest.get('models'):
            raise ValueError('Invalid anatomy portfolio')
        self.models = {}
        for group, spec in manifest['models'].items():
            if group not in ('spine', 'hip'):
                raise ValueError('Unknown portfolio anatomy')
            model_path = (path.parent/spec['path']).resolve()
            if not model_path.is_relative_to(path.parent.resolve()):
                raise ValueError('Model path escapes portfolio')
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError('Anatomy model checksum mismatch')
            self.models[group] = (LearnedAnatomy if group == 'spine' else LearnedHipAnatomy)(model_path)

    def predict(self, pixels, region):
        group = 'spine' if region == 'spine' else 'hip'
        if group not in self.models:
            return {'status': 'unavailable', 'regions': [], 'affects_decision': False}
        return self.models[group].predict(pixels, region)
