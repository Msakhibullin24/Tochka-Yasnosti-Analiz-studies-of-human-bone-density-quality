"""Train named mask heads on frozen dense features and confirmed expert contours.

This is a research anatomy baseline, not a complete TZ verdict. No training
occurs on blank templates, generated masks or assumed vertebral identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_research.common import CATALOG, digest, dump
from dxaqc.dicom_io import read_any
from review_geometry import contour_mask

CLASSES = {'spine': ('Th12', 'L1', 'L2', 'L3', 'L4'),
           'hip': ('femur', 'greater_trochanter', 'lesser_trochanter', 'ischium')}


def reviewed_target(case, names):
    """Overlapping anatomical regions are independent channels; absent annotation remains unknown."""
    h, w = case['height'], case['width']
    target = np.full((len(names), h, w), -1., np.float32)
    for channel, name in enumerate(names):
        label = case.get('named_regions', {}).get(name)
        if label is None or label.get('visible') is None:
            continue
        if label.get('visible') is False:
            if label.get('polygon') is not None:
                raise ValueError('Invisible structure must not have a contour')
            target[channel] = 0
        elif label.get('visible') is True:
            target[channel] = contour_mask(label.get('polygon'), (h, w))
        else:
            raise ValueError('Visibility must be true, false or unknown')
    return target


def to_patch_grid(target, transform, grid, mirrored):
    if mirrored:
        target = target[:, :, ::-1]
    size = transform['canvas_size']
    x, y, w, h = (transform[k] for k in ('x', 'y', 'width', 'height'))
    # Padding is excluded from loss, including for explicitly invisible structures.
    canvas = np.full((len(target), size, size), -1., np.float32)
    for channel, mask in enumerate(target):
        canvas[channel, y:y+h, x:x+w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.stack([cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_NEAREST) for mask in canvas])


def loss_known(logits, target):
    known = target >= 0
    if not known.any():
        raise ValueError('No confirmed compatible contour targets')
    bce = nn.functional.binary_cross_entropy_with_logits(logits.float(), target.clamp(0, 1), reduction='none')
    return bce[known].mean()


def train(args):
    if args.output.exists() or args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Choose a new output directory and positive epochs/batch size')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    reference = json.loads(args.reference.read_text())
    if (reference.get('review_package_version') != 4 or not reference.get('reviewer')
            or not reference.get('annotation_source')
            or reference.get('coordinate_system') != 'original_pixel_centres'):
        raise ValueError('Expert provenance and v4 original-coordinate review are required')
    metadata = json.loads((args.features / 'features.json').read_text())
    if (metadata.get('encoder_finetuned') is not False or not metadata.get('dense_features')
            or metadata.get('smoke_only') or metadata.get('features_sha256') != digest(args.features / 'features.npy')):
        raise ValueError('Complete frozen dense feature bundle required')
    paths = metadata['paths']
    if len(set(paths)) != len(paths):
        raise ValueError('Ambiguous feature paths')
    indexed = dict(zip(paths, range(len(paths))))
    names = CLASSES[args.region]
    patch_size = 14 if CATALOG[metadata['model']]['kind'] == 'siglip' else 16
    grid = metadata['size'] // patch_size
    x, y, cases, unreviewed = [], [], [], 0
    groups, seen_pixels, seen_uids = [], {}, set()
    for case in reference.get('cases', []):
        if case.get('status') != 'confirmed':
            unreviewed += 1
            continue
        if not any(name in case.get('named_regions', {}) for name in names):
            continue
        target = reviewed_target(case, names)
        if not (target >= 0).any():
            raise ValueError('Confirmed case has no contour targets for requested region')
        rel = case['path_to_file']
        if rel not in indexed:
            raise ValueError('Confirmed source is absent from the dense feature bundle')
        source = (args.dataset / rel).resolve()
        if Path(rel).is_absolute() or not source.is_relative_to(args.dataset.resolve()):
            raise ValueError('Review source path escapes dataset')
        image = read_any(source)
        if (digest(source) != case['source_sha256'] or image.pixel_sha256 != case['pixel_sha256']
                or image.image_uid != case['image_uid'] or image.pixels.shape != (case['height'], case['width'])):
            raise ValueError('Reviewed source identity or geometry differs')
        if image.image_uid in seen_uids:
            raise ValueError('Repeated reviewed image UID')
        seen_uids.add(image.image_uid)
        group = case.get('patient_group') or case.get('study_group')
        if not isinstance(group, str) or not group.strip():
            raise ValueError('Reviewed source requires an explicit grouping identity')
        earlier = seen_pixels.setdefault(image.pixel_sha256, group)
        if earlier != group:
            raise ValueError('Pixel duplicate crosses review groups')
        i = indexed[rel]
        transform = metadata['transforms'][i]
        if (transform.get('path') != rel or transform.get('pixel_sha256') != image.pixel_sha256
                or transform.get('source_width') != case['width'] or transform.get('source_height') != case['height']):
            raise ValueError('Feature transform order differs from source order')
        patch_path = args.features / f'patches_{i:03d}.npy'
        if metadata.get('dense_sha256', {}).get(patch_path.name) != digest(patch_path):
            raise ValueError('Dense feature checksum differs')
        tokens = np.load(patch_path, allow_pickle=False)
        if tokens.ndim != 2 or len(tokens) != grid * grid or not np.isfinite(tokens).all():
            raise ValueError('Invalid dense patch features')
        x.append(tokens.T.reshape(-1, grid, grid).astype(np.float32))
        y.append(to_patch_grid(target, {**transform, 'canvas_size': metadata['size']}, grid, transform['mirrored']))
        groups.append(group)
        cases.append({'path': rel, 'group': group, 'pixel_sha256': image.pixel_sha256})
    if len(set(groups)) < 8:
        raise ValueError('At least 8 independently grouped reviewed cases required for train/validation/test')
    indices = np.arange(len(cases))
    groups = np.asarray(groups)
    trainval, test = next(GroupShuffleSplit(1, test_size=.2, random_state=17).split(indices, groups=groups))
    tr, va = next(GroupShuffleSplit(1, test_size=.25, random_state=18).split(trainval, groups=groups[trainval]))
    training, validation = trainval[tr], trainval[va]
    for a, b in ((training, validation), (training, test), (validation, test)):
        if set(groups[a]) & set(groups[b]):
            raise ValueError('Anatomy group leakage')
    torch.manual_seed(17)
    torch.set_num_threads(4)
    values, targets = torch.tensor(np.stack(x)), torch.tensor(np.stack(y))
    model = nn.Sequential(nn.Conv2d(values.shape[1], 128, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(128, len(names), 1)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_loss, history = None, float('inf'), []
    rng = np.random.default_rng(17)
    def predict(part):
        with torch.no_grad():
            return torch.cat([model(values[part[pos:pos+args.batch_size]].to(args.device)).cpu()
                              for pos in range(0, len(part), args.batch_size)])
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(training)
        for pos in range(0, len(order), args.batch_size):
            part = order[pos:pos+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_known(model(values[part].to(args.device)), targets[part].to(args.device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite anatomy training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        validation_loss = float(loss_known(predict(validation), targets[validation]))
        history.append({'epoch': epoch+1, 'validation_loss': validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f'epoch={epoch+1} validation_loss={validation_loss:.5f}', flush=True)
    model.load_state_dict(best)
    predicted = torch.sigmoid(predict(test)).numpy() >= .5
    truth = targets[test].numpy()
    metrics = {}
    for channel, name in enumerate(names):
        known = truth[:, channel] >= 0
        a, b = truth[:, channel] == 1, predicted[:, channel] & known
        union = int(a.sum() + b.sum())
        metrics[name] = {'dice_at_patch_grid': 2 * int((a & b).sum()) / union if union else None,
                         'reference_positive_pixels': int(a.sum()), 'predicted_positive_pixels': int(b.sum()),
                         'known_pixels': int(known.sum())}
    args.output.mkdir(parents=True)
    torch.save({'state_dict': best, 'classes': names, 'grid': grid, 'size': metadata['size'],
                'encoder_weights_sha256': metadata['weights_sha256'], 'clinical_validation': False}, args.output / 'mask_head.pt')
    np.savez_compressed(args.output / 'test_patch_predictions.npz', predicted=predicted, reference=truth)
    dump(args.output / 'evaluation.json', {
        'protocol': __doc__, 'metrics': metrics, 'history': history,
        'training': [cases[i] for i in training], 'validation': [cases[i] for i in validation],
        'test': [cases[i] for i in test], 'unreviewed_cases_excluded': unreviewed,
        'patient_group_available_for_all': all(case.get('patient_group') for case in reference['cases'] if case.get('status') == 'confirmed'),
        'reference_sha256': digest(args.reference), 'feature_metadata_sha256': digest(args.features / 'features.json'),
        'code_sha256': digest(__file__), 'clinical_validation': False, 'model_affects_decision': False,
        'limitations': ['Internal grouped split; not independent clinical validation.',
                       'Dice measured on coarse patch grid, not source-resolution anatomy.',
                       'No landmark visibility head, angle estimator, disc numbering or neck ROI acceptance implemented here.',
                       'Expert identity/independence cannot be authenticated by a JSON field.']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--region', choices=CLASSES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8)
    train(p.parse_args())


if __name__ == '__main__':
    main()
