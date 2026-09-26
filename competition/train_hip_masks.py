"""Subject-disjoint automatic hip masks with partially observed ROI supervision."""
import argparse
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
import pydicom
import torch
from torch.nn import functional as F

from anatomy_training_data import records, split_records
from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.learned_anatomy import HIP_CLASSES, mask_head, spatial_features
from evaluate_external_roi_masks import read_seg_nrrd
from evaluate_organizer_dataset import sha256


def loss_with_unknown(logits, target):
    if logits.shape != target.shape or not (target >= 0).any():
        raise ValueError('No known compatible mask targets')
    known = target >= 0
    loss = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1), reduction='none')
    return loss[known].mean()


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(4); torch.manual_seed(17)
    rows = []
    for patient in sorted(args.deepfluoro.glob('subject*')):
        for image in sorted((patient/'xrays').glob('*.dcm')):
            annotation = patient/'segmentations_multichannel'/f'{image.stem}.pt'
            rows.append({'image': str(image), 'annotation': str(annotation), 'source': 'deepfluoro',
                         'group': patient.name, 'split': 'test' if patient.name == 'subject06' else 'train'})
    spine_rows, _ = records(args.aasce, args.dxa)
    test = split_records(spine_rows)
    for row, held in zip(spine_rows, test):
        if row['source'] == 'ramathibodi':
            image = Path(row['image']).with_name('hip_image.dcm')
            annotation = Path(row['annotation']).with_name('hip_image.seg.nrrd')
            rows.append({'image': str(image), 'annotation': str(annotation), 'source': 'ramathibodi',
                         'group': row['group'], 'split': 'test' if held else 'train'})
    args.output.mkdir(parents=True)
    cache = np.lib.format.open_memmap(args.output/'features.npy', mode='w+', dtype=np.float16,
                                    shape=(len(rows), 896, 40, 40))
    targets = []
    for i, row in enumerate(rows):
        if row['source'] == 'deepfluoro':
            ds = pydicom.dcmread(row['image'])
            raw = ds.pixel_array
            pixels = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if str(ds.PhotometricInterpretation) == 'MONOCHROME1':
                pixels = 255-pixels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                packed = torch.load(row['annotation'], weights_only=True).numpy()
            if packed.shape != pixels.shape:
                raise ValueError('Projected mask does not match X-ray raster')
            target = np.stack([((packed >> 4) & 1) | ((packed >> 5) & 1),
                               (packed & 1) | ((packed >> 1) & 1), np.full(pixels.shape, -1.)])
        else:
            pixels = read_dxa(Path(row['image'])).pixels
            masks = read_seg_nrrd(Path(row['annotation']), pixels.shape)
            target = np.full((3, *pixels.shape), -1., dtype=np.float32)
            target[2] = masks['Femoral_Neck_bone_area']
        cache[i] = spatial_features(pixels)
        targets.append(np.stack([cv2.resize(channel.astype(np.float32), (40, 40),
                                            interpolation=cv2.INTER_NEAREST) for channel in target]))
        row.update(image_sha256=sha256(Path(row['image'])), annotation_sha256=sha256(Path(row['annotation'])))
        if (i+1) % 25 == 0:
            print(f'Encoded hip {i+1}/{len(rows)}', flush=True)
    cache.flush(); targets = np.asarray(targets)
    training = np.array([i for i, row in enumerate(rows) if row['split'] == 'train'])
    testing = np.array([i for i, row in enumerate(rows) if row['split'] == 'test'])
    if {rows[i]['group'] for i in training} & {rows[i]['group'] for i in testing}:
        raise ValueError('Subject leakage')
    dxa_training = [i for i in training if rows[i]['source'] == 'ramathibodi']
    schedule = np.r_[training, np.tile(dxa_training, 12)]
    model = mask_head(3); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(17)
    for epoch in range(args.epochs):
        model.train(); losses = []
        order = rng.permutation(schedule)
        for start in range(0, len(order), 8):
            ids = order[start:start+8]
            x = torch.from_numpy(np.array(cache[ids], dtype=np.float32))
            y = torch.from_numpy(targets[ids].copy())
            loss = loss_with_unknown(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        print(f'Hip epoch {epoch+1}/{args.epochs}: {np.mean(losses):.4f}', flush=True)
    model.eval(); cases = []
    with torch.inference_mode():
        for i in testing:
            predictions = torch.sigmoid(model(torch.from_numpy(np.array(cache[i], dtype=np.float32))[None]))[0].numpy() >= .5
            result = []
            source_train = [j for j in training if rows[j]['source'] == rows[i]['source']]
            for channel, name in enumerate(HIP_CLASSES):
                known = targets[i, channel] >= 0
                if not known.any():
                    continue
                truth = targets[i, channel] == 1
                prediction = predictions[channel] & known
                prior = (targets[source_train, channel] == 1).mean(axis=0) >= .5
                dice = lambda x: 2*int((x & truth).sum())/max(1, int(x.sum()+truth.sum()))
                result.append({'name': name, 'dice': dice(prediction), 'layout_prior_dice': dice(prior & known)})
            cases.append({'source': rows[i]['source'], 'group': rows[i]['group'], 'masks': result})
    path = args.output/'hip_masks.pt'
    torch.save({'schema_version': 1, 'classes': HIP_CLASSES, 'encoder_sha256': WEIGHTS_SHA256,
                'input_size': 320, 'grid': 40, 'state_dict': model.state_dict(), 'clinical_validation': False}, path)
    summary = {}
    for source in sorted({case['source'] for case in cases}):
        selected = [c for c in cases if c['source'] == source]
        summary[source] = {k: float(np.mean([m[k] for c in selected for m in c['masks']]))
                           for k in ('dice', 'layout_prior_dice')}
        summary[source]['images'] = len(selected)
        summary[source]['subjects'] = len({c['group'] for c in selected})
    report = {'protocol': 'DeepFluoro subject06 withheld; same two Hologic patients as spine withheld; partial labels masked in BCE; fixed 20 epochs',
              'summary': summary, 'cases': cases, 'source_inventory': rows,
              'model_sha256': sha256(path), 'code_sha256': sha256(Path(__file__)),
              'clinical_validation': False, 'release_modified': False,
              'limitations': ['One cadaver and two Hologic patients in holdout; no clinical GE mask annotations.',
                              'Femoral neck target is vendor bone-area ROI, not a reviewed anatomical neck outline.',
                              'No independent greater/lesser trochanter or ischium supervision.',
                              'Dice measured on 40x40 grid.']}
    (args.output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('deepfluoro', 'aasce', 'dxa', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error('epochs must be 1..100')
    run(args)
