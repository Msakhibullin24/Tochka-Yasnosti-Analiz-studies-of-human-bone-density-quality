"""Train polarity invariance on whole fields as well as lumbar fragments."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from anatomy_training_data import load_record
from dxaqc.learned_anatomy import mask_head, spatial_features
from evaluate_organizer_dataset import sha256
from refine_lumbar_masks import lumbar_crop
from train_anatomy_masks import metrics


def photometric_variant(pixels, kind):
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
        raise ValueError('Expected uint8 grayscale image')
    if kind == 'inverse':
        return 255-pixels
    if kind in ('dark', 'bright'):
        gamma = 1.5 if kind == 'dark' else .7
        lut = np.rint(255*(np.arange(256)/255)**gamma).astype(np.uint8)
        return lut[pixels]
    raise ValueError('Unknown photometric transform')


def run(base, lumbar, output):
    if output.exists():
        raise ValueError('Choose a new training directory')
    torch.set_num_threads(4); torch.manual_seed(17)
    rows = json.loads((base/'inventory.json').read_text())['rows']
    source_cache = np.load(base/'features.npy', mmap_mode='r', allow_pickle=False)
    crop_cache = np.load(lumbar/'crop_features.npy', mmap_mode='r', allow_pickle=False)
    if source_cache.shape != (len(rows), 896, 40, 40) or source_cache.dtype != np.float16:
        raise ValueError('Invalid source cache')
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    original_targets, crop_targets, crop_parents, augmented, augmentation = [], [], [], [], []
    heldout = []
    for i, row in enumerate(rows):
        pixels, target = load_record(row)
        original_targets.append(cv2.resize(target.astype(np.float32), (40,40), interpolation=cv2.INTER_NEAREST).astype(np.int64))
        if row['split'] == 'test':
            heldout.append((row, 'original', pixels, target))
        else:
            for kind in (('inverse', 'dark', 'bright') if row['source']=='ramathibodi' else ('inverse',)):
                augmented.append(photometric_variant(pixels, kind))
                augmentation.append({'parent_index': i, 'group': row['group'], 'kind': kind})
        if row['source'] == 'aasce':
            pixels, target = lumbar_crop(pixels, target)
            if row['split'] == 'test':
                heldout.append((row, 'lumbar_crop', pixels, target))
            else:
                for _ in range(2):
                    crop_targets.append(cv2.resize(target.astype(np.float32), (40,40), interpolation=cv2.INTER_NEAREST).astype(np.int64))
                    crop_parents.append(row['group'])
    previous = json.loads((lumbar/'evaluation.json').read_text())
    if crop_parents != previous['crop_parents'] or crop_cache.shape != (len(crop_targets), 896,40,40) or crop_cache.dtype != np.float16:
        raise ValueError('Crop cache parent ordering differs')
    if sha256(base/'features.npy') != previous['base_features_sha256']:
        raise ValueError('Original feature cache changed since lumbar training')
    if any(rows[a['parent_index']]['split'] != 'train' for a in augmentation):
        raise ValueError('Held-out parent augmentation in training')
    output.mkdir(parents=True)
    extra = np.lib.format.open_memmap(output/'photometric_features.npy', mode='w+', dtype=np.float16,
                                      shape=(len(augmented), 896,40,40))
    for i, raster in enumerate(augmented):
        extra[i] = spatial_features(raster)
        if (i+1)%100 == 0:
            print(f'Encoded photometry {i+1}/{len(augmented)}', flush=True)
    extra.flush()
    offset = len(rows)+len(crop_targets)
    targets = np.concatenate([original_targets, crop_targets,
                              [original_targets[a['parent_index']] for a in augmentation]])
    dxa = [i for i in train if rows[i]['source']=='ramathibodi']
    extra_dxa = [offset+j for j,a in enumerate(augmentation) if rows[a['parent_index']]['source']=='ramathibodi']
    # Keep whole-field originals frequent while training both polarity families.
    schedule = np.r_[np.tile(train,2), np.arange(len(rows),offset), np.arange(offset,len(targets)),
                     np.tile(dxa,24), np.tile(extra_dxa,8)]
    bundle = torch.load(lumbar/'anatomy_masks.pt', weights_only=True)
    model = mask_head(); model.load_state_dict(bundle['state_dict']); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    rng = np.random.default_rng(17); weights = torch.tensor([.2]+[1.]*17)
    def feature(i):
        return source_cache[i] if i<len(rows) else crop_cache[i-len(rows)] if i<offset else extra[i-offset]
    for epoch in range(10):
        losses=[]; order=rng.permutation(schedule)
        for start in range(0,len(order),8):
            ids=order[start:start+8]
            logits=model(torch.from_numpy(np.array([feature(i) for i in ids],dtype=np.float32)))
            loss=F.cross_entropy(logits,torch.from_numpy(targets[ids]),weight=weights,ignore_index=-100)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        print(f'Photometry epoch {epoch+1}/10: {np.mean(losses):.4f}',flush=True)
    model.eval(); cases=[]
    with torch.inference_mode():
        for row,view,pixels,target in heldout:
            reference=cv2.resize(target.astype(np.float32),(40,40),interpolation=cv2.INTER_NEAREST).astype(np.int64)
            for kind in ('original','inverse'):
                raster=pixels if kind=='original' else photometric_variant(pixels,kind)
                prediction=model(torch.from_numpy(spatial_features(raster).astype(np.float32))[None])[0].argmax(0).numpy()
                scores=metrics(prediction,reference)
                cases.append({'source':row['source'],'group':row['group'],'view':view,'polarity':kind,
                              'dice':float(np.mean([m['dice'] for m in scores]))})
    bundle.update(state_dict=model.state_dict(),photometry_adaptation='train-parent-only full field inversions; DXA gamma variants',clinical_validation=False)
    torch.save(bundle,output/'anatomy_masks.pt')
    summary={}
    for c in cases:
        key='/'.join(c[k] for k in ('source','view','polarity')); summary.setdefault(key,[]).append(c['dice'])
    summary={k:{'mean_image_dice':float(np.mean(v)),'images':len(v)} for k,v in summary.items()}
    report={'protocol':'Fixed source splits; stage2 head; 10 fixed epochs, no holdout selection',
            'summary':summary,'cases':cases,'augmentations':augmentation,'clinical_validation':False,
            'base_model_sha256':sha256(lumbar/'anatomy_masks.pt'),'model_sha256':sha256(output/'anatomy_masks.pt'),
            'code_sha256':sha256(Path(__file__)),'source_features_sha256':sha256(base/'features.npy'),
            'crop_features_sha256':sha256(lumbar/'crop_features.npy'),'extra_features_sha256':sha256(output/'photometric_features.npy'),
            'limitations':['Previously inspected source holdout, not an independent clinical test.',
                           'Only two DXA patients held out; GE has no reference masks.',
                           'Gamma and inverse transforms preserve target positions but do not simulate scanner physics.']}
    (output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','lumbar','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.base,a.lumbar,a.output)
