"""Close the field-of-view gap with annotated lumbar crops; preserve source splits."""
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
from train_anatomy_masks import metrics


def lumbar_crop(pixels, target):
    ys, xs = np.where(target >= 12)
    if not len(xs):
        raise ValueError('No annotated thoracolumbar bodies')
    h, w = pixels.shape
    x0, x1 = max(0, int(xs.min())-w//8), min(w, int(xs.max())+w//8+1)
    y0, y1 = max(0, int(ys.min())-h//80), min(h, int(ys.max())+h//40+1)
    return pixels[y0:y1, x0:x1], target[y0:y1, x0:x1]


def run(base, output):
    if output.exists():
        raise ValueError('Choose a new refinement directory')
    torch.set_num_threads(4); torch.manual_seed(17)
    inventory = json.loads((base/'inventory.json').read_text())
    rows = inventory['rows']
    original = np.load(base/'features.npy', mmap_mode='r', allow_pickle=False)
    if original.shape != (len(rows), 896, 40, 40) or original.dtype != np.float16:
        raise ValueError('Incompatible local feature cache')
    original_targets = []
    cropped = []
    cropped_targets = []
    parents = []
    held_crops = []
    for i, row in enumerate(rows):
        pixels, target = load_record(row)
        original_targets.append(cv2.resize(target.astype(np.float32), (40,40), interpolation=cv2.INTER_NEAREST).astype(np.int64))
        if row['source'] != 'aasce':
            continue
        crop, annotation = lumbar_crop(pixels, target)
        if row['split'] == 'test':
            held_crops.append((i, crop, annotation))
        else:
            for variant in (crop, 255-crop):
                cropped.append(variant)
                cropped_targets.append(cv2.resize(annotation.astype(np.float32), (40,40), interpolation=cv2.INTER_NEAREST).astype(np.int64))
                parents.append(i)
    output.mkdir(parents=True)
    augmented = np.lib.format.open_memmap(output/'crop_features.npy', mode='w+', dtype=np.float16,
                                         shape=(len(cropped), 896, 40, 40))
    for i, raster in enumerate(cropped):
        augmented[i] = spatial_features(raster)
        if (i+1)%100 == 0:
            print(f'Encoded lumbar crop {i+1}/{len(cropped)}', flush=True)
    augmented.flush()
    targets = np.concatenate([original_targets, cropped_targets])
    training = np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    dxa = np.array([i for i in training if rows[i]['source']=='ramathibodi'])
    schedule = np.r_[training, np.arange(len(rows), len(targets)), np.tile(dxa, 24)]
    if any(rows[p]['split'] != 'train' for p in parents):
        raise ValueError('Crop from held-out parent in training')
    bundle = torch.load(base/'anatomy_masks.pt', weights_only=True)
    model = mask_head(); model.load_state_dict(bundle['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    rng = np.random.default_rng(17)
    weights = torch.tensor([.2]+[1.]*17)
    for epoch in range(10):
        model.train(); losses=[]
        order=rng.permutation(schedule)
        for start in range(0,len(order),8):
            ids=order[start:start+8]
            arrays=[original[i] if i<len(rows) else augmented[i-len(rows)] for i in ids]
            logits=model(torch.from_numpy(np.array(arrays,dtype=np.float32)))
            loss=F.cross_entropy(logits,torch.from_numpy(targets[ids]),weight=weights,ignore_index=-100)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        print(f'Lumbar refinement {epoch+1}/10: {np.mean(losses):.4f}',flush=True)
    model.eval(); cases=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            if row['split']!='test':continue
            prediction=model(torch.from_numpy(np.array(original[i],dtype=np.float32))[None])[0].argmax(0).numpy()
            cases.append({'source':row['source'],'group':row['group'],'view':'original','masks':metrics(prediction,targets[i])})
        for i,crop,annotation in held_crops:
            prediction=model(torch.from_numpy(spatial_features(crop).astype(np.float32))[None])[0].argmax(0).numpy()
            target=cv2.resize(annotation.astype(np.float32),(40,40),interpolation=cv2.INTER_NEAREST).astype(np.int64)
            cases.append({'source':'aasce','group':rows[i]['group'],'view':'lumbar_crop','masks':metrics(prediction,target)})
    bundle.update(state_dict=model.state_dict(),field_of_view_adaptation='annotated lumbar crops, original/inverted',clinical_validation=False)
    torch.save(bundle,output/'anatomy_masks.pt')
    summary={}
    for source,view in [('aasce','original'),('aasce','lumbar_crop'),('ramathibodi','original')]:
        selected=[c for c in cases if c['source']==source and c['view']==view]
        summary[source+'_'+view]={'dice':float(np.mean([m['dice'] for c in selected for m in c['masks']])), 'images':len(selected)}
    report={'protocol':'Fixed original splits; train-parent-only lumbar crops and polarity variants; base head plus fixed 10 epochs',
            'summary':summary,'cases':cases,'crop_parents':[rows[i]['group'] for i in parents],
            'base_model_sha256':sha256(base/'anatomy_masks.pt'),'base_features_sha256':sha256(base/'features.npy'),
            'model_sha256':sha256(output/'anatomy_masks.pt'),'code_sha256':sha256(Path(__file__)),
            'clinical_validation':False,'limitations':['Previously inspected source holdout, not a new independent test.',
                'GE mask accuracy cannot be measured without reference masks; coverage is not correctness.',
                'AASCE labels follow its 17-level dataset convention; transitional anatomy is not established.',
                'DXA masks are vendor ROI and have only two held-out patients.']}
    (output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.output)
