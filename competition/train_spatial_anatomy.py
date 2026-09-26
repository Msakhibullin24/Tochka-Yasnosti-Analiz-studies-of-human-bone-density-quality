"""Learn numbered vertebral heatmaps from frozen spatial features, not pooled vectors.

BUU patient split is fixed before training; held-out targets are used only after
the fixed epoch count. No masks, disc boundaries or clinical DXA validity claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dxaqc.embedding import _model, _MEAN, _STD, WEIGHTS_SHA256
from evaluate_buu_transfer import references
from evaluate_organizer_dataset import sha256
from train_buu_anatomy import numbered_metrics
from train_buu_projection import patient_split

SIZE, GRID, CHANNELS = 320, 40, 896


def spatial_embedding(pixels):
    image=cv2.resize(pixels,(SIZE,SIZE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    x=torch.from_numpy(((np.repeat(image[None],3,axis=0)-_MEAN)/_STD)[None])
    backbone=_model()
    with torch.inference_mode():
        x=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        x=backbone.layer1(x); maps=[]
        for layer in (backbone.layer2,backbone.layer3,backbone.layer4):
            x=layer(x);maps.append(F.interpolate(x,size=(GRID,GRID),mode='bilinear',align_corners=False))
    return torch.cat(maps,dim=1)[0].numpy().astype(np.float16)


def heatmap_targets(points, grid=GRID):
    if not isinstance(grid, int) or grid < 1:
        raise ValueError('Invalid heatmap grid')
    points=np.asarray(points,dtype=float)
    if points.shape!=(5,2) or not np.isfinite(points).all() or (points<0).any() or (points>1).any():
        raise ValueError('Invalid normalized numbered centers')
    yy,xx=np.mgrid[:grid,:grid]
    means=points*grid-.5
    values=np.exp(-((xx[None]-means[:,0,None,None])**2+(yy[None]-means[:,1,None,None])**2)/2)
    return (values/values.sum(axis=(1,2),keepdims=True)).astype(np.float32)


def decode(logits):
    values=np.asarray(logits)
    if values.ndim!=4 or values.shape[1]!=5 or min(values.shape[2:])<1 or not np.isfinite(values).all():
        raise ValueError('Invalid heatmaps')
    n,_,h,w=values.shape;result=np.zeros((n,5,2))
    for i in range(n):
        for level in range(5):
            cy,cx=np.unravel_index(values[i,level].argmax(),(h,w))
            y0,y1=max(0,cy-1),min(h,cy+2);x0,x1=max(0,cx-1),min(w,cx+2)
            patch=values[i,level,y0:y1,x0:x1];weights=np.exp(patch-patch.max());weights/=weights.sum()
            yy,xx=np.mgrid[y0:y1,x0:x1]
            result[i,level]=[(float((weights*xx).sum())+.5)/w,(float((weights*yy).sum())+.5)/h]
    return result


def head():
    return nn.Sequential(nn.Conv2d(CHANNELS,32,1),nn.ReLU(),
                         nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,5,1))


def run(args):
    torch.set_num_threads(4);torch.manual_seed(17)
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    inventory=json.loads(Path('competition/reports/buu_frozen_transfer_2026_09_26.json').read_text())
    records=[r for r in inventory['cases'] if r['lateral_reference']==0]
    _,held_out,test=patient_split(records)
    args.output.mkdir(parents=True)
    cache=np.lib.format.open_memmap(args.output/'spatial_features.npy',mode='w+',dtype=np.float16,
                                   shape=(len(records),CHANNELS,GRID,GRID))
    points,heights,sizes,targets,source_meta=[],[],[],[],[]
    for i,r in enumerate(records):
        p=args.root/r['file']
        if sha256(p)!=r['source_sha256']:
            raise ValueError('Source changed since inventory')
        pixels=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);h,w=pixels.shape
        centers,body_heights=references(p.with_suffix('.csv'),w,h)
        normalized=centers/np.array([w,h])
        cache[i]=spatial_embedding(pixels)
        points.append(centers/h);heights.append(body_heights/h);sizes.append([w,h])
        targets.append(heatmap_targets(normalized));source_meta.append({**r,'annotation_sha256':sha256(p.with_suffix('.csv'))})
        if (i+1)%50==0:
            print(f'Spatial features {i+1}/{len(records)}',flush=True)
    cache.flush()
    points,heights,sizes,targets=map(np.asarray,(points,heights,sizes,targets))
    training=np.flatnonzero(~test);testing=np.flatnonzero(test)
    model=head();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    generator=np.random.default_rng(17)
    for epoch in range(args.epochs):
        model.train();losses=[]
        order=generator.permutation(training)
        for start in range(0,len(order),8):
            index=order[start:start+8]
            x=torch.from_numpy(np.array(cache[index],dtype=np.float32))
            target=torch.from_numpy(targets[index]).flatten(2)
            logits=model(x).flatten(2)
            loss=-(target*F.log_softmax(logits,dim=2)).sum(dim=2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        print(f'Epoch {epoch+1}/{args.epochs}, training loss {np.mean(losses):.4f}',flush=True)
    model.eval();predictions=[]
    with torch.inference_mode():
        for start in range(0,len(testing),8):
            index=testing[start:start+8]
            predictions.extend(decode(model(torch.from_numpy(np.array(cache[index],dtype=np.float32))).numpy()))
    predictions=np.asarray(predictions);predictions[:,:,0]*=sizes[test,0,None]/sizes[test,1,None]
    prior=np.broadcast_to(points[~test].mean(axis=0),points[test].shape)
    torch.save({'state_dict':model.state_dict(),'levels':['L1','L2','L3','L4','L5'],
                'encoder_sha256':WEIGHTS_SHA256,'clinical_validation':False,
                'input_size':SIZE,'heatmap_grid':GRID},args.output/'spatial_head.pt')
    report={'protocol':f'Fixed BUU patient split, 300 train/100 test; {args.epochs} fixed epochs; frozen ResNet spatial layer2-4; held-out targets never used for training/early stopping',
            'clinical_validation':False,'release_modified':False,'training_patients':int((~test).sum()),
            'held_out_patients':int(test.sum()),'learned':numbered_metrics(predictions,points[test],heights[test]),
            'layout_prior':numbered_metrics(prior,points[test],heights[test]),
            'code_sha256':sha256(Path(__file__)),'model_sha256':sha256(args.output/'spatial_head.pt'),
            'encoder_sha256':WEIGHTS_SHA256,'cases':[{'patient_id':r['patient_id'],'source_sha256':r['source_sha256'],
              'annotation_sha256':r['annotation_sha256'],'reference':a.tolist(),'prediction':p.tolist(),'body_heights':h.tolist()}
              for r,a,p,h in zip([r for r,t in zip(source_meta,test) if t],points[test],predictions,heights[test])],
            'limitations':['Post-selection research on a previously examined source.',
                           'Plain radiographs, not DXA. No Th12, iliac crest, disc or ROI masks.',
                           'Five fixed level heatmaps cannot prove visibility or transitional anatomy.']}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('learned','layout_prior')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args()
    if not 1<=a.epochs<=100:p.error('epochs must be between 1 and 100')
    run(a)
