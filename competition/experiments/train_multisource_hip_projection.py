"""Exploratory FracAtlas + EOS followup; retain the previous FracAtlas holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix,roc_auc_score
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.embedding import embed,WEIGHTS_SHA256


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(cases):
    y=np.array([c['label'] for c in cases]);s=np.array([c['nonfrontal_score'] for c in cases])
    return {'images':len(cases),'confusion_matrix_0_frontal_1_nonfrontal':confusion_matrix(y,s>=.5,labels=[0,1]).tolist(),
            'auc':float(roc_auc_score(y,s)) if len(set(y))==2 else None,
            'nonfrontal_called_confident_frontal':int(((y==1)&(s<=.1)).sum()),
            'frontal_called_confident_nonfrontal':int(((y==0)&(s>=.9)).sum()),
            'uncertain':int(((s>.1)&(s<.9)).sum())}


def run(args):
    if args.output.exists():raise ValueError('Choose a new training directory')
    torch.set_num_threads(4)
    previous=json.loads((args.fracatlas/'evaluation.json').read_text())
    bundle=joblib.load(args.fracatlas/'candidate.joblib')
    if digest(args.fracatlas/'candidate.joblib')!=previous['model_sha256'] or bundle.get('scope')!='hip_frontal_vs_nonfrontal':
        raise ValueError('Previous candidate identity mismatch')
    cases=[{**c,'source':'fracatlas'} for c in previous['cases']]
    frac_features=np.load(args.fracatlas/'features.npy')
    if frac_features.shape!=(len(cases),2,1792) or not np.isfinite(frac_features).all():
        raise ValueError('Invalid earlier feature cache')
    inventory=json.loads((args.eos/'inventory.json').read_text())
    if inventory['doi']!='10.6084/m9.figshare.23820879.v1':raise ValueError('Unexpected EOS source')
    patients=sorted({r['patient_group'] for r in inventory['records']})
    held=set(np.random.default_rng(17).choice(patients,len(patients)//5,replace=False))
    pixels_seen={c['pixel_sha256']:('fracatlas',c['duplicate_group'],c['split']) for c in cases}
    source_seen={c['source_sha256'] for c in cases}
    eos_features=[];root=args.eos.resolve()
    for row in inventory['records']:
        path=(root/row['image']).resolve()
        if not path.is_relative_to(root) or digest(path)!=row['source_sha256'] or row['source_sha256'] in source_seen:
            raise ValueError('EOS source path, SHA or cross-source identity mismatch')
        with Image.open(path) as image:image.load();pixels=np.array(image.convert('L'))
        pixel_hash=hashlib.sha256(pixels.tobytes()+str(pixels.shape).encode()).hexdigest()
        split='test' if row['patient_group'] in held else 'train'
        group=('eos',row['patient_group'],split)
        if pixel_hash in pixels_seen and pixels_seen[pixel_hash]!=group:
            raise ValueError('Pixel overlap across different patient/source groups; resolve grouping before training')
        pixels_seen[pixel_hash]=group;source_seen.add(row['source_sha256'])
        square=cv2.resize(pixels,(320,320),interpolation=cv2.INTER_AREA)
        eos_features.append(np.stack([embed(square),embed(255-square)]))
        cases.append({**row,'source':'eos','label':1,'split':split,'pixel_sha256':pixel_hash})
        if len(eos_features)%25==0:print(f'Encoded EOS views {len(eos_features)}/{len(inventory["records"])}',flush=True)
    x=np.concatenate([frac_features,np.array(eos_features)])
    y=np.array([c['label'] for c in cases]);train=np.array([c['split']=='train' for c in cases]);test=~train
    # Identical classifier and C to the frozen source-only candidate, no new tuning.
    from sklearn.base import clone
    model=clone(bundle['model'])
    model.fit(x[train].reshape(-1,1792),np.repeat(y[train],2))
    scores=model.predict_proba(x[test].reshape(-1,1792))[:,1].reshape(-1,2).mean(axis=1)
    evaluated=[]
    for case,score in zip([c for c in cases if c['split']=='test'],scores):
        evaluated.append({**case,'previous_nonfrontal_score':case.get('nonfrontal_score'),
                          'nonfrontal_score':float(score)})
    args.output.mkdir(parents=True)
    output_bundle={**bundle,'model':model,'training_sources':['FracAtlas v7','EOS pelvic tilt v1']}
    joblib.dump(output_bundle,args.output/'candidate.joblib')
    np.save(args.output/'features.npy',x)
    report={'protocol':'Exploratory followup; frozen classifier/C; previous FracAtlas split retained; EOS patient holdout first 20% seeded choice17.',
            'training_images':int(train.sum()),'held_out_images':int(test.sum()),'eos_held_out_patients':len(held),
            'summary':{source:summary([c for c in evaluated if c['source']==source]) for source in ('fracatlas','eos')},
            'cases':evaluated,'model_sha256':digest(args.output/'candidate.joblib'),
            'input_cache_sha256':digest(args.fracatlas/'features.npy'),'features_sha256':digest(args.output/'features.npy'),
            'previous_report_sha256':digest(args.fracatlas/'evaluation.json'),
            'eos_inventory_sha256':digest(args.eos/'inventory.json'),'code_sha256':digest(Path(__file__)),
            'clinical_validation':False,'release_modified':False,'requirements_complete':False,
            'limitations':['Followup on previously examined cohorts, not fresh independent validation.',
                           'Frontal author tags do not distinguish AP/PA; FracAtlas patient identities are unavailable.',
                           'All EOS examples are lateral: scanner/site/domain shortcuts remain possible.',
                           'Three FracAtlas nonfrontal holdout images cannot establish reliable sensitivity.',
                           'No lateral/oblique DXA expert reference or clinical rotation-subtype labels.']}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fracatlas','eos','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
