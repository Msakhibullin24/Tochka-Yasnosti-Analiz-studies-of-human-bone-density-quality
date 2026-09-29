"""Frozen classifier test on a separate publisher-labelled EOS lateral cohort."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.embedding import embed, WEIGHTS_SHA256


def run(args):
    if args.output.exists():raise ValueError('Choose a new report')
    torch.set_num_threads(4)
    bundle=joblib.load(args.model)
    if (bundle.get('schema_version')!=2 or bundle.get('status')!='research_only' or
        bundle.get('scope')!='hip_frontal_vs_nonfrontal' or bundle.get('encoder_sha256')!=WEIGHTS_SHA256 or
        bundle.get('clinical_validation') is not False):
        raise ValueError('Unexpected frozen hip candidate')
    training=json.loads(args.training_report.read_text())
    model_hash=hashlib.sha256(args.model.read_bytes()).hexdigest()
    if training.get('model_sha256')!=model_hash:raise ValueError('Training report and model differ')
    known_sources={c['source_sha256'] for c in training['cases']}
    known_pixels={c['pixel_sha256'] for c in training['cases']}
    inventory_path=args.dataset/'inventory.json'
    source=json.loads(inventory_path.read_text())
    if source['doi']!='10.6084/m9.figshare.23820879.v1':raise ValueError('Unexpected test source')
    cases=[];root=args.dataset.resolve()
    for row in source['records']:
        path=(root/row['image']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['source_sha256']:
            raise ValueError('Test source path/hash mismatch')
        if row['source_sha256'] in known_sources:raise ValueError('External test source overlaps previous cohort')
        case={**row,'label':1}
        try:
            with Image.open(path) as image:image.load();pixels=np.array(image.convert('L'))
            pixel_hash=hashlib.sha256(pixels.tobytes()+str(pixels.shape).encode()).hexdigest()
            if pixel_hash in known_pixels:raise ValueError('External test pixels overlap previous cohort')
            case['pixel_sha256']=pixel_hash
            square=cv2.resize(pixels,(320,320),interpolation=cv2.INTER_AREA)
            score=float(bundle['model'].predict_proba(np.stack([embed(square),embed(255-square)]))[:,1].mean())
            if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid model score')
            case.update(nonfrontal_score=score,value='frontal' if score<=.1 else 'nonfrontal' if score>=.9 else 'unknown')
        except (OSError,Image.DecompressionBombError) as exc:
            case.update(value='unavailable',reason=str(exc),nonfrontal_score=None)
        cases.append(case)
        if len(cases)%25==0:print(f'Evaluated independent lateral views {len(cases)}/{source["images"]}',flush=True)
    positive=sum(c['nonfrontal_score'] is not None and c['nonfrontal_score']>=.5 for c in cases)
    report={'protocol':'Frozen FracAtlas candidate, no fitting or threshold adjustment; first independent EOS cohort assessment.',
            'images':len(cases),'patients':source['patients'],'prediction_counts':dict(Counter(c['value'] for c in cases)),
            'detected_at_fixed_point5':positive,'sensitivity_including_unavailable':positive/len(cases),
            'mean_patient_detection_fraction':float(np.mean([
                np.mean([c['nonfrontal_score'] is not None and c['nonfrontal_score']>=.5 for c in cases if c['patient_group']==patient])
                for patient in {c['patient_group'] for c in cases}])),
            'specificity':None,'auc':None,'cases':cases,'cross_source_exact_overlap':0,
            'training_report_sha256':hashlib.sha256(args.training_report.read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'source_inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'clinical_validation':False,'release_modified':False,'requirements_complete':False,
            'limitations':['Positive lateral EOS cohort only: specificity and AUC are undefined.',
                           '115 images represent 93 patients; repeated patient images are correlated.',
                           'Whole lateral pelvis and cropped lateral hip DXA are different acquisition domains.',
                           'No anatomically labelled DXA nonfrontal reference is supplied.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('images','patients','prediction_counts','sensitivity_including_unavailable')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','dataset','training-report','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
