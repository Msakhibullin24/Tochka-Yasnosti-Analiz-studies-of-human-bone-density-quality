"""Source-only partial-body visibility regression with grouped calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.embedding import embed,WEIGHTS_SHA256

FEATURE_CONTRACT='resnet18_square320_average_original_inverse_v1'


def features(pixels):
    square=cv2.resize(pixels,(320,320),interpolation=cv2.INTER_AREA)
    return np.stack([embed(square),embed(255-square)]).mean(axis=0)


def calibration_radius(errors,groups,alpha=.1):
    if not 0<alpha<1 or len(errors)!=len(groups) or not len(errors):raise ValueError('Invalid calibration cohort')
    if not np.isfinite(errors).all() or (np.asarray(errors)<0).any():raise ValueError('Invalid calibration residuals')
    maxima=np.array([np.max(np.asarray(errors)[np.asarray(groups)==group]) for group in sorted(set(groups))])
    rank=int(np.ceil((len(maxima)+1)*(1-alpha)))
    if rank>len(maxima):raise ValueError('Too few calibration parent groups for requested coverage')
    return float(np.sort(maxima)[rank-1])


def assess(bundle,pixels):
    if (bundle.get('schema_version')!=2 or bundle.get('scope')!='th12_partial_visibility' or
        bundle.get('encoder_sha256')!=WEIGHTS_SHA256 or bundle.get('feature_contract')!=FEATURE_CONTRACT or
        bundle.get('clinical_validation') is not False):raise ValueError('Incompatible visibility candidate')
    raw=float(bundle['model'].predict(features(pixels)[None])[0])
    if not np.isfinite(raw):raise ValueError('Invalid visibility prediction')
    value=float(np.clip(raw,0,1))
    radius=bundle['calibration_radius']
    if not np.isfinite(value) or not np.isfinite(radius) or radius<0:raise ValueError('Invalid visibility prediction')
    low,high=max(0.,value-radius),min(1.,value+radius)
    verdict='candidate_at_least_half_height' if low>=.5 else 'candidate_less_than_half_height' if high<.5 else 'undetermined'
    return {'predicted_vertical_fraction':value,'source_calibrated_interval':[low,high],
            'candidate_rule':verdict,'clinical_verdict':'undetermined','clinical_validation':False,
            'scope':'Derived ordinary-XR crop visibility; interval coverage on real DXA is not established'}


def run(args):
    if args.output.exists():raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    root=args.dataset.resolve();manifest=root/'dataset.json';data=json.loads(manifest.read_text());rows=data['cases']
    if data.get('version')!=1 or len(rows)!=data['derived_images']:raise ValueError('Unexpected crop dataset')
    train_groups=sorted({r['parent_group'] for r in rows if r['split']=='train'})
    test_groups={r['parent_group'] for r in rows if r['split']=='test'}
    if set(train_groups)&test_groups:raise ValueError('Parent split leakage')
    cal_groups=set(np.random.default_rng(17).choice(train_groups,len(train_groups)//5,replace=False))
    splits=['test' if r['split']=='test' else 'calibration' if r['parent_group'] in cal_groups else 'train' for r in rows]
    # Independent parent groups stay together for all variants; test is not used to calibrate.
    x=[];y=[]
    for i,row in enumerate(rows):
        path=(root/row['image']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError('Crop source path or hash mismatch')
        pixels=cv2.imread(str(path),0)
        if pixels is None or pixels.shape!=(row['height'],row['width']):raise ValueError('Crop raster mismatch')
        x.append(features(pixels));y.append(row['reference']['visible_vertical_extent_fraction'])
        if (i+1)%100==0:print(f'Encoded visibility views {i+1}/{len(rows)}',flush=True)
    x=np.asarray(x);y=np.asarray(y);splits=np.array(splits)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or ((y<0)|(y>1)).any():raise ValueError('Invalid training arrays')
    model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
    with threadpool_limits(limits=4):
        model.fit(x[splits=='train'],y[splits=='train'])
        raw_scores=model.predict(x)
        if not np.isfinite(raw_scores).all():raise ValueError('Invalid visibility prediction')
        scores=np.clip(raw_scores,0,1)
    cal=splits=='calibration';test=splits=='test';train=splits=='train'
    radius=calibration_radius(abs(scores[cal]-y[cal]),[row['parent_group'] for row,s in zip(rows,splits) if s=='calibration'])
    prior=float(y[train].mean());cases=[]
    for i in np.flatnonzero(test):
        low,high=max(0.,float(scores[i]-radius)),min(1.,float(scores[i]+radius))
        cases.append({'image':rows[i]['image'],'parent_group':rows[i]['parent_group'],'reference':float(y[i]),
                      'prediction':float(scores[i]),'interval':[low,high],
                      'candidate_rule':'candidate_at_least_half_height' if low>=.5 else 'candidate_less_than_half_height' if high<.5 else 'undetermined'})
    args.output.mkdir(parents=True)
    bundle={'schema_version':2,'scope':'th12_partial_visibility','feature_contract':FEATURE_CONTRACT,
            'encoder_sha256':WEIGHTS_SHA256,'model':model,'calibration_radius':radius,
            'clinical_validation':False,'status':'research_only'}
    joblib.dump(bundle,args.output/'candidate.joblib');np.save(args.output/'features.npy',x)
    parent_covered=[all(c['interval'][0]<=c['reference']<=c['interval'][1] for c in cases if c['parent_group']==parent) for parent in sorted(test_groups)]
    report={'protocol':'Fixed Ridge alpha100; source parent train/calibration/test; 90% grouped-max residual interval, calibration parents selected seed17 before fitting.',
            'training_views':int(train.sum()),'calibration_views':int(cal.sum()),'test_views':int(test.sum()),
            'training_parents':len(train_groups)-len(cal_groups),'calibration_parents':len(cal_groups),'test_parents':len(test_groups),
            'mae':float(np.mean(abs(scores[test]-y[test]))),'constant_prior_mae':float(np.mean(abs(prior-y[test]))),
            'calibration_radius':radius,'test_interval_coverage':float(np.mean([c['interval'][0]<=c['reference']<=c['interval'][1] for c in cases])),
            'test_parent_all_views_coverage':float(np.mean(parent_covered)),
            'determined_test_views':sum(c['candidate_rule']!='undetermined' for c in cases),'cases':cases,
            'model_sha256':hashlib.sha256((args.output/'candidate.joblib').read_bytes()).hexdigest(),
            'dataset_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'clinical_validation':False,'requirements_complete':False,'release_modified':False,
            'limitations':['Derived ordinary-XR crops, not real DXA visibility labels.',
                           'Quadrilateral height is a projected proxy; no equivalence to medical full-body area is asserted.',
                           'Image parents are not confirmed patient identities; exchangeability with DXA is not established.',
                           'Source holdout was previously inspected; exploratory followup, not a new independent cohort.',
                           'Calibration targets do not appear in input features, but reference-defined field generation can introduce shortcuts.',
                           'Intervals are source-calibrated only; their coverage on DXA or other domains is unknown.']}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('mae','constant_prior_mae','calibration_radius','test_interval_coverage','determined_test_views')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
