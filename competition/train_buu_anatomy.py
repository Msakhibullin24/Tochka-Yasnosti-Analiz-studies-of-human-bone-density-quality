"""Supervised numbered center regression on BUU AP, with a layout-prior baseline.

This tests only labeled centers L1-L5 on plain AP radiographs. It does not
establish visibility of Th12/iliac crests, disc contours or clinical DXA ROIs.
"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from evaluate_buu_transfer import references
from evaluate_organizer_dataset import sha256


def spatial_features(pixels):
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError('Invalid grayscale raster')
    image = cv2.resize(pixels,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    image = (image-image.mean())/max(float(image.std()),1.)
    return image.ravel()


def numbered_metrics(predicted, actual, heights):
    predicted, actual, heights = map(np.asarray,(predicted,actual,heights))
    if (predicted.shape != actual.shape or actual.ndim!=3 or actual.shape[1:]!=(5,2)
            or heights.shape!=actual.shape[:2] or (heights<=0).any()
            or not all(np.isfinite(a).all() for a in (predicted,actual,heights))):
        raise ValueError('Invalid numbered localization arrays')
    error = np.linalg.norm(predicted-actual,axis=-1)/heights
    return {'bodies':int(error.size),'within_half_body_height':int((error<=.5).sum()),
            'numbered_center_hit_rate':float((error<=.5).mean()),
            'median_error_body_heights_all_bodies':float(np.median(error)),
            'p95_error_body_heights_all_bodies':float(np.percentile(error,95))}


def run(root, projection_run, output):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    cohort=json.loads((projection_run/'evaluation.json').read_text())
    with np.load(projection_run/'features.npz',allow_pickle=False) as cached:
        embeddings=cached['features']
    if sha256(projection_run/'features.npz')!=cohort['feature_sha256']:
        raise ValueError('Feature cache checksum differs')
    cases=[(i,r) for i,r in enumerate(cohort['cases']) if r['source']=='buu' and r['target']==0]
    paths={sha256(p):p for p in root.rglob('*.jpg') if p.stem.endswith('0')}
    X,y,heights,metadata=[],[],[],[]
    for index, record in cases:
        path=paths[record['source_sha256']]
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        h,w=image.shape
        centers,body_heights=references(path.with_suffix('.csv'),w,h)
        # x and y use the same height unit, so Euclidean errors stay isotropic.
        X.append(np.concatenate([embeddings[index].mean(axis=0),spatial_features(image)]))
        y.append(centers/h);heights.append(body_heights/h)
        metadata.append({'source_sha256':record['source_sha256'],
                         'annotation_sha256':sha256(path.with_suffix('.csv')),
                         'patient_group':record['group'],'width_height':[w,h]})
    if len({r['patient_group'] for r in metadata})!=len(metadata):
        raise ValueError('Repeated patient: group-aware anatomy split required')
    X,y,heights=map(np.asarray,(X,y,heights))
    predictions,prior=np.zeros_like(y),np.zeros_like(y)
    folds=np.full(len(y),-1)
    make_model=lambda:ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,
                                         max_features=.5,random_state=17,n_jobs=4)
    for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=17).split(X)):
        model=make_model().fit(X[train],y[train].reshape(-1,10))
        predictions[test]=model.predict(X[test]).reshape(-1,5,2)
        prior[test]=y[train].mean(axis=0)
        folds[test]=fold
    model=make_model().fit(X,y.reshape(-1,10))
    output.mkdir(parents=True)
    joblib.dump({'model':model,'status':'research_only','levels':['L1','L2','L3','L4','L5'],
                 'clinical_validation':False,'training_domain':'BUU plain AP radiograph',
                 'coordinate_system':'xy_in_image_height_units'},output/'body_centers.joblib')
    report={'protocol':'Five patient-disjoint folds; fixed ExtraTrees settings; same folds for training-only mean layout prior',
            'patients':len(y),'clinical_validation':False,'release_modified':False,
            'learned':numbered_metrics(predictions,y,heights),
            'layout_prior':numbered_metrics(prior,y,heights),
            'model_sha256':sha256(output/'body_centers.joblib'),'code_sha256':sha256(Path(__file__)),
            'source_feature_sha256':cohort['feature_sha256'],
            'cases':[{**r,'fold':int(f),'reference':a.tolist(),'prediction':p.tolist(),
                      'layout_prior':b.tolist(),'body_heights':s.tolist()}
                     for r,f,a,p,b,s in zip(metadata,folds,y,predictions,prior,heights)],
            'limitations':['Center regression is not segmentation or disc boundary detection.',
                           'Absolute labels follow BUU annotation order, not a verified DXA numbering model.',
                           'Only same-source AP radiographs evaluated; no DXA transfer claims.',
                           'Five-center outputs cannot establish missing/extra vertebrae or transitional anatomy.']}
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('patients','learned','layout_prior')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','projection_run','output'):
        p.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    a=p.parse_args();run(a.root,a.projection_run,a.output)
