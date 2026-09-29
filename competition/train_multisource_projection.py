"""Group-held-out projection candidate across radiographs and two DXA sources.

Fixed logistic model, equal source mass in training, no threshold tuning. GE AP
labels come from the organizer protocol and are explicitly weak references.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import cv2
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import WEIGHTS_SHA256
from evaluate_organizer_dataset import binary_metrics, sha256
from evaluate_projection_candidate import assessment, classifier, features, source_patient_group


def source_weights(sources):
    sources = np.asarray(sources)
    kinds, counts = np.unique(sources, return_counts=True)
    if not len(sources):
        raise ValueError('Empty training cohort')
    weights = dict(zip(kinds, len(sources)/(len(kinds)*counts)))
    return np.array([weights[s] for s in sources])


def grouped_folds(records):
    """Split each domain by source identity groups; all domains occur in each fold."""
    folds = np.full(len(records), -1)
    sources = np.array([r['source'] for r in records])
    groups = np.array([r['group'] for r in records])
    y = np.array([r['target'] for r in records])
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        if len(set(groups[indices])) < 5:
            raise ValueError('Five identity groups per source required')
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=17)
        for fold, (_, test) in enumerate(splitter.split(indices, y[indices], groups[indices])):
            folds[indices[test]] = fold
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Identity group leakage across sources or folds')
        if ({r['pixel_sha256'] for r,t in zip(records,train) if t}
                & {r['pixel_sha256'] for r,t in zip(records,test) if t}):
            raise ValueError('Pixel leakage across folds')
    return folds


def collect(buu, ramathibodi, organizer, labels):
    records, rasters, seen = [], [], {}
    def add(pixels, pixel_hash, source_hash, source, group, target, basis):
        if pixel_hash in seen:
            previous = seen[pixel_hash]
            if previous['target'] != target:
                raise ValueError('Contradictory projection labels for duplicate pixels')
            if previous['group'] != group:
                # Ambiguous shared identity is not silently split or relabelled.
                raise ValueError('Duplicate pixels have different patient groups')
            return
        record = {'pixel_sha256':pixel_hash, 'source_sha256':source_hash,
                  'source':source, 'group':group, 'target':target, 'reference_basis':basis}
        seen[pixel_hash] = record
        records.append(record); rasters.append(pixels)
    import hashlib
    for path in sorted(buu.rglob('*.jpg')):
        pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if pixels is None or path.stem[-1] not in ('0','1'):
            raise ValueError('Invalid BUU raster')
        add(pixels,hashlib.sha256(pixels.tobytes()).hexdigest(),sha256(path),
            'buu','buu:'+path.stem[:4],int(path.stem[-1]),'source_AP_LA_view')
    for path in sorted((ramathibodi/'BMD').rglob('spine_image.dcm'))+sorted((ramathibodi/'VFA').rglob('*.dcm')):
        relative = path.relative_to(ramathibodi)
        image = read_dxa(path)
        add(image.pixels,image.pixel_sha256,sha256(path),'ramathibodi',
            'ramathibodi:'+source_patient_group(relative),
            int(relative.parts[0]=='VFA'),'source_BMD_VFA_branch')
    with labels.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['region']!='spine' or row['quality_class'] not in ('0','1'):
                continue
            path = (organizer/row['first_source_path']).resolve()
            if not path.is_relative_to(organizer.resolve()) or not row['study_key']:
                raise ValueError('Invalid organizer source identity')
            image = read_dxa(path)
            add(image.pixels,image.pixel_sha256,sha256(path),'organizer',
                'organizer:'+row['study_key'],0,'weak_AP_protocol_not_physician_review')
    return records, rasters


def evaluate(records, scores):
    result = {}
    for source in sorted({r['source'] for r in records}):
        index = np.array([r['source']==source for r in records])
        y = np.array([r['target'] for r in records])[index]
        s = scores[index]
        accepted = (s <= .1) | (s >= .9)
        result[source] = {'metrics':binary_metrics(y,(s >= .5).astype(int),s),
                          'undetermined':int((~accepted).sum()),
                          'accepted_errors':int(((s >= .5) != y)[accepted].sum())}
    return result


def run(args):
    import torch
    torch.set_num_threads(4)
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    records, rasters = collect(args.buu,args.ramathibodi,args.organizer,args.labels)
    folds = grouped_folds(records)
    vectors = []
    for i, image in enumerate(rasters):
        vectors.append(features(image))
        if (i+1) % 100 == 0:
            print(f'Embedded {i+1}/{len(records)}',flush=True)
    X = np.asarray(vectors)
    y = np.array([r['target'] for r in records])
    sources = np.array([r['source'] for r in records])
    scores = np.full(len(records),np.nan)
    for fold in range(5):
        train, test = folds!=fold, folds==fold
        model = classifier().fit(X[train].reshape(-1,X.shape[-1]),np.repeat(y[train],2),
                                logisticregression__sample_weight=np.repeat(source_weights(sources[train]),2))
        scores[test] = model.predict_proba(X[test].reshape(-1,X.shape[-1]))[:,1].reshape(-1,2).mean(axis=1)
    if not np.isfinite(scores).all():
        raise ValueError('Incomplete held-out scores')
    model = classifier().fit(X.reshape(-1,X.shape[-1]),np.repeat(y,2),
                            logisticregression__sample_weight=np.repeat(source_weights(sources),2))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'features.npz',features=X)
    joblib.dump({'schema_version':1,'status':'research_only','encoder_sha256':WEIGHTS_SHA256,
                 'model':model,'threshold':.5,'classes':['frontal','lateral'],
                 'clinical_validation':False,'method':'multisource_frozen_resnet18_lr'},args.output/'projection_multisource.joblib')
    cases = [{**r,'fold':int(f),'lateral_score':float(s),
              'value':assessment(float(s))['value']} for r,f,s in zip(records,folds,scores)]
    report = {'protocol':'Five folds by source identity group; DXA public numeric folder index is a conservative patient proxy; fixed C=.01, fixed .5 threshold; equal source weight; no test tuning',
              'clinical_validation':False,'release_modified':False,
              'encoder_sha256':WEIGHTS_SHA256,'code_sha256':sha256(Path(__file__)),
              'model_sha256':sha256(args.output/'projection_multisource.joblib'),
              'feature_sha256':sha256(args.output/'features.npz'),
              'images':len(records),'patients':len({r['group'] for r in records}),
              'held_out_by_source':evaluate(records,scores),'cases':cases,
              'limitations':['Repeated source-held-out research, not a new unseen-site test.',
                             'GE AP labels are weak protocol references; no GE lateral reference.',
                             'No hip/oblique projection detection or clinical DXA certification.']}
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['held_out_by_source'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('buu','ramathibodi','organizer','labels','output'):
        p.add_argument('--'+name,type=Path,required=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Invalid value for VR UI:.*')
        run(p.parse_args())
