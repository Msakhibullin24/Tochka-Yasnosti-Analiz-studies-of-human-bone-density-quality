"""Exploratory hip view classifier from publisher tags, never promoted automatically."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.embedding import embed, WEIGHTS_SHA256


def dhash(image):
    small=cv2.resize(image,(9,8),interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),'big')


def duplicate_groups(images):
    """Conservative transitive union of identical and similar 64-bit difference hashes."""
    parents=list(range(len(images)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    hashes=[dhash(image) for image in images]
    pixels=[hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest() for image in images]
    edges=[]
    for i in range(len(images)):
        for j in range(i):
            if pixels[i]==pixels[j] or (hashes[i]^hashes[j]).bit_count()<=4:
                a,b=root(i),root(j);parents[a]=b;edges.append([j,i])
    return np.array([root(i) for i in range(len(images))]),edges,pixels


def projection_label(row):
    names=('frontal','lateral','oblique')
    if any(row[k] not in ('0','1') for k in (*names,'mixed','multiscan')):
        raise ValueError('Unexpected publisher tag value')
    # Publisher mixed means multiple anatomical regions, not multiple views.
    if row['multiscan']=='1' or sum(int(row[k]) for k in names)!=1:
        return None
    return 0 if row['frontal']=='1' else 1


def run(args):
    if args.output.exists():raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    acquisition=json.loads((args.dataset.parent/'acquisition.json').read_text())
    if acquisition['archive_md5']!='fe9da2c7c285915ebee69dfdab8fd396':
        raise ValueError('Unexpected source archive')
    with (args.dataset/'hips.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    images=[];valid_rows=[];invalid_images=[]
    for row in rows:
        path=(args.dataset/'images'/row['image_id']).resolve()
        if not path.is_relative_to((args.dataset/'images').resolve()):raise ValueError('Unsafe image path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['source_sha256']:raise ValueError('Source hash drift')
        try:
            with Image.open(path) as source:
                source.load()  # Reject truncated publisher JPEGs rather than silently decoding partial pixels.
                image=np.array(source.convert('L'))
            if min(image.shape)<32:raise ValueError('Source raster is too small')
        except (OSError,ValueError,Image.DecompressionBombError) as exc:
            invalid_images.append({'image_id':row['image_id'],'source_sha256':row['source_sha256'],'reason':str(exc)})
            continue
        images.append(image);valid_rows.append(row)
    original_count=len(rows);rows=valid_rows
    groups,edges,pixels=duplicate_groups(images)
    labels=[projection_label(row) for row in rows]
    included=[i for i,label in enumerate(labels) if label is not None]
    # Conflicting labels within a detected duplicate component cannot form reliable supervision.
    conflict={g for g in set(groups) if len({labels[i] for i in included if groups[i]==g})>1}
    included=np.array([i for i in included if groups[i] not in conflict],dtype=int)
    y=np.array([labels[i] for i in included],dtype=int);g=groups[included]
    counts={str(k):int((y==k).sum()) for k in (0,1)}
    group_counts={str(k):len(set(g[y==k])) for k in (0,1)}
    args.output.mkdir(parents=True)
    report={'source':acquisition,'eligible_images':len(included),'class_counts':counts,
            'class_group_counts':group_counts,'duplicate_edges':edges,'conflicting_components':sorted(int(v) for v in conflict),
            'excluded_images':original_count-len(included),'invalid_images':invalid_images,'clinical_validation':False,'release_modified':False,
            'requirements_complete':False,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':['Ordinary radiographs, not DXA; frontal author tag does not distinguish AP from PA.',
                           'Patient IDs are unavailable: perceptual duplicate groups do not establish patient independence.',
                           '64-bit difference-hash distance <=4 catches some similar rasters, not every duplicated acquisition.',
                           'Composite, ambiguous and conflicting-view components are excluded; mixed means multiple body regions and is retained with one explicit view.',
                           'Multi-region radiographs may introduce anatomical-context shortcuts; no expert DXA nonfrontal or rotation-subtype labels are supplied.']}
    if min(group_counts.values())<5:
        report['status']='insufficient_independent_view_groups'
    else:
        train,test=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=17).split(included,y,g))
        if set(g[train])&set(g[test]) or len(set(y[train]))<2 or len(set(y[test]))<2:
            raise ValueError('Invalid frozen group holdout')
        features=[]
        for number,i in enumerate(included):
            square=cv2.resize(images[i],(320,320),interpolation=cv2.INTER_AREA)
            features.append(np.stack([embed(square),embed(255-square)]))
            if (number+1)%25==0:print(f'Encoded hip views {number+1}/{len(included)}',flush=True)
        x=np.array(features)
        np.save(args.output/'features.npy',x)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=17))
        model.fit(x[train].reshape(-1,x.shape[-1]),np.repeat(y[train],2))
        scores=model.predict_proba(x[test].reshape(-1,x.shape[-1]))[:,1].reshape(-1,2).mean(axis=1)
        predictions=(scores>=.5).astype(int)
        report.update(status='exploratory_candidate',protocol='Frozen 5-fold stratified duplicate-group split, first fold seed17; C=1; no parameter tuning.',
                      confusion_matrix_0_frontal_1_nonfrontal=confusion_matrix(y[test],predictions,labels=[0,1]).tolist(),
                      auc=float(roc_auc_score(y[test],scores)),training_images=len(train),held_out_images=len(test),
                      confident_frontal=int((scores<=.1).sum()),confident_nonfrontal=int((scores>=.9).sum()),
                      uncertain=int(((scores>.1)&(scores<.9)).sum()),
                      nonfrontal_called_confident_frontal=int(((y[test]==1)&(scores<=.1)).sum()),
                      frontal_called_confident_nonfrontal=int(((y[test]==0)&(scores>=.9)).sum()))
        bundle={'schema_version':2,'status':'research_only','encoder_sha256':WEIGHTS_SHA256,'model':model,'scope':'hip_frontal_vs_nonfrontal',
                'clinical_validation':False,'source_doi':acquisition['doi']}
        joblib.dump(bundle,args.output/'candidate.joblib')
        report['model_sha256']=hashlib.sha256((args.output/'candidate.joblib').read_bytes()).hexdigest()
        heldout={int(included[i]):float(s) for i,s in zip(test,scores)}
        report['cases']=[{'image_id':rows[i]['image_id'],'source_sha256':rows[i]['source_sha256'],
                          'pixel_sha256':pixels[i],'duplicate_group':int(groups[i]),'label':labels[i],'multiple_body_regions':rows[i]['mixed']=='1',
                          'split':'test' if int(i) in heldout else 'train',
                          'nonfrontal_score':heldout.get(int(i))} for i in included]
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','class_counts','class_group_counts','excluded_images')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
