"""DXA-only, study-held-out ablation of the organizer's combined spine coverage label.

The label combines upper Th12 and lower iliac-crest visibility; this experiment
cannot produce separate landmark truth or a clinical half-body measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.model import CRITERIA, SEED, _cnn_lr, _lr, _matrix, _rf, best_f1_threshold
from source_integrity import inspect_sources
from train import build_table

TARGET = 'spine_coverage'
VARIANTS = ('current_ensemble', 'joint_dxa_lr')


def counts(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not np.isin(truth,[0,1]).all() or not np.isin(prediction,[0,1]).all():
        raise ValueError('Binary labels and predictions required')
    tn=int(((truth==0)&(prediction==0)).sum());fp=int(((truth==0)&(prediction==1)).sum())
    fn=int(((truth==1)&(prediction==0)).sum());tp=int(((truth==1)&(prediction==1)).sum())
    return {'tn_fp_fn_tp':[tn,fp,fn,tp], 'f1':2*tp/max(1,2*tp+fp+fn),
            'sensitivity':tp/max(1,tp+fn),'specificity':tn/max(1,tn+fp)}


def predictors(geometry, embeddings, y, train, test):
    physical = _matrix([geometry[i] for i in train],CRITERIA['spine'][TARGET])
    physical_test = _matrix([geometry[i] for i in test],CRITERIA['spine'][TARGET])
    rf=_rf().fit(physical,y[train]);lr=_lr().fit(physical,y[train]);cnn=_cnn_lr().fit(embeddings[train],y[train])
    joint=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                        LogisticRegression(C=.01,class_weight='balanced',max_iter=5000,random_state=SEED))
    joint.fit(np.column_stack([physical,embeddings[train]]),y[train])
    baseline=np.mean([rf.predict_proba(physical_test)[:,1],lr.predict_proba(physical_test)[:,1],
                      cnn.predict_proba(embeddings[test])[:,1]],axis=0)
    combined=joint.predict_proba(np.column_stack([physical_test,embeddings[test]]))[:,1]
    return {'current_ensemble':baseline,'joint_dxa_lr':combined}


def run(labels,geometry,embeddings):
    selected=np.flatnonzero(labels.region.eq('spine').values)
    y=labels.loc[selected,TARGET].to_numpy(dtype=int)
    groups=labels.loc[selected,'study_key'].to_numpy()
    if len(selected)!=99 or int(y.sum())!=6 or len(set(groups))!=99:
        raise ValueError('Unexpected organizer DXA spine cohort')
    truth_all=pd.to_numeric(labels[TARGET],errors='coerce').to_numpy()
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    rows=[]
    for fold,(tr,te) in enumerate(outer.split(selected,y,groups)):
        if set(groups[tr]) & set(groups[te]):raise ValueError('Outer study leakage')
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+100)
        inner_scores={name:np.full(len(tr),np.nan) for name in VARIANTS}
        for itr,iva in inner.split(tr,y[tr],groups[tr]):
            if set(groups[tr[itr]]) & set(groups[tr[iva]]):raise ValueError('Inner study leakage')
            values=predictors(geometry,embeddings,truth_all,selected[tr[itr]],selected[tr[iva]])
            for name in VARIANTS:inner_scores[name][iva]=values[name]
        if any(not np.isfinite(v).all() for v in inner_scores.values()):raise ValueError('Missing inner scores')
        thresholds={name:best_f1_threshold(y[tr],inner_scores[name]) for name in VARIANTS}
        scores=predictors(geometry,embeddings,truth_all,selected[tr],selected[te])
        for j,pos in enumerate(te):
            row={'index':int(selected[pos]),'study_group':str(groups[pos]),'fold':fold,'reference':int(y[pos])}
            for name in VARIANTS:
                row[name+'_score']=float(scores[name][j]);row[name+'_prediction']=int(scores[name][j]>=thresholds[name])
            rows.append(row)
    table=pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    reference=table.reference.to_numpy()
    report={'protocol':'Organizer DXA only; source pixel identity audit; fixed 5-fold study-held-out outer CV, 3-fold inner threshold selection, seed17. No test-fold threshold tuning.',
            'images':len(table),'studies':int(table.study_group.nunique()),'positives':int(reference.sum()),
            'criterion':TARGET,'criterion_semantics':'Combined top Th12 and lower iliac-crest coverage; subtype not labelled',
            'variants':{},'clinical_validation':False,'release_modified':False,'requirements_complete':False,
            'limitations':['Only six organizer DXA positive cases; estimates have large uncertainty.',
                           'Previously inspected organizer data; exploratory ablation, not a fresh independent test.',
                           'Known spine region supplied, so routing and final quality decision are not evaluated.',
                           'Study-disjoint split does not establish patient-disjointness.',
                           'Combined coverage label does not identify which anatomical boundary failed.']}
    for name in VARIANTS:
        scores=table[name+'_score'].to_numpy();pred=table[name+'_prediction'].to_numpy()
        report['variants'][name]={**counts(reference,pred),'roc_auc':float(roc_auc_score(reference,scores)),
                                  'average_precision':float(average_precision_score(reference,scores))}
    return report,table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a new output report')
    labels_path=HERE/'labels/image_labels.csv'
    labels=pd.read_csv(labels_path)
    labels=labels[labels.quality_class.notna()].reset_index(drop=True)
    audit=inspect_sources([('organizer',labels_path,args.dataset)])
    fingerprint=hashlib.sha256(json.dumps([(x['path'],x['pixel_sha256_current']) for x in audit['entries']],
                                          ensure_ascii=False).encode()).hexdigest()
    geometry,embeddings,_=build_table(args.dataset,labels,HERE/'.cache',fingerprint)
    report,table=run(labels,geometry,embeddings)
    report.update(labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                  source_fingerprint=fingerprint,encoder_sha256=WEIGHTS_SHA256,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table.to_csv(args.output.with_name(args.output.stem+'_oof.csv'),index=False)
    print(json.dumps(report['variants'],indent=2))


if __name__=='__main__':main()
