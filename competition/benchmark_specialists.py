"""Paired fixed-study benchmark: refit hybrid, regional CNNs, validation-selected fusion.

Never uses the shipped all-data bundle to estimate held-out accuracy. All candidates
must use the fixed train/validation/test split and the exact source label table.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from dxaqc.dicom_io import read_any, CALIBRATION_VERSION
from dxaqc.model import CRITERIA, GroupModel, QualityBundle, RegionRouter, group_of, best_f1_threshold
from dxaqc.pipeline import Analyzer
from dxaqc.specialist_qc import OUTPUTS, digest
from experiments.cnn_quality import valid_labels
from train import build_table, binary_metrics
from train_specialist import split_indices
from report_specialists import load_run


def evaluate(y, score, pred):
    m = binary_metrics(y.astype(int),score,pred.astype(int))
    tn,fp,fn,tp = m['confusion_tn_fp_fn_tp']
    m['precision'] = tp/(tp+fp) if tp+fp else None
    m['accuracy'] = (tp+tn)/len(y) if len(y) else None
    if not np.any(y==1): m['sensitivity']=None
    if not np.any(y==0): m['specificity']=None
    return m


def choose_fusion(y, baseline, regional):
    # One small, predetermined search; no weights/thresholds selected on test.
    options = [(float(average_precision_score(y,(1-w)*baseline+w*regional)),w)
               for w in (0.,.25,.5,.75,1.)]
    _,weight = max(options,key=lambda pair:(pair[0],-pair[1]))
    threshold = float(best_f1_threshold(y, (1-weight)*baseline+weight*regional))
    return weight,threshold


def run(args):
    if args.output.exists(): raise ValueError('Choose a new benchmark directory')
    if os.environ.get('DXAQC_SPECIALIST_PATH'):
        raise ValueError('Unset DXAQC_SPECIALIST_PATH: benchmark must not load a deployed candidate')
    labels=valid_labels(args.labels);train,val,test=split_indices(labels)
    y=labels.quality_class.to_numpy(dtype=int)
    sources={'joint':args.joint,'spine':args.spine,'hip':args.hip}
    candidates={key:load_run(path,args.labels) for key,path in sources.items()}
    expected=np.empty(len(labels),dtype=object)
    for name,idx in zip(('train','validation','test'),(train,val,test)): expected[idx]=name
    for name,(meta,_,_,splits,_) in candidates.items():
        if not np.array_equal(splits,expected): raise ValueError('Candidate split mismatch')
        if meta.get('region_scope','all') != ('all' if name=='joint' else name):
            raise ValueError('Candidate region mismatch')
    started=time.perf_counter()
    feats,emb,raw=build_table(args.dataset,labels,Path('competition/.cache'))
    groups={}
    for region in ('spine','hip'):
        tr=np.array([i for i in train if group_of(labels.region[i])==region])
        va=np.array([i for i in val if group_of(labels.region[i])==region])
        targets={k:pd.to_numeric(labels[k],errors='raise').to_numpy(dtype=float) for k in CRITERIA[region]}
        model=GroupModel(region).fit([feats[i] for i in tr],emb[tr],y[tr],{k:v[tr] for k,v in targets.items()})
        score,criteria=model.predict([feats[i] for i in va],emb[va])
        model.quality_threshold=float(best_f1_threshold(y[va],score))
        for key in criteria:
            known=np.isfinite(targets[key][va])
            model.criterion_thresholds[key]=(float(best_f1_threshold(targets[key][va][known].astype(int),criteria[key][known]))
                                              if len(np.unique(targets[key][va][known]))==2 else .5)
        groups[region]=model
    router=RegionRouter().fit(raw[train],list(labels.region.iloc[train]))
    bundle=QualityBundle(router,groups,{'calibration_version':CALIBRATION_VERSION,
                                      'scope':'fixed train only; validation thresholds','labels_sha256':digest(args.labels)})
    analyzer=Analyzer(bundle)
    baseline_score,baseline_pred,routed,failures=[],[],[],[]
    criterion_scores={k:np.full(len(labels),np.nan) for k in OUTPUTS[1:]}
    for i,row in enumerate(labels.itertuples()):
        path=(args.dataset/row.first_source_path).resolve()
        if not path.is_relative_to(args.dataset.resolve()): raise ValueError('Image path outside dataset')
        result=analyzer.analyze(read_any(path))  # Fail the benchmark rather than drop unreadable samples.
        baseline_score.append(result['score']);baseline_pred.append(result['quality']);routed.append(result['region'])
        for key,value in result['criteria'].items(): criterion_scores[key][i]=value
        if (i+1)%50==0: print(f'pipeline {i+1}/{len(labels)}',flush=True)
    baseline_score=np.array(baseline_score);baseline_pred=np.array(baseline_pred)
    regional_score=np.array([candidates[group_of(r)][2][i,0] for i,r in enumerate(routed)])
    regional_pred=np.array([regional_score[i]>=candidates[group_of(r)][0]['thresholds']['quality'] for i,r in enumerate(routed)])
    joint_score=candidates['joint'][2][:,0]
    joint_pred=joint_score>=candidates['joint'][0]['thresholds']['quality']
    weight,threshold=choose_fusion(y[val],baseline_score[val],regional_score[val])
    fused=(1-weight)*baseline_score+weight*regional_score
    variants={'hybrid_refit':(baseline_score,baseline_pred),'joint_frozen_cnn':(joint_score,joint_pred),
              'regional_finetuned_cnn':(regional_score,regional_pred),'validation_selected_fusion':(fused,fused>=threshold)}
    metrics={name:{part:evaluate(y[idx],scores[idx],pred[idx]) for part,idx in zip(('train','validation','test'),(train,val,test))}
             for name,(scores,pred) in variants.items()}
    by_region={region:{name:evaluate(y[idx],scores[idx],pred[idx]) for name,(scores,pred) in variants.items()}
               for region in ('spine','hip')
               for idx in [np.array([i for i in test if group_of(labels.region[i])==region])]}
    report={'scope':'paired fixed-study split, raw DICOM hybrid inference; regional selection uses the trained router',
            'limitations':['Internal development benchmark; this test was previously inspected.',
                           'No patient-identity or external validation. No synthetic augmentation in refit baseline.',
                           'CNN training features are supervised only on train; epoch/threshold/fusion selection uses validation.'],
            'labels_sha256':digest(args.labels),'split':{part:{'images':len(idx),'studies':int(labels.study_key.iloc[idx].nunique())}
                for part,idx in zip(('train','validation','test'),(train,val,test))},
            'source_models':{k:{'model_sha256':v[0]['model_sha256'],'metadata_sha256':digest(sources[k]/'model.json')}
                             for k,v in candidates.items()},
            'router_test_accuracy':float(np.mean(np.array(routed)[test]==labels.region.to_numpy()[test])),
            'fusion':{'regional_weight':weight,'threshold':threshold,'selection':'validation AP then validation F1'},
            'metrics':metrics,'test_by_region':by_region,'duration_seconds':time.perf_counter()-started,
            'deployment_changed':False,'clinical_validation':False}
    args.output.mkdir(parents=True)
    joblib.dump(bundle,args.output/'train_only_bundle.joblib')
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=[{'label_row':i,'split':str(expected[i]),'study_hash':hashlib.sha256(str(labels.study_key[i]).encode()).hexdigest(),
           'true_region':labels.region[i],'predicted_region':routed[i],'quality_true':int(y[i]),
           **{name+'_score':float(scores[i]) for name,(scores,pred) in variants.items()},
           **{name+'_pred':int(pred[i]) for name,(scores,pred) in variants.items()}} for i in range(len(labels))]
    (args.output/'predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
    body=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение моделей DXA</title><main>',
          '<h1>Модели на одинаковых исследованиях</h1><p>149 train / 54 validation / 46 test. '
          'Гибрид переобучен только на train; пороги и вес ансамбля выбраны на validation. '
          'Test уже использовался в разработке; это не независимая клиническая оценка.</p>',
          '<p><a href="metrics.json">Все метрики</a></p><table border="1" cellpadding="6">',
          '<caption>Результаты test</caption><tr><th>Модель</th><th>AUC</th><th>AP</th><th>F1</th><th>Recall</th><th>Specificity</th></tr>']
    for name,parts in metrics.items():
        m=parts['test'];body.append('<tr><th>'+html.escape(name)+'</th>'+''.join(f'<td>{m.get(k,0):.3f}</td>' for k in ('roc_auc','pr_auc','f1','sensitivity','specificity'))+'</tr>')
    body.append(f'</table><p>Вес региональной CNN: {weight}; порог ансамбля: {threshold:.4f}. '
                'Основная модель автоматически не заменяется.</p></main></html>')
    (args.output/'index.html').write_text('\n'.join(body))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--labels',type=Path,default=Path(__file__).parent/'labels/image_labels.csv')
    for name in ('joint','spine','hip','output'): p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
