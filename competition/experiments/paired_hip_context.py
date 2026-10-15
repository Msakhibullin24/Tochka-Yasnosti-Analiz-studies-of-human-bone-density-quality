"""Study-grouped relational hip experiment, never averaging sibling diagnoses.

Partner identity uses study and opposite image route only. Own/partner feature
differences are learned with study-exclusive preprocessing, PCA and targets.
Primary: 75% incumbent quality ranking + 25% contextual head; no-context and
context-only controls. Spine stays byte-for-byte the incumbent OOF rows.
This experiment is conditional on historical routing and on supplied study
context; deployment requires an explicit study API and own-image fallback.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dxaqc.model import _matrix,_lr
from evaluate_organizer_dataset import evaluate,sha256
from experiments.nested_criterion_upgrade import positive_probability,validate_outer
from experiments.specialist_model_council import train_model,probability,write_decisions
from experiments.quality_gated_decoder import fit_quality_cut
from experiments.three_iteration_upgrade import dump,fit_decoder,decode_scores,require_validation_bank
from experiments.verify_metric_candidate import compare
from gpu_research.common import load_feature_bundle

SOURCE=Path(__file__).read_bytes();SOURCE_SHA=hashlib.sha256(SOURCE).hexdigest()


def partners_from_routes(studies,routes):
    partner=np.full(len(studies),-1,int)
    for study in np.unique(studies):
        left=np.flatnonzero((studies==study)&(routes=='hip_left'));right=np.flatnonzero((studies==study)&(routes=='hip_right'))
        if len(left)>1 or len(right)>1:raise ValueError('Ambiguous hip acquisition; do not select a partner silently')
        if len(left)==len(right)==1:partner[left[0]]=right[0];partner[right[0]]=left[0]
    return partner


def fit_transformers(geometry,embedding,train):
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(geometry[train])
    scaler=StandardScaler().fit(imputer.transform(geometry[train]))
    pca=PCA(n_components=min(16,len(train)-1),svd_solver='full',random_state=17).fit(embedding[train])
    return {'imputer':imputer,'scaler':scaler,'pca':pca}


def relational_features(transform,geometry,embedding,partners,query,paired):
    def own(ids):return np.column_stack([transform['scaler'].transform(transform['imputer'].transform(geometry[ids])),transform['pca'].transform(embedding[ids])])
    values=own(query)
    if not paired:return values
    present=partners[query]>=0;other=values.copy()
    if present.any():other[present]=own(partners[query[present]])
    difference=values-other
    return np.column_stack([values,other,difference,abs(difference),present])


def fit_pair(geometry,embedding,partners,targets,train):
    transform=fit_transformers(geometry,embedding,train);models={}
    for paired in (False,True):
        x=relational_features(transform,geometry,embedding,partners,train,paired)
        # Reindex locally so training cannot inspect targets outside train.
        models[paired]={c:train_model(x,y[train],np.arange(len(train)),c=.01) for c,y in targets.items()}
    return {'transform':transform,'models':models}


def predict_pair(model,geometry,embedding,partners,query):
    output={}
    for paired,heads in model['models'].items():
        x=relational_features(model['transform'],geometry,embedding,partners,query,paired)
        output[paired]={c:probability(head,x,np.arange(len(query))) for c,head in heads.items()}
    absent=partners[query]<0
    for code in output[True]:output[True][code][absent]=output[False][code][absent]
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','ranking_bank','output'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--labels',type=Path,default=ROOT/'labels/image_labels.csv')
    p.add_argument('--reference',type=Path,default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15/quality-gated-own-roi/candidate_oof.csv'))
    args=p.parse_args()
    if args.output.exists():raise ValueError('Choose a new paired context output')
    banks=[];manifests=[]
    for directory in (args.bank,args.ranking_bank):
        m=json.loads((directory/'manifest.json').read_text());require_validation_bank(m)
        if sha256(directory/'bank.joblib')!=m['bank_sha256'] or sha256(args.labels)!=m['labels_sha256']:raise ValueError('Source bank identity differs')
        banks.append(joblib.load(directory/'bank.joblib'));manifests.append(m)
    bank,prior=banks;labels=pd.read_csv(args.labels);labels=labels[labels.quality_class.notna()].reset_index(drop=True);base=bank['baseline'];validate_outer(labels,base)
    reference=pd.read_csv(args.reference).sort_values('index').reset_index(drop=True)
    if not reference[['index','fold','study','source_path','quality_true']].equals(base[['index','fold','study','source_path','quality_true']]):raise ValueError('Reference rows differ')
    studies=base.study.astype(str).to_numpy();routes=base.predicted_region.to_numpy();partner=partners_from_routes(studies,routes)
    path=Path('data/gpu-runs/dinov3l-frozen-2026-10-04/dinov3-l-features/features.npy')
    embedding,meta=load_feature_bundle(path,labels,manifests[0]['source_fingerprint'],sha256(args.labels))
    columns=sorted({name for state in bank['folds'] if state['group']=='hip' for name in state['columns']['quality']})
    geometry=_matrix(bank['features'],columns)
    targets={c:pd.to_numeric(labels['quality_class' if c=='quality' else c],errors='coerce').to_numpy(float) for c in ('quality','hip_position_rotation','hip_roi_coverage')}
    variants={'contextual_blend':(True,.25),'single_blend':(False,.25),'context_only':(True,1.)}
    tables={name:reference.copy() for name in variants};audit=[];saved=[]
    args.output.mkdir(parents=True);(args.output/'executed_paired_hip_context.py').write_bytes(SOURCE)
    old_states={(s['fold'],s['group']):s for s in prior['folds']}
    with threadpool_limits(limits=2):
        for state in (s for s in bank['folds'] if s['group']=='hip'):
            fold=state['fold'];train,test=state['train'],state['test'];old=old_states[fold,'hip']
            for key in ('train','test'):np.testing.assert_array_equal(state[key],old[key])
            if any(partner[i]>=0 and partner[i] not in set(train) for i in train):raise ValueError('Training partner escaped study fold')
            inner={paired:{c:np.full(len(train),np.nan) for c in targets} for paired in (False,True)}
            splits=[]
            for it,iv in GroupKFold(4).split(train,groups=studies[train]):
                tr,va=train[it],train[iv]
                if set(studies[tr])&set(studies[va]):raise ValueError('Study leakage')
                prediction=predict_pair(fit_pair(geometry,embedding,partner,targets,tr),geometry,embedding,partner,va)
                for paired in inner:
                    for c in targets:inner[paired][c][iv]=prediction[paired][c]
                splits.append({'training_studies':sorted(set(studies[tr])),'validation_studies':sorted(set(studies[va]))})
            model=fit_pair(geometry,embedding,partner,targets,train);outer=predict_pair(model,geometry,embedding,partner,test)
            qspec=fit_decoder(old,1);typespec=fit_decoder(state,1);codes=state['codes']
            _,prior_inner=decode_scores(qspec,codes,np.column_stack([old['inner'][c] for c in codes]),old['inner']['quality'])
            _,prior_outer=decode_scores(qspec,codes,np.column_stack([old['outer'][c] for c in codes]),old['outer']['quality'])
            np.testing.assert_allclose(prior_outer,reference.loc[test,'quality_score'],atol=1e-12,rtol=0)
            specs={}
            for name,(paired,weight) in variants.items():
                newq={kind:.5*pred['quality']+.5*(1-np.prod(1-np.column_stack([pred[c] for c in codes]),axis=1)) for kind,pred in [('inner',inner[paired]),('outer',outer[paired])]}
                qi=(1-weight)*prior_inner+weight*newq['inner'];qo=(1-weight)*prior_outer+weight*newq['outer']
                inner_types={c:(1-weight)*state['inner'][c]+weight*inner[paired][c] for c in codes}
                outer_types={c:(1-weight)*state['outer'][c]+weight*outer[paired][c] for c in codes}
                from experiments.strict_nested_calibration import threshold
                thresholds={c:threshold(targets[c][train].astype(int),inner_types[c]) for c in codes}
                changed={**state,'inner':{**state['inner'],**inner_types}}
                cut=fit_quality_cut(changed,{'thresholds':thresholds},qi)
                spec={'alpha':1.,'quality_cut':cut};specs[name]={**spec,'weight':weight,'criterion_thresholds':thresholds}
                write_decisions(tables[name],bank['features'],'hip',test,{**outer_types,'quality':qo},codes,spec,thresholds)
            audit.append({'fold':fold,'training_studies':sorted(set(studies[train])),'held_out_studies':sorted(set(studies[test])),'inner_splits':splits,'specifications':specs})
            saved.append({'fold':fold,'model':model,'specifications':specs,'train':train,'test':test})
            print(f'Paired hip fold {fold} complete',flush=True)
    joblib.dump({'folds':saved,'columns':columns},args.output/'paired_heads.joblib',compress=3)
    dump(args.output/'protocol.json',{'protocol':__doc__,'primary':'contextual_blend','selection_audit':audit,'code_sha256':SOURCE_SHA,
        'reference_sha256':sha256(args.reference),'embedding_sha256':sha256(path),'clinical_validation':False,'deployment_ready':False,
        'paired_images':int(np.sum(partner>=0)),'single_hip_images':int(np.sum((partner<0)&(routes!='spine'))),'partner_target_values_used_as_inputs':False})
    for name,table in tables.items():
        folder=args.output/name;folder.mkdir();path=folder/'candidate_oof.csv';table.to_csv(path,index=False)
        metrics=evaluate(args.labels,path,repeats=2000);screen=compare(args.reference,path,args.labels)
        dump(folder/'evaluation.json',{'protocol':__doc__,'metrics':metrics,'screen':screen,'code_sha256':SOURCE_SHA,'clinical_validation':False,'automatic_promotions':0})
        print(json.dumps({'variant':name,'quality':metrics['overall_quality'],'macro_f1':metrics['criterion_macro_f1']}),flush=True)


if __name__=='__main__':main()
