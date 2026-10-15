"""Task-specialized eight-source council with three-level study cross-fitting.

Each quality/type target gets its own visual and geometric experts and arbiter.
Legacy geometry/DINO/RAD combinations are explicit controls. Policy selection,
quality blending and thresholds use middle OOF only; outer labels are withheld.
Metrics remain exploratory on the previously inspected, conditionally routed
cohort. Saved research models require export/raw replay before deployment.
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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dxaqc.decision import decide
from dxaqc.model import _matrix,_lr,_rf,official_violation_type
from evaluate_organizer_dataset import evaluate,sha256
from experiments.nested_criterion_upgrade import positive_probability,validate_outer
from experiments.nested_model_council import FEATURES,fit_arbiter,predict_arbiter,quality_cut
from experiments.quality_gated_decoder import gated_decisions
from experiments.strict_nested_calibration import threshold,grouped_threshold
from experiments.three_iteration_upgrade import dump,require_validation_bank
from experiments.verify_metric_candidate import compare
from gpu_research.common import load_feature_bundle

SOURCE=Path(__file__).read_bytes();SOURCE_SHA=hashlib.sha256(SOURCE).hexdigest()
POLICIES=('mean','simplex','stack_shrink','stack_regular','legacy_reference','legacy_anatomy')
NAMES=('dinov3','raddino','medsiglip','stanford_xray','local_anatomy','iqa','anatomical_geometry','reference_geometry')


def train_model(x,target,train,kind='lr',c=.01):
    known=train[np.isfinite(target[train])];y=target[known].astype(int)
    if not len(known):raise ValueError('No known target in training studies')
    if len(np.unique(y))==1:
        model=DummyClassifier(strategy='constant',constant=int(y[0]));model.fit(np.zeros((len(y),1)),y)
    else:
        model=_rf() if kind=='rf' else _lr(c)
        model.fit(x[known],y)
    return model


def probability(model,x,query):
    return positive_probability(model,np.zeros((len(query),1)) if isinstance(model,DummyClassifier) else x[query])


def fit_raw(embeddings,geometries,targets,train):
    result={}
    for code,target in targets.items():
        heads=[train_model(x,target,train,c=.3 if i==5 else .01) for i,x in enumerate(embeddings)]
        for matrix in geometries[code]:
            heads.append([train_model(matrix,target,train,'rf'),train_model(matrix,target,train,c=.3)])
        result[code]=heads
    return result


def predict_raw(models,embeddings,geometries,query):
    output={}
    for code,heads in models.items():
        columns=[probability(model,x,query) for model,x in zip(heads[:6],embeddings,strict=True)]
        for pair,matrix in zip(heads[6:],geometries[code],strict=True):
            columns.append(np.mean([probability(model,matrix,query) for model in pair],axis=0))
        output[code]=np.column_stack(columns)
    return output


def crossfit(embeddings,geometries,targets,studies,train,count):
    output={c:np.full((len(train),8),np.nan) for c in targets};audit=[]
    for it,iv in GroupKFold(count).split(train,groups=studies[train]):
        tr,va=train[it],train[iv]
        if set(studies[tr])&set(studies[va]):raise ValueError('Study leakage')
        prediction=predict_raw(fit_raw(embeddings,geometries,targets,tr),embeddings,geometries,va)
        for code in output:output[code][iv]=prediction[code]
        audit.append({'train':tr.tolist(),'validation':va.tolist()})
    if not all(np.isfinite(x).all() for x in output.values()):raise ValueError('Incomplete raw OOF')
    return output,audit


def fit_policy(policy,scores,targets,train):
    result={}
    for code,x in scores.items():
        if policy.startswith('legacy_'):result[code]={'policy':policy}
        else:
            known=np.isfinite(targets[code][train])
            result[code]=fit_arbiter(policy,x[known],np.zeros(known.sum()),targets[code][train][known].astype(int))
    return result


def predict_policy(models,scores):
    result={}
    for code,model in models.items():
        policy=model['policy'];x=scores[code]
        if policy.startswith('legacy_'):
            result[code]=.25*x[:,0]+.25*x[:,1]+.5*x[:,7 if policy=='legacy_reference' else 6]
        else:result[code]=predict_arbiter(model,x,np.zeros(len(x)))
    return result


def blend_quality(prediction,codes,alpha):
    raw=np.column_stack([prediction[c] for c in codes])
    return alpha*prediction['quality']+(1-alpha)*(1-np.prod(1-raw,axis=1))


def select_operating_point(prediction,codes,target,forced,groups,bags):
    alpha=max((0.,.5,1.),key=lambda a:(roc_auc_score(target,blend_quality(prediction,codes,a)),-abs(a-.5)))
    q=blend_quality(prediction,codes,alpha)
    # The forced physical axis is included in every resampled operating point.
    cut=quality_cut(target,q,forced)
    if bags:
        rng=np.random.default_rng(17);unique=np.unique(groups);cuts=[]
        for _ in range(bags):
            chosen=rng.choice(unique,len(unique),replace=True)
            ids=np.concatenate([np.flatnonzero(groups==s) for s in chosen])
            if len(np.unique(target[ids]))==2:cuts.append(quality_cut(target[ids],q[ids],forced[ids]))
        if cuts:cut=float(np.median(cuts))
    cls=(q>=cut)|forced;tp=np.sum((target==1)&cls);fp=np.sum((target==0)&cls);fn=np.sum((target==1)&~cls)
    return {'alpha':alpha,'quality_cut':cut,'inner_f1':float(2*tp/max(1,2*tp+fp+fn)),
            'inner_auc':float(roc_auc_score(target,q)), 'threshold_bootstrap_studies':bags}


def write_decisions(table,features,group,test,predictions,codes,spec,thresholds):
    q=blend_quality(predictions,codes,spec['alpha']);raw=np.column_stack([predictions[c] for c in codes])
    angles=np.array([features[i].get('spine_abs_angle_deg',np.nan) for i in test],float)
    selected,fallback=gated_decisions(raw,[thresholds[c] for c in codes],codes,angles,q,spec['quality_cut'])
    for pos,index in enumerate(test):
        scores={c:float(predictions[c][pos]) for c in codes}
        decision=decide(group,float(q[pos]),scores,features[index],spec['quality_cut'],thresholds)
        for j,code in enumerate(codes):
            if code!='spine_axis':decision['criterion_states'][code]={'status':'fail' if selected[pos,j] else 'pass',
                'basis':'specialist_council_argmax_review' if fallback[pos] and selected[pos,j] else 'specialist_council_gate',
                'criterion_score':scores[code],'clinical_validation':False}
        violations=[c for j,c in enumerate(codes) if selected[pos,j]]
        updates={'quality_score':float(q[pos]),'quality_pred':int(bool(violations)),
            'violation_type':official_violation_type(violations),'criterion_scores':json.dumps(scores,sort_keys=True),
            'criterion_states':json.dumps(decision['criterion_states'],sort_keys=True),
            'criterion_thresholds':json.dumps(thresholds,sort_keys=True),'quality_threshold':spec['quality_cut'],
            'decision_reason':'specialist_council_argmax_review' if fallback[pos] else 'specialist_council_gate',
            'decision_version':'research-specialist-council-1'}
        for key,value in updates.items():table.at[index,key]=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','ranking_bank','output'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--labels',type=Path,default=ROOT/'labels/image_labels.csv')
    p.add_argument('--reference',type=Path,default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15/quality-gated-own-roi/candidate_oof.csv'))
    p.add_argument('--bootstrap',type=int,default=2000);args=p.parse_args()
    if args.output.exists():raise ValueError('Choose a new specialist council directory')
    manifests=[];banks=[]
    for folder in (args.bank,args.ranking_bank):
        manifest=json.loads((folder/'manifest.json').read_text());require_validation_bank(manifest)
        if manifest['bank_sha256']!=sha256(folder/'bank.joblib') or manifest['labels_sha256']!=sha256(args.labels):raise ValueError('Bank identity differs')
        manifests.append(manifest);banks.append(joblib.load(folder/'bank.joblib'))
    if manifests[0]['source_fingerprint']!=manifests[1]['source_fingerprint']:raise ValueError('Source pixels differ')
    bank,prior=banks;baseline=bank['baseline'];labels=pd.read_csv(args.labels);labels=labels[labels.quality_class.notna()].reset_index(drop=True)
    validate_outer(labels,baseline)
    np.testing.assert_array_equal(baseline['index'],np.arange(len(labels)))
    reference=pd.read_csv(args.reference).sort_values('index').reset_index(drop=True)
    if not reference[['index','fold','study','source_path','quality_true']].equals(baseline[['index','fold','study','source_path','quality_true']]):raise ValueError('Reference identity differs')
    embeddings=[];feature_audit=[]
    for name in FEATURES:
        path=Path(name);x,meta=load_feature_bundle(path,labels,manifests[0]['source_fingerprint'],sha256(args.labels));embeddings.append(x)
        feature_audit.append({'path':name,'sha256':sha256(path),'metadata_sha256':sha256(path.with_suffix('.json'))})
    studies=labels.study_key.astype(str).to_numpy();reference_states={(s['fold'],s['group']):s for s in prior['folds']}
    tables={(policy,bags):baseline.copy() for policy in (*POLICIES,'selected_by_inner') for bags in (0,100)}
    saved=[];audit=[];args.output.mkdir(parents=True);(args.output/'executed_specialist_model_council.py').write_bytes(SOURCE)
    with threadpool_limits(limits=2):
        for state in bank['folds']:
            fold,group=state['fold'],state['group'];train,test=state['train'],state['test'];codes=state['codes']
            old=reference_states[fold,group]
            for key in ('train','test'):np.testing.assert_array_equal(state[key],old[key])
            targets={c:pd.to_numeric(labels['quality_class' if c=='quality' else c],errors='coerce').to_numpy(float) for c in [*codes,'quality']}
            safe={c:np.full(len(labels),np.nan) for c in targets}
            for c in safe:safe[c][train]=targets[c][train]
            geometries={c:(_matrix(bank['features'],state['columns'][c]),_matrix(prior['features'],old['columns'][c])) for c in safe}
            raw_oof,middle_audit=crossfit(embeddings,geometries,safe,studies,train,4)
            middle={policy:{c:np.full(len(train),np.nan) for c in safe} for policy in POLICIES};subaudits=[]
            for split,(it,iv) in enumerate(GroupKFold(4).split(train,groups=studies[train])):
                tr,va=train[it],train[iv]
                np.testing.assert_array_equal(middle_audit[split]['train'],tr);np.testing.assert_array_equal(middle_audit[split]['validation'],va)
                sub_oof,subaudit=crossfit(embeddings,geometries,safe,studies,tr,3)
                query={c:values[iv] for c,values in raw_oof.items()}
                for policy in POLICIES:
                    prediction=predict_policy(fit_policy(policy,sub_oof,safe,tr),query)
                    for c in safe:middle[policy][c][iv]=prediction[c]
                subaudits.append({'middle_validation':va.tolist(),'subfolds':subaudit})
                print(f'Specialist outer {fold} {group}: middle {split} complete',flush=True)
            experts=fit_raw(embeddings,geometries,safe,train);raw_outer=predict_raw(experts,embeddings,geometries,test)
            forced=np.isfinite(state['angles'])&(abs(state['angles'])>5.) if group=='spine' else np.zeros(len(train),bool)
            specifications={};arbiters={};outer_predictions={}
            for policy in POLICIES:
                arbiters[policy]=fit_policy(policy,raw_oof,safe,train);outer_predictions[policy]=predict_policy(arbiters[policy],raw_outer)
                cuts={c:threshold(targets[c][train][np.isfinite(targets[c][train])].astype(int),middle[policy][c][np.isfinite(targets[c][train])]) for c in codes}
                for bags in (0,100):
                    spec=select_operating_point(middle[policy],codes,targets['quality'][train].astype(int),forced,studies[train],bags)
                    specifications[policy,bags]={**spec,'criterion_thresholds':cuts}
                    write_decisions(tables[policy,bags],bank['features'],group,test,outer_predictions[policy],codes,spec,cuts)
            selections={}
            for bags in (0,100):
                policy=max(POLICIES,key=lambda policy:(specifications[policy,bags]['inner_f1'],specifications[policy,bags]['inner_auc'],-POLICIES.index(policy)))
                selections[bags]=policy;spec=specifications[policy,bags]
                write_decisions(tables['selected_by_inner',bags],bank['features'],group,test,outer_predictions[policy],codes,spec,spec['criterion_thresholds'])
            saved.append({'fold':fold,'group':group,'train':train,'test':test,'experts':experts,'arbiters':arbiters,
                'specifications':specifications,'selected_by_inner':selections,'raw_expert_oof':raw_oof,'raw_expert_outer':raw_outer,'middle_predictions':middle,
                'geometry_columns':{c:[state['columns'][c],old['columns'][c]] for c in safe}})
            audit.append({'fold':fold,'group':group,'train_studies':sorted(set(studies[train])),'test_studies':sorted(set(studies[test])),
                'middle_splits':middle_audit,'subfolds':subaudits,'selected_by_inner':selections,'outer_labels_used':False})
            joblib.dump({'folds':saved,'source_names':NAMES},args.output/'specialist_bank.joblib',compress=3)
            dump(args.output/'progress.json',{'completed_group_folds':len(saved),'planned_group_folds':len(bank['folds'])})
    dump(args.output/'protocol.json',{'protocol':__doc__,'source_names':NAMES,'feature_audit':feature_audit,'code_sha256':SOURCE_SHA,
        'bank_sha256':[m['bank_sha256'] for m in manifests],'selection_audit':audit,'reference_sha256':sha256(args.reference),
        'clinical_validation':False,'deployment_ready':False,'score_calibration_established':False,
        'dependency_code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in ('nested_model_council.py','quality_gated_decoder.py','three_iteration_upgrade.py')}})
    for (policy,bags),table in tables.items():
        folder=args.output/(policy+f'-bags{bags}');folder.mkdir();path=folder/'candidate_oof.csv';table.to_csv(path,index=False)
        metrics=evaluate(args.labels,path,repeats=args.bootstrap);screen=compare(args.reference,path,args.labels)
        dump(folder/'evaluation.json',{'protocol':__doc__,'metrics':metrics,'screen':screen,'policy':policy,'threshold_bootstrap_studies':bags,
            'code_sha256':SOURCE_SHA,'clinical_validation':False,'meta_predictions_fully_cross_fitted':True,
            'policy_selected_without_outer_labels':policy=='selected_by_inner','automatic_promotions':0})
        print(json.dumps({'policy':policy,'bags':bags,'quality':metrics['overall_quality'],'macro_f1':metrics['criterion_macro_f1']}),flush=True)


if __name__=='__main__':main()
