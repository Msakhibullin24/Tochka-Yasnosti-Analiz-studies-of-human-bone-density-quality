"""Seven-source council with fully cross-fitted second-level predictions.

Five external study folds; four middle folds select operating points; three
subfolds generate expert inputs for each middle-training arbiter. An external
label cannot enter experts, arbitration, calibration or threshold selection.
Existing global OOF is used only as an identity template and final comparator.
Routing/canonicalization remain conditional on the historical image route.
Repeated use of this cohort is exploratory, not independent patient validation.
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
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.decision import decide
from dxaqc.model import _matrix, official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from experiments.nested_criterion_upgrade import positive_probability, validate_outer
from experiments.quality_gated_decoder import gated_decisions
from experiments.three_iteration_upgrade import dump, fit_decoder, require_validation_bank
from experiments.verify_metric_candidate import compare
from gpu_research.common import load_feature_bundle

SOURCE = Path(__file__).read_bytes()
SOURCE_SHA = hashlib.sha256(SOURCE).hexdigest()
POLICIES = ('mean', 'geometry_half', 'simplex', 'stack_shrink', 'stack_regular')
EXPERT_NAMES = ('dinov3', 'raddino', 'medsiglip', 'stanford_xray', 'local_anatomy', 'iqa', 'geometry')
FEATURES = (
    'data/gpu-runs/dinov3l-frozen-2026-10-04/dinov3-l-features/features.npy',
    'data/gpu-runs/rad-dino-frozen-2026-10-04/features.npy',
    'data/gpu-runs/medsiglip-frozen-2026-10-04/medsiglip-features/features.npy',
    'data/gpu-runs/dinov2-xray-frozen-2026-10-04/features.npy',
    'data/gpu-runs/anatomy-upgrade-2026-10-15/anatomical-dino-v2/features.npy',
    'data/gpu-runs/iqa-2026-10-04/features.npy')


def classifier(x, y, ids, kind='lr', c=.01):
    target = y[ids]
    if not len(ids) or not np.isfinite(target).all(): raise ValueError('Invalid training targets')
    target = target.astype(int)
    if len(np.unique(target)) == 1:
        model = DummyClassifier(strategy='constant', constant=int(target[0]))
    elif kind == 'rf':
        model = RandomForestClassifier(n_estimators=256, min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=17, n_jobs=1)
    else:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=17)
    fitted = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), model)
    fitted.fit(x[ids], target)
    return fitted


def fit_experts(matrices, regions, y, train):
    """No label outside train is inspected, including class-count fallbacks."""
    models = []
    for i, x in enumerate(matrices):
        global_model = classifier(x, y, train, c=.3 if i in (5, 6) else .01)
        regional = {}
        if i < 5:
            for region in (0, 1):
                selected = train[regions[train] == region]
                if len(selected): regional[region] = classifier(x, y, selected)
        forest = classifier(x, y, train, 'rf') if i == 6 else None
        models.append({'global': global_model, 'regional': regional, 'forest': forest})
    return models


def predict_experts(models, matrices, regions, query):
    columns = []
    for model, x in zip(models, matrices, strict=True):
        value = positive_probability(model['global'], x[query])
        if model['regional']:
            local = value.copy()
            for group, estimator in model['regional'].items():
                positions = np.flatnonzero(regions[query] == group)
                if len(positions): local[positions] = positive_probability(estimator, x[query[positions]])
            value = .5*(value+local)
        if model['forest'] is not None: value = .5*(value+positive_probability(model['forest'], x[query]))
        columns.append(value)
    result = np.column_stack(columns)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite council expert score')
    return result


def crossfit_experts(matrices, regions, target, studies, train, splits):
    values = np.full((len(train), len(matrices)), np.nan)
    audit = []
    for it, iv in GroupKFold(splits).split(train, groups=studies[train]):
        tr, va = train[it], train[iv]
        if set(studies[tr]) & set(studies[va]): raise ValueError('Study leakage')
        models = fit_experts(matrices, regions, target, tr)
        values[iv] = predict_experts(models, matrices, regions, va)
        audit.append({'training_studies': sorted(set(studies[tr])), 'validation_studies': sorted(set(studies[va]))})
    if not np.isfinite(values).all(): raise ValueError('Incomplete expert OOF')
    return values, audit


def meta_design(scores, regions):
    return np.column_stack([logit(np.clip(scores, 1e-5, 1-1e-5)), regions])


def fit_arbiter(policy, scores, regions, target):
    if policy in ('mean', 'geometry_half'): return {'policy': policy}
    if policy == 'simplex':
        prior = np.full(scores.shape[1], 1/scores.shape[1])
        def objective(weights):
            q = np.clip(scores @ weights, 1e-6, 1-1e-6)
            return float(-np.mean(target*np.log(q)+(1-target)*np.log1p(-q))+.1*np.sum((weights-prior)**2))
        solution = minimize(objective, prior, method='SLSQP', bounds=[(0,1)]*len(prior),
            constraints={'type':'eq', 'fun':lambda w: w.sum()-1}, options={'maxiter':200, 'ftol':1e-10})
        if not solution.success: raise ValueError('Simplex optimization failed: '+solution.message)
        return {'policy': policy, 'weights': solution.x}
    model = LogisticRegression(C=.01 if policy == 'stack_shrink' else .1, max_iter=2000, random_state=17)
    if len(np.unique(target)) == 1: model = DummyClassifier(strategy='constant', constant=int(target[0]))
    model.fit(meta_design(scores, regions), target)
    return {'policy': policy, 'model': model}


def predict_arbiter(model, scores, regions):
    policy = model['policy']
    if policy == 'mean': return scores.mean(1)
    if policy == 'geometry_half': return .5*scores[:,-1]+.5*scores[:,:-1].mean(1)
    if policy == 'simplex': return np.clip(scores @ model['weights'],0,1)
    return positive_probability(model['model'], meta_design(scores, regions))


def quality_cut(truth, scores, forced):
    best, selected = None, None
    for cut in np.r_[np.unique(scores), np.nextafter(1., np.inf)]:
        prediction = (scores >= cut) | forced
        tp = np.sum((truth == 1) & prediction)
        fp = np.sum((truth == 0) & prediction)
        fn = np.sum((truth == 1) & ~prediction)
        rank = (2*tp/max(1,2*tp+fp+fn), -int(fp), float(cut))
        if best is None or rank > best: best, selected = rank, float(cut)
    return selected


def apply_predictions(table, bank, fold, quality, cut_by_region):
    baseline = bank['baseline']
    full_test = np.flatnonzero(baseline.fold.to_numpy() == fold)
    for state in (s for s in bank['folds'] if s['fold'] == fold):
        test = state['test']; codes = state['codes']; spec = fit_decoder(state, 1)
        q = quality[np.searchsorted(full_test, test)]
        cut = cut_by_region[0 if state['group'] == 'spine' else 1]
        scores = np.column_stack([state['outer'][c] for c in codes])
        thresholds = np.array([spec['thresholds'][c] for c in codes])
        angles = np.array([bank['features'][i].get('spine_abs_angle_deg', np.nan) for i in test], float)
        selected, fallback = gated_decisions(scores, thresholds, codes, angles, q, cut)
        for pos, row_index in enumerate(test):
            raw_scores = dict(zip(codes,map(float,scores[pos])))
            decision = decide(state['group'], float(q[pos]), raw_scores, bank['features'][row_index], cut, spec['thresholds'])
            for j, code in enumerate(codes):
                if code != 'spine_axis':
                    decision['criterion_states'][code] = {'status':'fail' if selected[pos,j] else 'pass',
                        'basis':'council_conditioned_argmax' if fallback[pos] and selected[pos,j] else 'council_quality_gate',
                        'criterion_score':raw_scores[code], 'clinical_validation':False}
            violations = [c for j,c in enumerate(codes) if selected[pos,j]]
            for key, value in {'quality_score':float(q[pos]), 'quality_pred':int(bool(violations)),
                'violation_type':official_violation_type(violations),
                'criterion_states':json.dumps(decision['criterion_states'],sort_keys=True),
                'criterion_scores':json.dumps(raw_scores,sort_keys=True),
                'criterion_thresholds':json.dumps(spec['thresholds'],sort_keys=True),
                'quality_threshold':cut,'decision_version':'research-nested-council-1',
                'decision_reason':'council_conditional_argmax_review' if fallback[pos] else 'council_quality_gated_types'}.items():
                table.at[row_index,key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT/'labels/image_labels.csv')
    parser.add_argument('--reference', type=Path, default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15/quality-gated-own-roi/candidate_oof.csv'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new council output directory')
    manifest = json.loads((args.bank/'manifest.json').read_text()); require_validation_bank(manifest)
    if sha256(args.bank/'bank.joblib') != manifest['bank_sha256'] or sha256(args.labels) != manifest['labels_sha256']:
        raise ValueError('Bank or source labels differ')
    bank = joblib.load(args.bank/'bank.joblib')
    labels = pd.read_csv(args.labels); labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = bank['baseline'].sort_values('index').reset_index(drop=True)
    validate_outer(labels, baseline)
    reference = pd.read_csv(args.reference).sort_values('index').reset_index(drop=True)
    identity = ['index','fold','study','source_path','quality_true']
    if not reference[identity].equals(baseline[identity]): raise ValueError('Reference identity differs')
    matrices, features_audit = [], []
    for path in map(Path, FEATURES):
        values, metadata = load_feature_bundle(path, labels, manifest['source_fingerprint'], manifest['labels_sha256'])
        matrices.append(values)
        features_audit.append({'path':str(path),'sha256':sha256(path),'metadata_sha256':sha256(path.with_suffix('.json')),
                               'columns':values.shape[1],'encoder_finetuned':metadata['encoder_finetuned']})
    columns = sorted({key for row in bank['features'] for key in row})
    geometry = _matrix(bank['features'], columns)
    regions = baseline.predicted_region.ne('spine').to_numpy(int)
    matrices.append(np.column_stack([geometry,regions]))
    studies = labels.study_key.astype(str).to_numpy(); target = labels.quality_class.to_numpy(int)
    angles = np.array([f.get('spine_abs_angle_deg',np.nan) for f in bank['features']],float)
    forced_axis = (regions == 0) & np.isfinite(angles) & (abs(angles)>5.)
    tables = {(p,c):baseline.copy() for p in POLICIES for c in ('global_cut','region_cut')}
    fitted, audit = [], []
    args.output.mkdir(parents=True)
    (args.output/'executed_nested_model_council.py').write_bytes(SOURCE)
    with threadpool_limits(limits=2):
        for fold in sorted(baseline.fold.unique()):
            test = np.flatnonzero(baseline.fold.to_numpy() == fold)
            held = set(studies[test]); train = np.flatnonzero(~np.isin(studies,list(held)))
            safe_target = np.full(len(target),np.nan); safe_target[train] = target[train]
            raw_oof, middle_audit = crossfit_experts(matrices,regions,safe_target,studies,train,4)
            middle = {p:np.full(len(train),np.nan) for p in POLICIES}
            subaudits = []
            for split,(it,iv) in enumerate(GroupKFold(4).split(train,groups=studies[train])):
                tr,va = train[it],train[iv]
                sub_oof, inner_audit = crossfit_experts(matrices,regions,safe_target,studies,tr,3)
                for policy in POLICIES:
                    arbiter = fit_arbiter(policy,sub_oof,regions[tr],target[tr])
                    middle[policy][iv] = predict_arbiter(arbiter,raw_oof[iv],regions[va])
                subaudits.append({'middle_validation_studies':sorted(set(studies[va])), 'expert_subfolds':inner_audit})
                print(f'Council outer {fold}: strict middle {split} complete',flush=True)
            expert_models = fit_experts(matrices,regions,safe_target,train)
            raw_outer = predict_experts(expert_models,matrices,regions,test)
            final_arbiters, cuts_saved = {}, {}
            for policy in POLICIES:
                if not np.isfinite(middle[policy]).all(): raise ValueError('Incomplete arbiter OOF')
                model = fit_arbiter(policy,raw_oof,regions[train],target[train]); final_arbiters[policy]=model
                q = predict_arbiter(model,raw_outer,regions[test])
                # Singleton prediction may not depend on other images in its batch.
                for j in range(len(test)):
                    np.testing.assert_allclose(predict_arbiter(model,raw_outer[j:j+1],regions[test[j:j+1]]),q[j:j+1],rtol=0,atol=1e-12)
                global_cut = quality_cut(target[train],middle[policy],forced_axis[train])
                regional = {g:quality_cut(target[train][regions[train]==g],middle[policy][regions[train]==g],forced_axis[train][regions[train]==g]) for g in (0,1)}
                cuts_saved[policy]={'global_cut':{0:global_cut,1:global_cut},'region_cut':regional}
                for kind,cuts in cuts_saved[policy].items(): apply_predictions(tables[policy,kind],bank,int(fold),q,cuts)
            fitted.append({'fold':int(fold),'train':train,'test':test,'experts':expert_models,'arbiters':final_arbiters,
                'thresholds':cuts_saved,'raw_expert_oof':raw_oof,'raw_expert_outer':raw_outer,'middle_predictions':middle})
            audit.append({'fold':int(fold),'training_studies':sorted(set(studies[train])),'held_out_studies':sorted(held),
                'middle_splits':middle_audit,'subfolds':subaudits,'thresholds':cuts_saved,
                'outer_labels_used':False,'meta_predictions_fully_cross_fitted':True})
            joblib.dump({'folds':fitted,'expert_names':EXPERT_NAMES,'geometry_columns':columns},args.output/'council_bank.joblib')
            dump(args.output/'progress.json',{'completed_outer_folds':len(fitted),'planned_outer_folds':5})
            print(f'Council outer {fold} complete',flush=True)
    dump(args.output/'protocol.json',{'protocol':__doc__,'experts':EXPERT_NAMES,'features':features_audit,'selection_audit':audit,
        'bank_sha256':manifest['bank_sha256'],'code_sha256':SOURCE_SHA,'reference_sha256':sha256(args.reference),
        'clinical_validation':False,'cached_global_oof_used_to_train_meta':False})
    for (policy,cut),table in tables.items():
        directory=args.output/(policy+'-'+cut);directory.mkdir();path=directory/'candidate_oof.csv';table.to_csv(path,index=False)
        metrics=evaluate(args.labels,path,repeats=args.bootstrap)
        screen=compare(args.reference,path,args.labels)
        dump(directory/'evaluation.json',{'protocol':__doc__,'metrics':metrics,'screen':screen,'policy':policy,'cut_policy':cut,
            'code_sha256':SOURCE_SHA,'bank_sha256':manifest['bank_sha256'],'clinical_validation':False,
            'outer_labels_used_for_training':False,'meta_predictions_fully_cross_fitted':True,
            'automatic_promotions':0,'comparison_reference':str(args.reference)})
        print(json.dumps({'policy':policy,'cut':cut,'quality':metrics['overall_quality'],'macro_f1':metrics['criterion_macro_f1']}),flush=True)


if __name__ == '__main__': main()
