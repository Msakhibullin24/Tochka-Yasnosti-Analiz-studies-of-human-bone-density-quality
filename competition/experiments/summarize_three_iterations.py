"""Aggregate the fixed October 15 experiment artifacts without choosing a model.

Run from the repository root after the three QC and two anatomy probes.
"""
import hashlib,json,sys,statistics
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,f1_score
root=Path.cwd(); base=root/'data/gpu-runs/three-iterations-2026-10-15'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
items=[]
for n in (1,2,3):
 p=base/f'iteration-{n}/evaluation.json'; x=json.loads(p.read_text())
 items.append({'iteration':n, 'metrics':x['metrics'], 'selection_audit':x['selection_audit'],
 'paired_delta_f1_ci95':x['screen']['paired_delta_f1_ci95'],
 'screen_passed':x['screen']['passes_internal_metric_screen'],
 'screen_reasons':x['screen']['failure_reasons'], 'evaluation_sha256':sha(p),
 'candidate_csv_sha256':sha(p.parent/'candidate_oof.csv'), 'code_sha256_at_execution':x['code_sha256']})
first=json.loads((base/'iteration-1/evaluation.json').read_text())
probe=json.loads((base/'tsxr-spine-all/probe.json').read_text()); labels=pd.read_csv('competition/labels/image_labels.csv').set_index('pixel_sha256')
anatomy={}
for pol in ('bone-bright','inverted'):
 r=[a for a in probe['results'] if a['polarity']==pol];y=np.array([labels.loc[a['pixel_sha256'],'spine_axis'] for a in r]);s=np.array([abs(a['candidate_axis_degrees']) if a['candidate_axis_degrees'] is not None else 0 for a in r])
 anatomy[pol]={'images':len(r),'complete_l1_l4':sum(a['complete_l1_l4'] for a in r),'ordered_l1_l4':sum(a['ordered_l1_l4'] for a in r), 'median_inference_seconds':statistics.median(a['seconds'] for a in r), 'axis_fixed_5deg':{'tn_fp_fn_tp':confusion_matrix(y,s>5).ravel().tolist(), 'f1':f1_score(y,s>5),'roc_auc_abs_angle_missing_zero':roc_auc_score(y,s)}}
public=json.loads((base/'tsxr-public/evaluation.json').read_text())
old=json.loads(Path('docs/competition/dxa_spine_numbered_roi_cv_2026_09_28.json').read_text())
result={'date':'2026-10-15','schema_version':1,'iterations_completed':3,'clinical_validation':False,
 'requirements_complete':False,'model_promotions':0,'release':'1.11.6',
 'protocol':{'images':249,'study_groups':100,'outer_folds':5,'inner_group_folds':4,'seed':17,'inner_selection_only':True,'singleton_batch_assertions_passed':True,'routing_and_baseline_angles_frozen':True,'new_labels_used':False,'patient_disjoint_established':False,'previously_inspected_cohort':True},
 'bank_manifest':json.loads((base/'bank/manifest.json').read_text()),
 'baseline_metrics':first['screen']['overall_score_metrics']['baseline'],
 'baseline_criterion_macro_f1':first['screen']['criterion_macro_f1']['baseline'],
 'iterations':items,
 'tsxr':{'model_source':'https://zenodo.org/records/17052912','code_source':'https://github.com/risc-mi/totalsegmentator2D',
 'weights_license':'CC-BY-NC-4.0','archive_bytes':250997762,'archive_md5':'0cebb83dd922ed7be9e4d7a89d2a7979', 'checkpoint_sha256':probe['checkpoint_sha256'],'organizer_anatomy':anatomy,
 'public_roi_comparison':public, 'historical_roi_40':{'model_dice':old['mean_prediction_dice_40'],'layout_prior_dice':old['mean_layout_prior_dice_40'],'same_protocol':False},
 'limitations':['Ordered predicted labels do not establish numbering accuracy on organizer images.',
 'Publisher ROIs differ from projected full anatomical bodies; Dice is an overlap comparison of different targets.',
 'Ten external annotated folders do not establish generalization to all scanners; pretraining overlap not independently audited.',
 'Default physical scale for external images is uncertain; distances are not established millimeter accuracy.',
 'No iliac-crest, hip-trochanter or source-ROI acceptance established.']},
 'verification':{'competition_tests_passed':430,'competition_tests_skipped':5,'log':'data/logs/competition-tests-three-iterations-final-2026-10-15.log'},
 'remaining_required_work':['Supported projection detection across regions including non-target views.', 'Reliable numbered spine and hip anatomy with evidence for required boundaries and scan margins.', 'Implement and verify all required ROI checks for applicable inputs.', 'Clean offline container build/run and target-hardware timing.', 'Frozen reproducible submission for organizer evaluation.']}
inventory=pd.read_csv('data/gpu-runs/chinese-dxa-inventory-2026-10-04/inventory.csv')
inventory=inventory[inventory.parsed].copy()
inventory['person_group']=inventory.patient.str[8:].str.casefold()
result['external_corpus_integrity']={'legacy_date_name_keys':int(inventory.patient.nunique()),
 'conservative_name_groups':int(inventory.person_group.nunique()),
 'name_groups_with_multiple_legacy_keys':int((inventory.groupby('person_group').patient.nunique()>1).sum()),
 'new_diagnostic_metrics_claimed':False,'patient_identity_verified':False}
# Paired study resampling of ROC-AUC/AP; no model is selected from bootstrap samples.
before=pd.read_csv('competition/reports/quality_review_axis_guard_posthoc_2026_09_29.csv').sort_values('index'); after=pd.read_csv(base/'iteration-1/candidate_oof.csv').sort_values('index')
y=before.quality_true.to_numpy(int);b=before.quality_score.to_numpy();a=after.quality_score.to_numpy();groups=before.study.astype(str).to_numpy();units=[np.flatnonzero(groups==g) for g in np.unique(groups)];rng=np.random.default_rng(17);ds=[]
for _ in range(2000):
 ix=np.concatenate([units[j] for j in rng.integers(len(units),size=len(units))]); yy=y[ix]
 if len(np.unique(yy))<2:continue
 ds.append([roc_auc_score(yy,a[ix])-roc_auc_score(yy,b[ix]),average_precision_score(yy,a[ix])-average_precision_score(yy,b[ix])])
result['iteration_1_paired_score_delta_ci95']={'resamples':len(ds),'roc_auc':np.percentile(np.array(ds)[:,0],[2.5,97.5]).tolist(),'average_precision':np.percentile(np.array(ds)[:,1],[2.5,97.5]).tolist()}
out=Path('docs/competition/three_iterations_2026_10_15.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(out);print(result['iteration_1_paired_score_delta_ci95'])
