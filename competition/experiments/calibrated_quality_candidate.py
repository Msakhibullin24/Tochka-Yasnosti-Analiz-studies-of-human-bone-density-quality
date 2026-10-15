"""Nested training-only calibration; preserve the selected QC/type decisions.

Calibrators use only the current outer fold's inner-OOF scores and training
truths. Outer labels are used later by the evaluator, never by calibration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_research.common import digest
from experiments.three_iteration_upgrade import fit_decoder, decode_scores, require_validation_bank
from experiments.quality_score_calibration import MODES, fit_calibration, calibrate, calibrated_cut
from experiments.verify_metric_candidate import compare
from evaluate_organizer_dataset import evaluate

EXECUTION_SOURCE = Path(__file__).read_bytes()
EXECUTION_SHA256 = digest(Path(__file__))
DEPENDENCY_SOURCE = Path(__file__).with_name('quality_score_calibration.py').read_bytes()
DEPENDENCY_SHA256 = digest(Path(__file__).with_name('quality_score_calibration.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranking_bank', 'candidate', 'output'):
        parser.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT/'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT/'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new calibration directory')
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (args.bank, args.ranking_bank)]
    for path, metadata in zip((args.bank, args.ranking_bank), manifests):
        require_validation_bank(metadata)
        if (metadata['bank_sha256'] != digest(path/'bank.joblib') or metadata['labels_sha256'] != digest(args.labels)
                or metadata['baseline_sha256'] != digest(args.baseline)):
            raise ValueError('Bank or cohort identity differs')
    if any(manifests[0][k] != manifests[1][k] for k in ('source_fingerprint','labels_sha256','baseline_sha256')):
        raise ValueError('Source identities differ')
    prior_evaluation = json.loads((args.candidate/'evaluation.json').read_text())
    for name, metadata in zip(('bank_sha256','ranking_bank_sha256'), manifests):
        if prior_evaluation[name] != metadata['bank_sha256']: raise ValueError('Candidate head identities differ')
    table = pd.read_csv(args.candidate/'candidate_oof.csv')
    bank, ranking = [joblib.load(p/'bank.joblib') for p in (args.bank,args.ranking_bank)]
    previous = {(s['fold'],s['group']):s for s in ranking['folds']}
    all_angles = np.array([f.get('spine_abs_angle_deg') if f.get('spine_abs_angle_deg') is not None else np.nan
                           for f in bank['features']])
    specifications, audits = [], []
    with threadpool_limits(limits=4):
        for fold in sorted({s['fold'] for s in bank['folds']}):
            groups, scores, targets, angles, fold_states = [], [], [], [], []
            for state in [s for s in bank['folds'] if s['fold'] == fold]:
                old = previous[fold,state['group']]
                for key in ('train','test','quality_truth'):
                    np.testing.assert_array_equal(state[key],old[key])
                spec = fit_decoder(old,1)
                _, inner = decode_scores(spec,old['codes'],np.column_stack([old['inner'][c] for c in old['codes']]),old['inner']['quality'])
                _, outer = decode_scores(spec,old['codes'],np.column_stack([old['outer'][c] for c in old['codes']]),old['outer']['quality'])
                np.testing.assert_allclose(table.loc[state['test'],'quality_score'],outer,atol=1e-12,rtol=0)
                scores.extend(inner); targets.extend(state['quality_truth']); angles.extend(state['angles'])
                groups.extend([state['group']]*len(inner)); fold_states.append((state,outer))
            calibration = fit_calibration(scores,groups,angles,targets,args.mode)
            specifications.append({'fold':fold,'calibration':calibration})
            for state,outer in fold_states:
                selected = state['test']; group = state['group']; angle = all_angles[selected]
                calibrated = calibrate(calibration,outer,[group]*len(selected),angle)
                cuts = table.loc[selected,'quality_threshold'].to_numpy(float)
                new_cuts = np.array([calibrated_cut(calibration,cut,group,a) for cut,a in zip(cuts,angle)])
                np.testing.assert_array_equal(outer >= cuts,calibrated >= new_cuts)
                table.loc[selected,'quality_score'] = calibrated
                table.loc[selected,'quality_threshold'] = new_cuts
                table.loc[selected,'decision_version'] = 'research-probability-calibration-1'
            audits.append({'fold':fold,'training_images':len(targets),'outer_labels_used_for_fit':False,
                           'mode':args.mode,'positive_score_slopes':True})
    args.output.mkdir(parents=True)
    (args.output/'executed_calibrated_quality_candidate.py').write_bytes(EXECUTION_SOURCE)
    (args.output/'executed_quality_score_calibration.py').write_bytes(DEPENDENCY_SOURCE)
    joblib.dump(specifications,args.output/'calibrators.joblib',compress=3)
    output = args.output/'candidate_oof.csv';table.to_csv(output,index=False)
    metrics = evaluate(args.labels,output,repeats=2000)
    screen = compare(args.baseline,output,args.labels)
    truth = bank['baseline']['quality_true'].to_numpy(int)
    probability = table['quality_score'].to_numpy(float)
    report = {'protocol':__doc__,'metrics':metrics,'screen':screen,'mode':args.mode,'selection_audit':audits,
        'probability_metrics':{'brier':brier_score_loss(truth,probability),'log_loss':log_loss(truth,probability)},
        'bank_sha256':manifests[0]['bank_sha256'],'ranking_bank_sha256':manifests[1]['bank_sha256'],
        'input_candidate_sha256':digest(args.candidate/'candidate_oof.csv'),'code_sha256':EXECUTION_SHA256,
        'dependency_code_sha256':{'quality_score_calibration.py':DEPENDENCY_SHA256},
        'clinical_validation':False,'automatic_promotions':0,'class_and_type_decisions_changed':False}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'mode':args.mode,'f1':metrics['overall_quality']['f1'],'roc_auc':metrics['overall_quality']['roc_auc'],
        'macro_f1':metrics['criterion_macro_f1'],'probability_metrics':report['probability_metrics']}),flush=True)


if __name__ == '__main__': main()
