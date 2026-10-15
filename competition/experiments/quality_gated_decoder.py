"""Research joint decoder: overall QC gate followed by conditional type prediction.

When the gate predicts a defect and no type exceeds its independent threshold,
predict the strongest non-axis type relative to its training threshold, marking
that conditional argmax explicitly. The physical >5 degree axis rule always
takes priority. Type errors are evaluated, never treated as confirmed findings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.decision import decide
from dxaqc.model import official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from experiments.verify_metric_candidate import compare
from experiments.three_iteration_upgrade import fit_decoder, decode_scores, utility, dump, require_validation_bank

EXECUTION_SOURCE = Path(__file__).read_text()
EXECUTION_CODE_SHA256 = sha256(Path(__file__))


def gated_decisions(scores, cuts, codes, angles, quality, quality_cut, preserve_independent=False):
    scores, cuts, quality = np.asarray(scores), np.asarray(cuts), np.asarray(quality)
    gated = quality >= quality_cut
    selected = (scores >= cuts) & (np.ones_like(gated[:, None]) if preserve_independent else gated[:, None])
    if 'spine_axis' in codes:
        selected[:, codes.index('spine_axis')] = np.isfinite(angles) & (abs(angles) > 5.)
    conditional = gated & ~selected.any(axis=1)
    allowed = [i for i, code in enumerate(codes) if code != 'spine_axis']
    if not allowed:
        raise ValueError('At least one modeled non-axis type required')
    normalized = scores[:, allowed] / np.maximum(cuts[allowed], 1e-8)
    strongest = np.array(allowed)[normalized.argmax(axis=1)]
    selected[np.flatnonzero(conditional), strongest[conditional]] = True
    return selected, conditional


def fit_quality_cut(state, spec, quality, preserve_independent=False):
    codes = state['codes']
    scores = np.column_stack([state['inner'][c] for c in codes])
    cuts = np.array([spec['thresholds'][c] for c in codes])
    best, chosen = None, None
    for cut in np.r_[np.unique(quality), np.nextafter(1., np.inf)]:
        predicted, _ = gated_decisions(scores, cuts, codes, state['angles'], quality, cut, preserve_independent)
        rank = (*utility(state['quality_truth'], state['criterion_truth'], predicted),
                -int(np.sum((state['quality_truth'] == 1) & ~predicted.any(axis=1))), float(cut))
        if best is None or rank > best:
            best, chosen = rank, float(cut)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranking_bank', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    parser.add_argument('--preserve-independent', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (args.bank, args.ranking_bank)]
    for directory, manifest in zip((args.bank, args.ranking_bank), manifests):
        require_validation_bank(manifest)
        if (manifest['bank_sha256'] != sha256(directory / 'bank.joblib') or
                manifest['labels_sha256'] != sha256(args.labels) or manifest['baseline_sha256'] != sha256(args.baseline)):
            raise ValueError('Bank or label identity differs')
    if (manifests[0]['source_fingerprint'] != manifests[1]['source_fingerprint'] or
            json.loads((args.bank / 'split_audit.json').read_text()) != json.loads((args.ranking_bank / 'split_audit.json').read_text())):
        raise ValueError('Source or study partitions differ')
    bank, prior = [joblib.load(p / 'bank.joblib') for p in (args.bank, args.ranking_bank)]
    reference = {(s['fold'], s['group']): s for s in prior['folds']}
    table = bank['baseline'].copy()
    specifications, audit = [], []
    with threadpool_limits(limits=4):
        for state in bank['folds']:
            old = reference[state['fold'], state['group']]
            for key in ('train', 'test', 'quality_truth', 'criterion_truth'):
                np.testing.assert_array_equal(state[key], old[key])
            spec, qspec = fit_decoder(state, 1), fit_decoder(old, 1)
            _, inner_quality = decode_scores(qspec, old['codes'], np.column_stack([old['inner'][c] for c in old['codes']]), old['inner']['quality'])
            _, quality = decode_scores(qspec, old['codes'], np.column_stack([old['outer'][c] for c in old['codes']]), old['outer']['quality'])
            cut = fit_quality_cut(state, spec, inner_quality, args.preserve_independent)
            scores = np.column_stack([state['outer'][c] for c in state['codes']])
            cuts = np.array([spec['thresholds'][c] for c in state['codes']])
            angles = np.array([bank['features'][i].get('spine_abs_angle_deg')
                               if bank['features'][i].get('spine_abs_angle_deg') is not None else np.nan for i in state['test']])
            selected, fallback = gated_decisions(scores, cuts, state['codes'], angles, quality, cut, args.preserve_independent)
            for pos, row_index in enumerate(state['test']):
                raw_scores = dict(zip(state['codes'], map(float, scores[pos])))
                decision = decide(state['group'], float(quality[pos]), raw_scores, bank['features'][row_index], cut, spec['thresholds'])
                for j, code in enumerate(state['codes']):
                    if code != 'spine_axis':
                        decision['criterion_states'][code] = {
                            'status': 'fail' if selected[pos, j] else 'pass',
                            'basis': 'quality_conditioned_argmax' if fallback[pos] and selected[pos, j] else
                                     'criterion_threshold_preserved' if args.preserve_independent else 'quality_gated_criterion',
                            'criterion_score': raw_scores[code], 'independent_threshold': float(cuts[j]),
                            'clinical_validation': False}
                violations = [c for j, c in enumerate(state['codes']) if selected[pos, j]]
                for key, value in {'quality_score': float(quality[pos]), 'quality_pred': int(bool(violations)),
                        'violation_type': official_violation_type(violations), 'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                        'criterion_scores': json.dumps(raw_scores, sort_keys=True), 'criterion_thresholds': json.dumps(spec['thresholds'], sort_keys=True),
                        'quality_threshold': cut, 'decision_reason': 'quality_conditioned_argmax_review' if fallback[pos] else 'quality_gated_types',
                        'decision_version': 'research-quality-gated-1'}.items():
                    table.at[row_index, key] = value
            audit.append({'fold': state['fold'], 'group': state['group'], 'quality_cut': cut,
                          'conditional_argmax_predictions': int(fallback.sum())})
            specifications.append({'fold': state['fold'], 'group': state['group'], 'criterion_decoder': spec,
                                   'quality_decoder': qspec, 'quality_cut': cut, 'preserve_independent': args.preserve_independent})
    args.output.mkdir(parents=True)
    (args.output / 'executed_quality_gated_decoder.py').write_text(EXECUTION_SOURCE)
    joblib.dump(specifications, args.output / 'decoders.joblib')
    path = args.output / 'candidate_oof.csv'
    table.to_csv(path, index=False)
    metrics = evaluate(args.labels, path, repeats=2000)
    screen = compare(args.baseline, path, args.labels)
    dump(args.output / 'evaluation.json', {'protocol': __doc__, 'metrics': metrics, 'screen': screen,
         'selection_audit': audit, 'bank_sha256': manifests[0]['bank_sha256'], 'ranking_bank_sha256': manifests[1]['bank_sha256'],
         'code_sha256': EXECUTION_CODE_SHA256, 'clinical_validation': False, 'automatic_promotions': 0})
    print(json.dumps({'metrics': metrics['overall_quality'], 'criterion_macro_f1': metrics['criterion_macro_f1'],
                      'conditional_argmax_predictions': sum(a['conditional_argmax_predictions'] for a in audit)}), flush=True)


if __name__ == '__main__':
    main()
