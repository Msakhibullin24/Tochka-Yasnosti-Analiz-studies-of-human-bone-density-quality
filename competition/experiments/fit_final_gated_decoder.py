"""Fit deployment thresholds on full-data banks' inner study-OOF scores.

This produces deployment artifacts, never evaluation predictions. Validation
metrics belong to the separately retained nested study-held-out experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_research.common import digest
from experiments.three_iteration_upgrade import fit_decoder, decode_scores
from experiments.quality_gated_decoder import fit_quality_cut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranking_bank', 'validation', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new deployment decoder directory')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (args.bank, args.ranking_bank)]
    for p, metadata in zip((args.bank, args.ranking_bank), manifests):
        if not metadata.get('full_data_fit') or digest(p / 'bank.joblib') != metadata['bank_sha256']:
            raise ValueError('Expected a checked full-data deployment bank')
    if any(manifests[0][key] != manifests[1][key] for key in ('labels_sha256', 'baseline_sha256', 'source_fingerprint')):
        raise ValueError('Full-data bank source identities differ')
    validation = json.loads((args.validation / 'evaluation.json').read_text())
    if validation.get('metrics', {}).get('labels_sha256') != manifests[0]['labels_sha256']:
        raise ValueError('Validation cohort identity differs')
    bank, prior = [joblib.load(p / 'bank.joblib') for p in (args.bank, args.ranking_bank)]
    previous = {(s['fold'], s['group']): s for s in prior['folds']}
    decoders, audit = [], []
    for state in bank['folds']:
        old = previous[state['fold'], state['group']]
        if len(state['test']) or len(old['test']):
            raise ValueError('Deployment fit must not contain evaluation rows')
        for key in ('train', 'quality_truth', 'criterion_truth'):
            np.testing.assert_array_equal(state[key], old[key])
        typed, ranking = fit_decoder(state, 1), fit_decoder(old, 1)
        _, quality = decode_scores(ranking, old['codes'], np.column_stack([old['inner'][c] for c in old['codes']]), old['inner']['quality'])
        cutoff = fit_quality_cut(state, typed, quality, False)
        decoders.append({'fold': state['fold'], 'group': state['group'], 'criterion_decoder': typed,
                         'quality_decoder': ranking, 'quality_cut': cutoff, 'preserve_independent': False})
        audit.append({'group': state['group'], 'training_images': len(state['train']),
                      'quality_cut': cutoff, 'threshold_source': 'four-fold study-OOF predictions within all available training data'})
    args.output.mkdir(parents=True)
    joblib.dump(decoders, args.output / 'decoders.joblib', compress=3)
    report = {'protocol': __doc__, 'bank_sha256': manifests[0]['bank_sha256'],
              'ranking_bank_sha256': manifests[1]['bank_sha256'], 'full_data_fit': True,
              'decoder_sha256': digest(args.output / 'decoders.joblib'), 'validation_evaluation_sha256': digest(args.validation / 'evaluation.json'),
              'training_accuracy_reported': False, 'independent_accuracy_of_final_weights_established': False,
              'fit_audit': audit, 'code_sha256': digest(Path(__file__)), 'clinical_validation': False, 'automatic_promotions': 0}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
