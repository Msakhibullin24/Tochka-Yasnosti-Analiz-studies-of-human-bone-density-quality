"""Select criterion experts inside each training fold while freezing incumbent QC.

Primary replacement margin is 0.05 inner gated type-F1; zero margin is a control.
No outer labels enter expert selection. Previously inspected, conditional-routing
cohort: these results are exploratory and cannot establish independent accuracy.
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
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_organizer_dataset import evaluate, sha256
from experiments.nested_criterion_upgrade import validate_outer
from experiments.specialist_model_council import NAMES, write_decisions
from experiments.three_iteration_upgrade import dump, fit_decoder, decode_scores, require_validation_bank
from experiments.quality_gated_decoder import fit_quality_cut
from experiments.verify_metric_candidate import compare

SOURCE = Path(__file__).read_bytes()
# IQA has no anatomical target; exclude it before evaluating this experiment.
EXPERT_IDS = (0, 1, 2, 3, 4, 6, 7)


def choose_expert(truth, incumbent, incumbent_cut, candidates, gate, margin):
    """Input contains outer-training OOF rows only; retain default on ties."""
    known = np.isfinite(truth)
    y = truth[known].astype(int)
    gate = np.asarray(gate, bool)[known]
    def score(values, cut):
        return float(f1_score(y, (values[known] >= cut) & gate, zero_division=0))
    baseline = score(incumbent, incumbent_cut)
    baseline_ap = float(average_precision_score(y, incumbent[known])) if y.any() else 0.
    best = None
    for serial, (name, values) in enumerate(candidates.items()):
        if not np.isfinite(values).all():
            raise ValueError('Incomplete inner expert predictions')
        cuts = np.r_[np.unique(values[known]), np.nextafter(1., np.inf)]
        cut = max(cuts, key=lambda t: (score(values, t), float(t)))
        current = score(values, cut)
        ap = float(average_precision_score(y, values[known])) if y.any() else 0.
        if ap < baseline_ap:
            continue
        rank = (current, ap, -serial)
        if best is None or rank > best[0]:
            best = (rank, name, float(cut))
    if best is not None and best[0][0] > baseline + margin:
        return {'source': best[1], 'cut': best[2], 'inner_f1': best[0][0],
                'incumbent_inner_f1': baseline, 'margin': margin}
    return {'source': 'incumbent', 'cut': float(incumbent_cut), 'inner_f1': baseline,
            'incumbent_inner_f1': baseline, 'margin': margin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15/bank-own-roi-only'))
    parser.add_argument('--ranking-bank', type=Path, default=Path('data/gpu-runs/three-iterations-2026-10-15/bank'))
    parser.add_argument('--specialists', type=Path, default=Path('data/gpu-runs/specialist-council-2026-10-15/specialist_bank.joblib'))
    parser.add_argument('--labels', type=Path, default=ROOT/'labels/image_labels.csv')
    parser.add_argument('--reference', type=Path, default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15/quality-gated-own-roi/candidate_oof.csv'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    banks = []
    for directory in (args.bank, args.ranking_bank):
        manifest = json.loads((directory/'manifest.json').read_text())
        require_validation_bank(manifest)
        if sha256(directory/'bank.joblib') != manifest['bank_sha256'] or sha256(args.labels) != manifest['labels_sha256']:
            raise ValueError('Bank identity differs')
        banks.append(joblib.load(directory/'bank.joblib'))
    bank, prior = banks
    specialists = joblib.load(args.specialists)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    base = bank['baseline']
    validate_outer(labels, base)
    np.testing.assert_array_equal(base['index'], np.arange(len(labels)))
    reference = pd.read_csv(args.reference).sort_values('index').reset_index(drop=True)
    identity = ['index', 'repeat', 'fold', 'study', 'source_path', 'quality_true']
    if not base[identity].equals(reference[identity]):
        raise ValueError('Reference identity differs')
    args.output.mkdir(parents=True)
    (args.output/'executed_locked_gate_specialists.py').write_bytes(SOURCE)
    protocol = {'protocol': __doc__, 'primary': 'margin005', 'margins': [.05, 0.],
                'code_sha256': hashlib.sha256(SOURCE).hexdigest(),
                'reference_sha256': sha256(args.reference),
                'specialists_sha256': sha256(args.specialists),
                'labels_sha256': sha256(args.labels),
                'clinical_validation': False, 'deployment_ready': False,
                'quality_scores_and_predictions_frozen': True, 'automatic_promotions': 0}
    dump(args.output/'protocol.json', protocol)
    by_key = lambda value: {(s['fold'], s['group']): s for s in value['folds']}
    old_states, expert_states = by_key(prior), by_key(specialists)
    tables = {margin: reference.copy() for margin in (.05, 0.)}
    audit = []
    for state in bank['folds']:
        fold, group, train, test = state['fold'], state['group'], state['train'], state['test']
        old, expert = old_states[fold, group], expert_states[fold, group]
        for key in ('train', 'test'):
            np.testing.assert_array_equal(state[key], old[key])
            np.testing.assert_array_equal(state[key], expert[key])
        spec, qspec = fit_decoder(state, 1), fit_decoder(old, 1)
        codes = state['codes']
        _, qi = decode_scores(qspec, codes, np.column_stack([old['inner'][c] for c in codes]), old['inner']['quality'])
        cut = fit_quality_cut(state, spec, qi)
        np.testing.assert_allclose(reference.loc[test, 'quality_threshold'], cut, rtol=0, atol=1e-12)
        # Keep the exact serialized incumbent value, including float round-trip.
        cut = float(reference.loc[test[0], 'quality_threshold'])
        for margin, table in tables.items():
            predictions = {'quality': reference.loc[test, 'quality_score'].to_numpy()}
            thresholds = dict(spec['thresholds'])
            for code in codes:
                if code == 'spine_axis':
                    predictions[code] = state['outer'][code]
                    continue  # Never learn the physical five-degree definition.
                raw_inner = expert['raw_expert_oof'][code]
                raw_outer = expert['raw_expert_outer'][code]
                candidates = {NAMES[i]: raw_inner[:, i] for i in EXPERT_IDS}
                queries = {NAMES[i]: raw_outer[:, i] for i in EXPERT_IDS}
                for name, ids in (('mean_all', range(8)), ('mean_visual', range(5))):
                    candidates[name] = raw_inner[:, list(ids)].mean(axis=1)
                    queries[name] = raw_outer[:, list(ids)].mean(axis=1)
                choice = choose_expert(state['criterion_truth'][:, codes.index(code)],
                                       state['inner'][code], thresholds[code], candidates, qi >= cut, margin)
                source = choice['source']
                predictions[code] = (state['outer'][code] if source == 'incumbent' else
                                     queries[source])
                thresholds[code] = choice['cut']
                audit.append({'fold': int(fold), 'group': group, 'criterion': code, **choice})
            write_decisions(table, bank['features'], group, test, predictions, codes,
                            {'alpha': 1., 'quality_cut': cut}, thresholds)
            # Preserve the incumbent type fallback when new experts are silent.
            fallback = table.loc[test, 'decision_reason'].eq('specialist_council_argmax_review')
            fallback_ids = np.asarray(test)[fallback.to_numpy()]
            type_columns = ['violation_type', 'criterion_scores', 'criterion_states',
                            'criterion_thresholds', 'decision_reason', 'decision_version']
            table.loc[fallback_ids, type_columns] = reference.loc[fallback_ids, type_columns]
            if group == 'spine':
                for index in test:
                    states = json.loads(table.at[index, 'criterion_states'])
                    states['spine_axis'] = json.loads(reference.at[index, 'criterion_states'])['spine_axis']
                    table.at[index, 'criterion_states'] = json.dumps(states, sort_keys=True)
    dump(args.output/'selection_audit.json', audit)
    for margin, table in tables.items():
        for column in ('quality_score', 'quality_pred', 'quality_threshold'):
            np.testing.assert_array_equal(reference[column], table[column])
        folder = args.output/('margin005' if margin else 'margin0')
        folder.mkdir()
        path = folder/'candidate_oof.csv'
        table.to_csv(path, index=False)
        metrics = evaluate(args.labels, path, repeats=2000)
        result = {**protocol, 'margin': margin, 'metrics': metrics,
                  'screen': compare(args.reference, path, args.labels)}
        dump(folder/'evaluation.json', result)
        print(json.dumps({'margin': margin, 'f1': metrics['overall_quality']['f1'],
                          'auc': metrics['overall_quality']['roc_auc'],
                          'macro_f1': metrics['criterion_macro_f1']}), flush=True)


if __name__ == '__main__':
    main()
