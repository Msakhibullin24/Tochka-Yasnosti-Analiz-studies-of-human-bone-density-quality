"""Evaluate raw saved-head replay and compare it with the cached-feature experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_organizer_dataset import evaluate, sha256
from experiments.verify_metric_candidate import compare
from source_integrity import inspect_sources
from dxaqc.model import official_violation_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'predictions', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    source = inspect_sources([('organiser', args.labels, args.dataset)])
    source_lookup = {e['path']: e for e in source['entries']}
    records = json.loads(args.predictions.read_text())
    lookup = {r['decoded_pixel_sha256']: r for r in records}
    if len(lookup) != len(records):
        raise ValueError('Duplicate raw image inference records')
    table = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    for i, row in enumerate(labels.itertuples()):
        record = lookup[source_lookup[row.first_source_path]['pixel_sha256_current']]
        if record['folds'] != [int(table.iloc[i].fold)]:
            raise ValueError('Raw prediction is not from the held-out study fold')
        for key, value in {'quality_score': record['score'], 'quality_pred': record['quality'],
                'predicted_region': record['region'], 'violation_type': official_violation_type(record['violations']),
                'criterion_scores': json.dumps(record['criterion_scores'], sort_keys=True),
                'criterion_states': json.dumps(record['criterion_states'], sort_keys=True),
                'criterion_thresholds': json.dumps(record['criterion_thresholds'], sort_keys=True),
                'quality_threshold': float(np.nextafter(1., np.inf)),
                'decision_reason': 'criterion_failure' if record['quality'] else 'no_detected_violation',
                'decision_version': 'research-raw-live-anatomy-1'}.items():
            table.at[i, key] = value
    args.output.mkdir(parents=True)
    path = args.output / 'candidate_oof.csv'
    table.to_csv(path, index=False)
    reference = pd.read_csv(args.reference).sort_values('index').reset_index(drop=True)
    metrics = evaluate(args.labels, path, repeats=2000)
    screen = compare(args.baseline, path, args.labels)
    audit = {'raw_images_inferred': len(records), 'labelled_images_evaluated': len(table),
             'same_class_as_cached_experiment': int(np.sum(table.quality_pred.to_numpy() == reference.quality_pred.to_numpy())),
             'same_type_as_cached_experiment': int(np.sum(table.violation_type.fillna('').to_numpy() == reference.violation_type.fillna('').to_numpy())),
             'same_region_as_historical_routing': int(np.sum(table.predicted_region.to_numpy() == reference.predicted_region.to_numpy())),
             'max_absolute_quality_score_drift': float(np.max(abs(table.quality_score - reference.quality_score)))}
    (args.output / 'evaluation.json').write_text(json.dumps({'metrics': metrics, 'screen': screen,
        'raw_replay_audit': audit, 'code_sha256': sha256(Path(__file__)),
        'raw_predictions_sha256': sha256(args.predictions), 'cached_reference_sha256': sha256(args.reference),
        'clinical_validation': False, 'automatic_promotions': 0,
        'limitations': ['Previously inspected studies; raw replay is not independent new validation.',
                        'Study-exclusive QC heads and decoders; baseline region/projection components remain frozen.',
                        'Default five-fold deployment ensemble has not been evaluated on a new cohort.']}, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'metrics': metrics['overall_quality'], 'audit': audit}), flush=True)


if __name__ == '__main__':
    main()
