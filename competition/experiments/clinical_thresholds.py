"""Unfitted clinical-margin thresholds for the two coverage criteria.

label_visibility.py showed the opposite of the usual problem: hip_roi_coverage
separates at AUC 0.939 and spine_coverage at 0.841, yet they score F1 0.17-0.46.
The signal is there and the cut cannot be placed, because a 6- or 7-positive
criterion inside a ~200-row training fold leaves an ~18-study inner split.

The AND-gated oracle experiment then failed for a specific reason: the feature and
the threshold were both chosen on that same 18-study split, so inner gains did not
survive the outer folds (false positives 34 -> 112).

This script removes the fitting entirely. ISCD and the competition's own violation
texts specify the margins numerically - 3 cm top and bottom, 2 cm lateral - so the
threshold is a published constant, not an estimate. Nothing here is fitted to the
249 labels, which makes it immune to the failure above by construction.

Every threshold is also reported alongside the shipped fitted one so the two can be
compared on identical rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.dicom_io import read_any  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from train import build_table  # noqa: E402
from gpu_research.common import dump  # noqa: E402

# ISCD / competition violation text: 3 cm above and below, 2 cm lateral.
CLINICAL_MM = 30.0
CLINICAL_LATERAL_MM = 20.0


def auc(y, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], score[ok]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path,
                        default=HERE / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    features, _, _ = build_table(args.dataset, labels, HERE / '.cache', fingerprint)
    features = [dict(f) for f in features]
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    shipped = {}
    for row in baseline.itertuples():
        states = json.loads(row.criterion_states)
        shipped[row.index] = states

    def column(name):
        return np.array([f.get(name, np.nan) for f in features], float)

    rules = {
        'hip_roi_coverage': {
            'definition': 'fail if any required margin is below the published minimum',
            'clinical': {'ischium_bottom_margin_mm': ('lt', CLINICAL_MM),
                         'troch_top_margin_mm': ('lt', CLINICAL_MM),
                         'lateral_margin_mm': ('lt', CLINICAL_LATERAL_MM)},
            'region': ('hip_left', 'hip_right')},
        'spine_coverage': {
            'definition': 'fail if the iliac crest tops or half of Th12 are not visualised',
            'clinical': {'crest_min_frac': ('lt', 0.0),
                         'crest_height_mm': ('lt', 0.0)},
            'region': ('spine',)},
    }

    report = {'protocol': __doc__, 'published_constants_mm': {'top_bottom': CLINICAL_MM,
                                                              'lateral': CLINICAL_LATERAL_MM}, 'criteria': {}}
    for criterion, spec in rules.items():
        target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
        mask = labels.region.isin(spec['region']).to_numpy() & np.isfinite(target)
        y = (target[mask] == 1).astype(int)
        entry = {'n': int(mask.sum()), 'positives': int(y.sum()), 'measurements': {}}
        # Individual measurements: how well does each one separate on its own?
        for name in spec['clinical']:
            values = column(name)[mask]
            direction, _ = spec['clinical'][name]
            score = -values if direction == 'lt' else values
            entry['measurements'][name] = {'auc': auc(y, score),
                                           'min_mm_among_positives': float(np.nanmin(values[y == 1])) if (y == 1).any() else None,
                                           'max_mm_among_negatives': float(np.nanmax(values[y == 0])) if (y == 0).any() else None,
                                           'positives_below_constant': int((values[y == 1] < (
                                               CLINICAL_LATERAL_MM if name == 'lateral_margin_mm' else CLINICAL_MM)).sum())
                                           if (y == 1).any() else 0}
        # The combined unfitted rule.
        combined = np.zeros(int(mask.sum()), bool)
        any_measurable = False
        for name, (direction, limit) in spec['clinical'].items():
            values = column(name)[mask]
            usable = np.isfinite(values)
            if usable.sum() < len(values) * 0.5:
                continue
            any_measurable = True
            if name in ('crest_min_frac', 'crest_height_mm'):
                # A crest must be present and have positive visible height.
                combined |= (values > limit)
            else:
                combined |= (values < limit)
        entry['clinical_rule_applied'] = any_measurable
        entry['clinical_rule_positives_caught'] = int((combined & (y == 1)).sum()) if any_measurable else None
        entry['clinical_rule_negatives_flagged'] = int((combined & (y == 0)).sum()) if any_measurable else None
        entry['clinical_rule_f1'] = (float(2 * (combined & (y == 1)).sum() /
                                          max(2 * (combined & (y == 1)).sum() + (y == 1).sum() - (combined & (y == 1)).sum()
                                              + (combined & (y == 0)).sum(), 1)) if any_measurable else None)
        # The shipped pipeline's verdict on the identical rows.
        shipped_fail = np.array([shipped[i].get(criterion, {}).get('status') == 'fail'
                                 for i in np.flatnonzero(mask)])
        tp = int((shipped_fail & (y == 1)).sum())
        fp = int((shipped_fail & (y == 0)).sum())
        entry['shipped_fitted_threshold'] = {
            'positives_caught': tp, 'negatives_flagged': fp,
            'f1': float(2 * tp / max(2 * tp + (y == 1).sum() - tp + fp, 1))}
        report['criteria'][criterion] = entry

    dump(args.output / 'clinical_threshold_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()