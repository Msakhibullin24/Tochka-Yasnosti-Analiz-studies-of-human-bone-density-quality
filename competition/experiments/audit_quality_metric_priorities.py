"""Audit DXA QC errors and a conservative alarm gate on frozen OOF rows.

This is a post hoc diagnostic. It never trains a model or edits release policy.
Only aggregate DICOM identity statistics are written; patient fields are not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from dxaqc.model import SEED, VIOLATION_LABEL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def confusion(truth: np.ndarray, predicted: np.ndarray) -> dict:
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tp = int(((truth == 1) & (predicted == 1)).sum())
    return {
        'tn_fp_fn_tp': [tn, fp, fn, tp],
        'f1': 2 * tp / max(1, 2 * tp + fp + fn),
        'sensitivity': tp / max(1, tp + fn),
        'specificity': tn / max(1, tn + fp),
        'false_safe_rate_among_auto_normal': fn / max(1, tn + fn),
    }


def identity_audit(labels: pd.DataFrame, root: Path) -> dict:
    """Count tag usefulness for grouping without retaining their values."""
    fields = ('PatientID', 'PatientName', 'PatientBirthDate', 'AccessionNumber',
              'StudyInstanceUID')
    groups: dict[str, dict[str, set[str]]] = {field: defaultdict(set) for field in fields}
    observed = Counter()
    logger = logging.getLogger('pydicom')
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        for row in labels.itertuples():
            path = (root / row.first_source_path).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('Source DICOM path is missing or escapes dataset root')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
                ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(fields))
                for field in fields:
                    value = str(ds.get(field, '')).strip()
                    if value:
                        observed[field] += 1
                        groups[field][value].add(str(row.study_key))
    finally:
        logger.setLevel(old_level)
    return {
        field: {
            'nonempty_images': observed[field],
            'distinct_values': len(groups[field]),
            'values_spanning_multiple_studies': sum(len(studies) > 1 for studies in groups[field].values()),
            'max_studies_per_value': max((len(studies) for studies in groups[field].values()), default=0),
            'usable_to_prove_patient_disjointness': False,
        }
        for field in fields
    }


def audit(oof_path: Path, labels_path: Path, root: Path) -> dict:
    oof_all = pd.read_csv(oof_path)
    oof = oof_all[oof_all['repeat'] == 0].copy().sort_values('index').reset_index(drop=True)
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    if len(oof) != 249 or len(labels) != 249 or oof['index'].duplicated().any():
        raise ValueError('Expected all 249 labelled organizer DXA images')
    for i, row in oof.iterrows():
        label = labels.iloc[i]
        if (row['index'] != i or row['source_path'] != label.first_source_path
                or str(row['study']) != str(label.study_key)
                or row['true_region'] != label.region
                or int(row['quality_true']) != int(label.quality_class)):
            raise ValueError(f'OOF and reference identities differ at index {i}')
    truth = oof.quality_true.to_numpy(dtype=int)
    predicted = oof.quality_pred.to_numpy(dtype=int)
    if set(np.unique(truth)) != {0, 1} or set(np.unique(predicted)) != {0, 1}:
        raise ValueError('Invalid binary quality values')
    current = confusion(truth, predicted)
    regions = {
        region: confusion(truth[indices], predicted[indices])
        for region in ('spine', 'hip_left', 'hip_right')
        if (indices := np.flatnonzero(oof.true_region.eq(region).to_numpy())).size
    }
    false_safe = (truth == 1) & (predicted == 0)
    false_alarm = (truth == 0) & (predicted == 1)
    criteria = ('spine_coverage', 'spine_axis', 'spine_artifact',
                'hip_position_rotation', 'hip_roi_coverage')
    # Use explicit parentheses for boolean intersections: labels can overlap.
    missed_criteria = {
        criterion: int(((pd.to_numeric(labels[criterion], errors='coerce').fillna(0).to_numpy() == 1)
                        & false_safe).sum())
        for criterion in criteria
    }
    predicted_false_alarm_types = {
        code: int(oof.loc[false_alarm & oof.true_region.isin(
            ['spine'] if code.startswith('spine') else ['hip_left', 'hip_right']).to_numpy(),
            'violation_type'].fillna('').str.contains(label, regex=False).sum())
        for code, label in VIOLATION_LABEL.items() if code in criteria
    }
    explanations = {
        reason: {
            'false_safe': int((false_safe & oof.decision_reason.eq(reason).to_numpy()).sum()),
            'false_alarm': int((false_alarm & oof.decision_reason.eq(reason).to_numpy()).sum()),
        }
        for reason in sorted(oof.decision_reason.dropna().unique())
    }
    # A fixed counterfactual: require the global quality score to confirm a
    # criterion alarm. Preserve any measured-axis type mandated by the TЗ.
    measured_axis = oof.violation_type.fillna('').str.contains(
        VIOLATION_LABEL['spine_axis'], regex=False).to_numpy()
    criterion_alarm = oof.decision_reason.eq('criterion_failure').to_numpy()
    keep = ~criterion_alarm | measured_axis | (
        oof.quality_score.to_numpy(dtype=float) >= oof.quality_threshold.to_numpy(dtype=float))
    gated = predicted & keep.astype(int)
    gated_metrics = confusion(truth, gated)
    if (predicted[measured_axis] != gated[measured_axis]).any():
        raise ValueError('Mandatory measured-axis result changed')
    rng = np.random.default_rng(SEED)
    studies = oof.study.astype(str).to_numpy()
    grouped = [np.flatnonzero(studies == study) for study in np.unique(studies)]
    delta_f1, delta_sensitivity = [], []
    current_draws = {key: [] for key in ('f1', 'sensitivity', 'specificity',
                                         'false_safe_rate_among_auto_normal')}
    regional_f1_draws = {region: [] for region in regions}
    for _ in range(2000):
        sample = np.concatenate([grouped[i] for i in rng.integers(len(grouped), size=len(grouped))])
        before = confusion(truth[sample], predicted[sample])
        after = confusion(truth[sample], gated[sample])
        delta_f1.append(after['f1'] - before['f1'])
        delta_sensitivity.append(after['sensitivity'] - before['sensitivity'])
        for key in current_draws:
            current_draws[key].append(before[key])
        for region in regions:
            region_sample = sample[oof.true_region.to_numpy()[sample] == region]
            regional_f1_draws[region].append(confusion(
                truth[region_sample], predicted[region_sample])['f1'])
    current_ci = {key: [float(v) for v in np.percentile(values, [2.5, 97.5])]
                  for key, values in current_draws.items()}
    for region, draws in regional_f1_draws.items():
        regions[region]['f1_ci95_study_bootstrap'] = [
            float(v) for v in np.percentile(draws, [2.5, 97.5])]
    return {
        'scope': 'Frozen organizer DXA one-repeat OOF; post hoc diagnostic, not independent validation',
        'images': len(oof), 'studies': len(grouped), 'positives': int(truth.sum()),
        'current': current, 'current_ci95_study_bootstrap': current_ci,
        'by_region': regions,
        'false_safe_studies': int(np.unique(studies[false_safe]).size),
        'false_alarm_studies': int(np.unique(studies[false_alarm]).size),
        'missed_positive_criteria_overlapping': missed_criteria,
        'false_alarm_predicted_types_overlapping': predicted_false_alarm_types,
        'errors_by_decision_reason': explanations,
        'counterfactual_global_score_gate': {
            'rule': 'Only criterion_failure: retain if quality_score >= quality_threshold or measured spine-axis type is present',
            'metrics': gated_metrics,
            'paired_delta_f1': gated_metrics['f1'] - current['f1'],
            'paired_delta_f1_ci95': [float(v) for v in np.percentile(delta_f1, [2.5, 97.5])],
            'paired_delta_sensitivity': gated_metrics['sensitivity'] - current['sensitivity'],
            'paired_delta_sensitivity_ci95': [float(v) for v in np.percentile(delta_sensitivity, [2.5, 97.5])],
            'release_candidate': False,
        },
        'dicom_identity_fields': identity_audit(labels, root),
        'patient_disjointness_proven': False,
        'oof_sha256': sha256(oof_path), 'labels_sha256': sha256(labels_path),
        'code_sha256': sha256(Path(__file__)),
        'limitations': [
            'All cases have been used during prior development and candidate selection.',
            'Study groups do not establish patient independence; patient identifiers are uninformative in this export.',
            'Post hoc alarm gate was inspected on the same OOF labels; it cannot justify a release change.',
            'No independent anatomy landmarks, clinical action labels, or projection subtype labels exist here.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new report path')
    report = audit(args.oof, args.labels, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('current', 'by_region', 'false_safe_studies',
                                                   'false_alarm_studies', 'counterfactual_global_score_gate')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
