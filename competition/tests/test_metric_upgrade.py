import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from nested_criterion_upgrade import fit_head, predict_head, select_head, validate_outer
from verify_metric_candidate import compare, read_oof

@pytest.fixture
def cohort(tmp_path):
    """Portable synthetic contract data; no research reports needed in Docker."""
    rows, labels = [], []
    for i in range(249):
        region = 'spine' if i < 99 else 'hip_left' if i < 177 else 'hip_right'
        truth = int(i % 4 == 0)
        name = 'Присутствуют посторонние предметы' if region == 'spine' else 'Некорректная укладка'
        study, path = f'study-{i // 3}', f'images/{i}.dcm'
        rows.append({'index': i, 'repeat': 0, 'fold': (i // 3) % 5, 'study': study,
                     'source_path': path, 'true_region': region, 'predicted_region': region,
                     'quality_true': truth, 'quality_pred': truth, 'quality_score': .9 if truth else .1,
                     'violation_type': name if truth else '', 'criterion_states': '{}', 'criterion_scores': '{}'})
        labels.append({'first_source_path': path, 'study_key': study, 'region': region, 'quality_class': truth,
                       'spine_coverage': 0 if region == 'spine' else np.nan,
                       'spine_axis': 0 if region == 'spine' else np.nan,
                       'spine_artifact': truth if region == 'spine' else np.nan,
                       'hip_position_rotation': truth if region != 'spine' else np.nan,
                       'hip_roi_coverage': 0 if region != 'spine' else np.nan})
    baseline, reference = tmp_path / 'baseline.csv', tmp_path / 'labels.csv'
    pd.DataFrame(rows).to_csv(baseline, index=False)
    reference_table = pd.DataFrame(labels)
    for column in ('spine_coverage', 'spine_axis', 'spine_artifact',
                   'hip_position_rotation', 'hip_roi_coverage'):
        reference_table[column] = reference_table[column].map(lambda v: '' if pd.isna(v) else str(int(v)))
    reference_table.to_csv(reference, index=False)
    return baseline, reference


@pytest.mark.parametrize('field,value,message', [
    ('quality_score', np.nan, 'probabilities'),
    ('quality_score', 1.1, 'probabilities'),
    ('predicted_region', 'other', 'anatomical region'),
])
def test_screen_rejects_invalid_predictions(tmp_path, cohort, field, value, message):
    table = pd.read_csv(cohort[0])
    table.loc[0, field] = value
    path = tmp_path / 'candidate.csv'
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        read_oof(path)


def test_screen_rejects_split_study_and_duplicate_source(tmp_path, cohort):
    table = pd.read_csv(cohort[0])
    repeated = table.index[table.study.eq(table.study.iloc[0])]
    assert len(repeated) > 1
    table.loc[repeated[0], 'fold'] = 99
    path = tmp_path / 'split.csv'
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='split across'):
        read_oof(path)
    table = pd.read_csv(cohort[0])
    table.loc[1, 'source_path'] = table.source_path.iloc[0]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate paths'):
        read_oof(path)


def test_screen_detects_probability_regression_with_identical_decisions(tmp_path, cohort):
    table = pd.read_csv(cohort[0])
    # Same binary answers and types, but deliberately inverted ranking.
    table['quality_score'] = 1 - table.quality_true
    path = tmp_path / 'bad-probabilities.csv'
    table.to_csv(path, index=False)
    report = compare(cohort[0], path, cohort[1])
    assert report['baseline'] == report['candidate']
    assert 'roc_auc_lower' in report['failure_reasons']
    assert 'brier_score_higher' in report['failure_reasons']
    assert not report['passes_internal_metric_screen']


def test_training_never_reads_held_out_labels_and_handles_single_class():
    geometry = np.arange(24).reshape(12, 2)
    embedding = np.arange(36).reshape(12, 3)
    truth = np.array([0.] * 6 + [1.] * 6)
    train, test = np.arange(6), np.arange(6, 12)
    for name in ('geometry_rf', 'geometry_lr', 'embedding_lr_001', 'combined_lr_001', 'baseline_ensemble'):
        head = fit_head(name, geometry, embedding, truth, train)
        assert np.array_equal(predict_head(head, geometry, embedding, test), np.zeros(6))
        altered = truth.copy()
        altered[test] = np.nan
        another = fit_head(name, geometry, embedding, altered, train)
        assert np.array_equal(predict_head(another, geometry, embedding, test), np.zeros(6))


def test_inner_selection_and_outer_identity_validation(cohort):
    name, threshold = select_head(np.array([0., 0., 1., 1.]), {
        'bad': np.array([.9, .8, .1, .2]), 'good': np.array([.1, .2, .8, .9])})
    assert name == 'good' and .2 < threshold < .8
    with pytest.raises(ValueError, match='Incomplete'):
        select_head(np.array([0., 1.]), {'bad': np.array([.1, np.nan])})
    labels = pd.read_csv(cohort[1])
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(cohort[0])
    validate_outer(labels, baseline)
    baseline.loc[0, 'quality_true'] = 1 - baseline.loc[0, 'quality_true']
    with pytest.raises(ValueError, match='quality_true'):
        validate_outer(labels, baseline)
