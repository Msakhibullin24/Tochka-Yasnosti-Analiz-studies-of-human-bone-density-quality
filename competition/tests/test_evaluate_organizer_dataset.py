import csv

import pytest

from evaluate_organizer_dataset import evaluate, evaluate_release


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labelled_rows():
    return [
        {'first_source_path': 'spine.dcm', 'study_key': 'study-a', 'region': 'spine',
         'quality_class': '1', 'spine_coverage': '0', 'spine_axis': '1', 'spine_artifact': '0',
         'hip_position_rotation': '', 'hip_roi_coverage': ''},
        {'first_source_path': 'left.dcm', 'study_key': 'study-b', 'region': 'hip_left',
         'quality_class': '0', 'spine_coverage': '', 'spine_axis': '', 'spine_artifact': '',
         'hip_position_rotation': '0', 'hip_roi_coverage': '0'},
        {'first_source_path': 'right.dcm', 'study_key': 'study-c', 'region': 'hip_right',
         'quality_class': '1', 'spine_coverage': '', 'spine_axis': '', 'spine_artifact': '',
         'hip_position_rotation': '1', 'hip_roi_coverage': '0'},
    ]


def test_release_metrics_keep_untyped_positive_as_type_false_negative(tmp_path):
    labels = tmp_path / 'labels.csv'
    results = tmp_path / 'results.csv'
    write_csv(labels, labelled_rows())
    write_csv(results, [
        {'path_to_file': 'spine.dcm', 'processing_status': 'Success', 'quality_class': '1',
         'quality_prob': '0.9', 'violation_type': ''},
        {'path_to_file': 'left.dcm', 'processing_status': 'Success', 'quality_class': '0',
         'quality_prob': '0.1', 'violation_type': ''},
        {'path_to_file': 'right.dcm', 'processing_status': 'Success', 'quality_class': '1',
         'quality_prob': '0.8', 'violation_type': 'Некорректная укладка'},
    ])
    report = evaluate_release(labels, results)
    assert report['overall_quality']['f1'] == 1
    assert report['positive_without_official_type'] == 1
    assert report['by_criterion']['spine_axis']['tn_fp_fn_tp'] == [0, 0, 1, 0]
    assert report['by_criterion']['hip_position_rotation']['tn_fp_fn_tp'] == [1, 0, 0, 1]


def test_oof_rejects_same_study_in_two_test_folds(tmp_path):
    reference = labelled_rows()
    reference[1]['study_key'] = reference[0]['study_key']
    labels = tmp_path / 'labels.csv'
    oof = tmp_path / 'oof.csv'
    write_csv(labels, reference)
    write_csv(oof, [
        {'source_path': row['first_source_path'], 'study': row['study_key'],
         'true_region': row['region'], 'quality_true': row['quality_class'],
         'quality_pred': row['quality_class'], 'quality_score': '0.5',
         'violation_type': '', 'repeat': '0', 'fold': str(index)}
        for index, row in enumerate(reference)
    ])
    with pytest.raises(ValueError, match='split across test folds'):
        evaluate(labels, oof, repeats=100)
