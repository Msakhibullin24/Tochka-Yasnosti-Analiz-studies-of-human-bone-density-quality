import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('sklearn')

spec = importlib.util.spec_from_file_location('study_baseline', Path(__file__).parents[1] / 'scripts' / 'train_study_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_connected_groups_prevent_transitive_duplicate_leakage():
    records = [
        {'image_id': 'a', 'source_study_key': 'A', 'pixel_sha256': 'same'},
        {'image_id': 'b', 'source_study_key': 'B', 'pixel_sha256': 'same'},
        {'image_id': 'c', 'source_study_key': 'C', 'pixel_sha256': 'other'},
        {'image_id': 'd', 'source_study_key': 'D', 'pixel_sha256': 'separate'},
    ]
    groups = baseline.connected_groups(records, [{'first_image': 'b', 'second_image': 'c'}])
    assert groups['A'] == groups['B'] == groups['C']
    assert groups['D'] != groups['A']


def test_study_targets_preserve_missing_and_use_original_overall_labels():
    rows = [{'source_study_key': k, 'regions': {str(i): {'quality_class': v} for i, v in enumerate(values)}}
            for k, values in [('A', [None, 0, 1]), ('B', [0, None, None]), ('C', [None, None])]]
    assert baseline.study_targets(rows) == {'A': 1, 'B': 0}


def test_metrics_distinguish_f1_from_discrimination():
    y = np.array([0, 0, 1, 1])
    good = baseline.metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    dummy = baseline.metrics(y, np.full(4, 0.5))
    assert good['roc_auc'] == good['f1'] == 1
    assert dummy['roc_auc'] == dummy['balanced_accuracy'] == 0.5
    assert dummy['f1'] == 2 / 3
