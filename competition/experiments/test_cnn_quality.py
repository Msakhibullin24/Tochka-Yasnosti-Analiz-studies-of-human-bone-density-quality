"""Guard the evaluation boundary of the isolated CNN experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.cnn_quality import metrics, outer_folds, valid_labels


def test_study_folds_hold_out_each_image_once():
    labels = pd.DataFrame({
        "study_key": np.repeat(np.arange(20).astype(str), 2),
        "region": ["spine"] * 40,
    })
    y = np.tile([0, 1], 20)
    folds = list(outer_folds(labels, y, "spine"))
    assert len(folds) == 5
    assert sorted(np.concatenate([test for _, _, test in folds]).tolist()) == list(range(40))
    for _, train, test in folds:
        assert set(labels.study_key.iloc[train]).isdisjoint(labels.study_key.iloc[test])


def test_duplicate_source_across_studies_is_rejected(tmp_path):
    labels = pd.DataFrame({
        "first_source_path": ["a.dcm", "b.dcm"],
        "pixel_sha256": ["same", "same"],
        "study_key": ["one", "two"],
        "region": ["spine", "spine"],
        "quality_class": [0, 1],
    })
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError, match="identical image"):
        valid_labels(path)


def test_fixed_threshold_metrics_count_misses():
    result = metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .9]))
    assert result["confusion_tn_fp_fn_tp"] == [1, 1, 1, 1]
    assert result["sensitivity"] == .5
