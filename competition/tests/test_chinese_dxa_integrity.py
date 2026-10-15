import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from experiments.chinese_dxa_probe import aligned_features, grouped_ridge
from experiments.prepare_chinese_dxa import parse_name, person_group


def test_person_group_merges_dates_and_case_without_claiming_identity():
    assert person_group('20240101Zhang') == person_group('20250101zhang')
    assert parse_name('20240101Zhang_LumbarP_LumbarP')['person_group'] == 'zhang'
    assert not parse_name('20240101_LumbarP')['parsed']


def test_feature_identity_survives_filtering_reordering_and_join_duplicates():
    values = np.arange(20).reshape(10, 2)
    joined = pd.DataFrame({'_feature_row': [8, 3, 8]})
    np.testing.assert_array_equal(aligned_features(values, joined), values[[8, 3, 8]])


def test_outer_targets_do_not_affect_predictions_of_their_heldout_group():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(60, 4))
    y = x[:, 0] + rng.normal(size=60)
    x[::7, 1] = np.nan
    groups = np.repeat(np.arange(20), 3)
    _, held = next(GroupKFold(5).split(x, y, groups))
    first = grouped_ridge(x, y, groups, alphas=(1, 100))
    altered = y.copy()
    altered[held] = rng.normal(100, 10, len(held))
    second = grouped_ridge(x, altered, groups, alphas=(1, 100))
    np.testing.assert_array_equal(first[2][held], second[2][held])
    assert first[1][0] == second[1][0]


def test_single_alpha_still_predicts_every_source_row_once():
    x = np.arange(120).reshape(60, 2)
    y = x[:, 0] / 4
    groups = np.repeat(np.arange(20), 3)
    score, alphas, predictions = grouped_ridge(x, y, groups, alphas=(10,))
    assert np.isfinite(predictions).all() and score > .95
    assert alphas == [10] * 5
