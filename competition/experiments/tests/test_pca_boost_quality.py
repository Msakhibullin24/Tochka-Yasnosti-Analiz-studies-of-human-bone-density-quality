"""Protect outer-fold isolation in every fitted PCA/boost preprocessing stage."""
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pca_boost_quality import fit_model, predict_model


@pytest.mark.parametrize('configuration', [(4, 2), (8, 3)])
def test_heldout_feature_and_target_changes_cannot_change_fitted_model(configuration):
    rng = np.random.default_rng(71)
    latent = rng.normal(size=(80, 3))
    embeddings = [latent @ rng.normal(size=(3, 24)) + .05*rng.normal(size=(80, 24))
                  for _ in range(4)]
    geometry = latent @ rng.normal(size=(3, 12)) + .05*rng.normal(size=(80, 12))
    geometry[0, 2] = np.nan  # Exercise training-only imputation, too.
    regions = np.arange(80) % 2
    target = (latent[:, 0] > 0).astype(float)
    train, held = np.arange(60), np.arange(60, 80)
    changed_embeddings = [x.copy() for x in embeddings]
    for x in changed_embeddings:
        x[held] = rng.normal(size=x[held].shape)*1e8 + 1e10
    changed_geometry = geometry.copy()
    changed_geometry[held] = rng.normal(size=changed_geometry[held].shape)*1e9 - 1e11
    changed_target = target.copy()
    changed_target[held] = np.nan
    changed_regions = regions.copy()
    changed_regions[held] = 1-regions[held]

    with threadpool_limits(limits=1):
        original = fit_model(embeddings, geometry, regions, target, train, configuration)
        changed = fit_model(changed_embeddings, changed_geometry, changed_regions,
                            changed_target, train, configuration)
        # Evaluate both models against the same unmodified query images.
        query = np.arange(80)
        one = predict_model(original, embeddings, geometry, regions, query)
        two = predict_model(changed, embeddings, geometry, regions, query)
        np.testing.assert_array_equal(one, two)
        assert np.isfinite(one).all() and np.all((one >= 0) & (one <= 1))
        assert np.ptp(one[train]) > .01  # A constant classifier cannot satisfy this guard.
        for left, right in zip(original['projections'], changed['projections'], strict=True):
            np.testing.assert_array_equal(left.components_, right.components_)
            np.testing.assert_array_equal(left.mean_, right.mean_)
        for name, attributes in (
                ('imputer', ('statistics_',)),
                ('scaler', ('mean_', 'scale_')),
                ('geometry_projector', ('components_', 'mean_'))):
            for attribute in attributes:
                np.testing.assert_array_equal(getattr(original[name], attribute),
                                              getattr(changed[name], attribute))
        singleton = predict_model(original, embeddings, geometry, regions, held[:1])
        np.testing.assert_allclose(singleton, one[held[:1]], atol=1e-15, rtol=0)
