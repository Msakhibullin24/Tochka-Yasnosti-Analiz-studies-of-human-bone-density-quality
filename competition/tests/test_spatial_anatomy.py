import numpy as np
import pytest
import torch

from train_spatial_anatomy import decode, heatmap_targets
from adapt_spatial_dxa import labeled_loss
from evaluate_spatial_transfer import exclusion_reason


def test_numbered_heatmaps_preserve_xy_and_level_order():
    points = np.array([[.2, .3], [.4, .45], [.5, .6], [.6, .7], [.7, .8]])
    targets = heatmap_targets(points)
    assert targets.shape == (5, 40, 40)
    np.testing.assert_allclose(targets.sum(axis=(1, 2)), 1, atol=1e-6)
    predicted = decode(np.log(np.maximum(targets, 1e-30))[None])[0]
    np.testing.assert_allclose(predicted, points, atol=.01)


def test_edge_centers_remain_finite_and_inside_image():
    points = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [.5, .5]])
    predicted = decode(np.log(np.maximum(heatmap_targets(points), 1e-30))[None])
    assert np.isfinite(predicted).all()
    assert (predicted >= 0).all() and (predicted <= 1).all()


@pytest.mark.parametrize('points', [np.zeros((4, 2)), np.full((5, 2), np.nan),
                                  np.full((5, 2), 1.1), np.full((5, 2), -.1)])
def test_invalid_numbered_references_fail(points):
    with pytest.raises(ValueError):
        heatmap_targets(points)


@pytest.mark.parametrize('values', [np.zeros((1, 4, 40, 40)), np.zeros((1, 5, 0, 40)),
                                  np.full((1, 5, 40, 40), np.inf)])
def test_invalid_predictions_fail(values):
    with pytest.raises(ValueError):
        decode(values)


def test_missing_l5_reference_contributes_no_training_gradient():
    logits = torch.zeros((1, 5, 4, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 1, 2] = 1
    visible = torch.tensor([[True, True, True, True, False]])
    labeled_loss(logits, target, visible).backward()
    assert logits.grad[:, :4].abs().sum() > 0
    assert logits.grad[:, 4].abs().sum() == 0
    with pytest.raises(ValueError):
        labeled_loss(logits, target, torch.zeros_like(visible))


def test_transfer_excludes_all_existing_patient_views_and_pixels():
    from pathlib import Path
    import hashlib
    pixels = np.zeros((10, 10), dtype=np.uint8)
    assert exclusion_reason(Path('0001-F-037Y0.jpg'), pixels, {'0001-F-037Y'}, set()) == 'existing_patient_filename'
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    assert exclusion_reason(Path('newY0.jpg'), pixels, set(), {digest}) == 'existing_pixels'
    assert exclusion_reason(Path('newY0.jpg'), pixels, set(), set()) is None
