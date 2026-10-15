import cv2
import numpy as np
import pytest
import torch

from experiments.train_silhouette_heatmaps import coordinates, normalize_proximal_mask


def test_heatmap_pixel_centers_preserve_original_coordinate_convention():
    logits = torch.full((1, 4, 64, 64), -100.)
    logits[:, :, 10, 20] = 100.
    points, _ = coordinates(logits)
    restored = (points[0, 0].numpy() + 1) / 2 * 320 - .5
    assert restored == pytest.approx([(20.5) * 5 - .5, (10.5) * 5 - .5])


def test_physical_normalization_has_invertible_affine_and_preserves_both_axes():
    mask = np.zeros((400, 200), np.uint8)
    mask[30:350, 50:100] = 1
    normalized, affine = normalize_proximal_mask(mask, .6, 1.05)
    assert normalized.shape == (320, 320)
    assert affine[0, 0] / affine[1, 1] == pytest.approx(.6 / 1.05)
    point = np.array([75., 80.])
    output = point @ affine[:, :2].T + affine[:, 2]
    inverse = cv2.invertAffineTransform(affine)
    assert output @ inverse[:, :2].T + inverse[:, 2] == pytest.approx(point)
    assert normalized.sum() > 0


def test_missing_mask_or_scale_cannot_produce_anatomical_points():
    mask = np.zeros((100, 100), np.uint8)
    with pytest.raises(ValueError, match='Nonempty'):
        normalize_proximal_mask(mask, 1., 1.)
    mask[10:80, 20:50] = 1
    for sx, sy in ((0, 1), (1, np.nan)):
        with pytest.raises(ValueError, match='calibration'):
            normalize_proximal_mask(mask, sx, sy)
