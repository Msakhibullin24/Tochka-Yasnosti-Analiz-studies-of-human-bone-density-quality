import numpy as np
import pytest

from experiments.anatomy_patch_pooling import project_mask, weighted_pool


def test_mask_projection_preserves_padding_and_applies_recorded_reflection():
    mask = np.zeros((4, 4), np.float32)
    mask[1, 0] = 1.
    transform = {'mirrored': False, 'x': 2, 'y': 2, 'width': 4, 'height': 4}
    original = project_mask(mask, transform, 8, 8).reshape(8, 8)
    assert original[3, 2] == 1
    assert original.sum() == 1
    transform['mirrored'] = True
    reflected = project_mask(mask, transform, 8, 8).reshape(8, 8)
    assert reflected[3, 5] == 1
    assert reflected.sum() == 1


def test_weighted_pool_has_no_cross_image_dependence_and_handles_absent_anatomy():
    tokens = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.testing.assert_array_equal(weighted_pool(tokens, [0, 1, 0, 0]), tokens[1])
    np.testing.assert_array_equal(weighted_pool(tokens, [0, 0, 0, 0]), np.zeros(3))
    np.testing.assert_array_equal(weighted_pool(tokens, [1, 1, 1, 1]), tokens.mean(0))
    with pytest.raises(ValueError):
        weighted_pool(tokens, [1, -1, 0, 0])
