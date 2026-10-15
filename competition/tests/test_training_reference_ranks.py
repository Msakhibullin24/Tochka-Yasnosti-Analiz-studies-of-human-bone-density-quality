import numpy as np
import pytest

from experiments.threshold_stability import ranks, reference_ranks


def test_prediction_is_invariant_to_other_test_images_and_order():
    reference = np.array([.1, .2, .2, .4, .8])
    alone = reference_ranks(np.array([.3]), reference)[0]
    batch = reference_ranks(np.array([.9, .3, .01, .3]), reference)
    assert alone == batch[1] == batch[3]
    np.testing.assert_array_equal(batch[::-1], reference_ranks(np.array([.3, .01, .3, .9]), reference))
    # Historical within-batch ranking fails this property.
    assert ranks(np.array([.3]))[0] != ranks(np.array([.3, .9]))[0]


def test_training_reference_preserves_tied_training_ranks_and_score_order():
    reference = np.array([.2, .1, .2, .9, .4])
    np.testing.assert_array_equal(reference_ranks(reference, reference), ranks(reference))
    values = reference_ranks(np.linspace(0, 1, 101), reference)
    assert np.all(np.diff(values) >= 0)
    assert np.all((values >= 0) & (values <= 1))


@pytest.mark.parametrize('values,reference', [([.1], []), ([np.nan], [.1]), ([.1], [np.inf]), ([[.1]], [.1])])
def test_invalid_or_incomplete_reference_is_rejected(values, reference):
    with pytest.raises(ValueError, match='training reference'):
        reference_ranks(values, reference)
