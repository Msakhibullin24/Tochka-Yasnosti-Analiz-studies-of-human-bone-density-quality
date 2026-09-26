import pytest

from audit_anatomy_on_public_rois import match_centers


def test_body_matching_is_one_to_one_and_does_not_supply_numbering():
    result = match_centers([[5, 5], [5, 10]], [[5, 6]])
    assert result == [{'reference_index': 0, 'candidate_index': 0, 'distance_pixels': 1.}]
    assert match_centers([[5, 5]], []) == []
    assert match_centers([], [[5, 5]]) == []
    with pytest.raises(ValueError):
        match_centers([[float('nan'), 5]], [[5, 5]])
