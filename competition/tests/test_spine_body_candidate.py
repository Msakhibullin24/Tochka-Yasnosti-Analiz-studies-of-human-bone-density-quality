import numpy as np
import pytest

import spine_body_candidate as candidate


def test_candidate_uses_physical_grid_and_returns_original_coordinates(monkeypatch):
    widths = []
    def centreline(pixels):
        h, w = pixels.shape
        widths.append(w)
        return np.arange(h), np.full(h, (w-1)/2), np.ones(h)
    monkeypatch.setattr(candidate, 'spine_centerline', centreline)
    image = np.full((200, 100), 160, dtype=np.uint8)
    for y in (40, 80, 120, 160):
        image[y-5:y+6] = 50
    result = candidate.detect(image, 1., .5)
    assert widths == [50]
    assert len(result['points']) == 4
    assert all(abs(x-49.5) < 1e-6 for x, y in result['points'])
    assert result['absolute_levels'] is None
    assert result['clinical_validation'] is False
    assert candidate.detect(np.zeros_like(image), 1., 1.)['points'] == []
    with pytest.raises(ValueError, match='spacing'):
        candidate.detect(image, 0., 1.)


def test_candidate_stability_rejects_lost_points_and_applies_physical_spacing():
    first = {'points': [[10., 10.], [20., 20.]]}
    second = {'points': [[12., 10.], [22., 20.]]}
    assert candidate.stability(first, second, 1., .5)['max_shift_mm'] == 1.
    assert candidate.stability(first, second, 1., .5)['status'] == 'repeatable'
    assert candidate.stability(first, {'points': [[10., 10.]]}, 1., 1.)['status'] == 'unstable'
    assert candidate.stability(first, {'points': []}, 1., 1.)['status'] == 'unavailable'
