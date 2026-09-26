import numpy as np
import pytest

from evaluate_buu_transfer import match_bodies, references


def test_far_candidates_do_not_count_as_localized_bodies():
    result = match_bodies([[0, 0], [0, 20]], [10, 10], [[0, 1], [100, 20]])
    assert len(result['matched']) == 1
    assert result['missed'] == result['extra'] == 1


def test_one_candidate_cannot_match_two_reference_bodies():
    result = match_bodies([[0, 0], [0, 4]], [10, 10], [[0, 2]])
    assert len(result['matched']) == 1 and result['missed'] == 1


def test_matching_maximizes_valid_pairs_before_distance():
    result = match_bodies([[0, 0], [0, 4]], [10, 10], [[0, 1], [0, -4]])
    assert len(result['matched']) == 2


def test_reference_edges_require_correct_shape_and_bounds(tmp_path):
    p = tmp_path/'a.csv'
    p.write_text('0,0,2,0,0\n0,2,2,2,0\n'*5)
    with pytest.raises(ValueError, match='unordered'):
        references(p, 10, 20)
    with pytest.raises(ValueError, match='outside'):
        references(p, 2, 20)
    with pytest.raises(ValueError):
        match_bodies([[0, 0]], [np.nan], [[0, 0]])


def test_projection_split_keeps_both_views_together_and_rejects_pixel_leakage():
    from train_buu_projection import patient_split
    rows = [{'patient_id':str(p),'pixel_sha256':f'{p}:{v}'} for p in range(8) for v in range(2)]
    _, _, test = patient_split(rows)
    assert all(test[i] == test[i+1] for i in range(0,len(rows),2))
    held = next(i for i,t in enumerate(test) if t)
    train = next(i for i,t in enumerate(test) if not t)
    rows[held]['pixel_sha256'] = rows[train]['pixel_sha256']
    with pytest.raises(ValueError, match='leakage'):
        patient_split(rows)
