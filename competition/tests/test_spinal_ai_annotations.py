import numpy as np
import pytest

from spinal_ai_annotations import normalize, quad_polygon


def test_source_bowtie_corners_become_simple_quad_without_changing_points():
    points = [10, 10, 20, 10, 10, 20, 20, 20]
    actual = np.array(quad_polygon(points, 100, 100)).reshape(4, 2)
    assert set(map(tuple, actual)) == set(map(tuple, np.array(points).reshape(4, 2)))
    x, y = actual.T
    assert abs((x*np.roll(y, -1)-y*np.roll(x, -1)).sum()/2) == 100


@pytest.mark.parametrize('points', [[-1, 10, 20, 10, 10, 20, 20, 20],
                                   [10, 10, 100, 10, 10, 20, 20, 20],
                                   [10, 10]*4, [10, 10, 20, 20, 30, 30, 40, 40]])
def test_invalid_corners_are_not_clipped_or_invented(points):
    with pytest.raises(ValueError):
        quad_polygon(points, 100, 100)


def test_normalization_quarantines_bad_geometry_and_preserves_source_semantics():
    data = {'images': [{'id': 1, 'width': 100, 'height': 100}],
            'categories': [{'id': 1, 'name': 'text'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'transcription': 'random',
                             'segmentation': [[10, 10, 20, 10, 10, 20, 20, 20]]},
                            {'id': 2, 'image_id': 1, 'segmentation': [[-1, 10, 20, 10, 10, 20, 20, 20]]}]}
    converted, quarantine = normalize(data)
    assert len(converted['annotations']) == 1
    assert converted['categories'] == data['categories']
    assert converted['annotations'][0]['transcription'] == 'random'
    assert converted['annotations'][0]['area'] == 100
    assert converted['conversion']['numbered_anatomical_labels'] is False
    assert quarantine[0]['annotation_id'] == 2
    assert len(data['annotations']) == 2
