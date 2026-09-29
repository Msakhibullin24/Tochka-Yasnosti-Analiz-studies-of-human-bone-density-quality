"""Geometry contracts for explicitly synthetic external-anatomy experiments."""
import numpy as np
import pytest

from experiments.prepare_vsd_hip_projection import sphere_center
from experiments.train_vsd_hip_landmarks import features


def test_head_sphere_center_recovers_translated_physical_coordinates():
    center = np.array([100., -250., -600.])
    points = center + np.vstack([np.eye(3)*20, -np.eye(3)*20])
    assert sphere_center(points) == pytest.approx(center)
    with pytest.raises(ValueError, match='sphere'):
        sphere_center([[0,0,0], [1,0,0], [0,1,0], [1,1,0]])


def test_silhouette_feature_contract_is_translation_invariant_and_invertible():
    mask = np.zeros((100,100),np.uint8); mask[10:50,20:40] = 255
    shifted = np.zeros_like(mask); shifted[20:60,35:55] = 255
    vector, origin, extent = features(mask)
    moved, moved_origin, moved_extent = features(shifted)
    assert vector == pytest.approx(moved)
    assert extent == pytest.approx(moved_extent)
    assert moved_origin-origin == pytest.approx([15,10])
    point = np.array([25,25])
    assert ((point-origin)/extent)*extent+origin == pytest.approx(point)
    with pytest.raises(ValueError): features(np.zeros_like(mask))


def test_canonical_mask_frame_is_reflection_equivariant_and_rejects_ambiguity():
    from experiments.train_vsd_hip_landmarks import canonical_features, predict_mask
    mask=np.zeros((100,100),np.uint8)
    mask[10:45,10:45]=255; mask[45:85,30:50]=255
    a,origin,extent,reflected=canonical_features(mask)
    b,other_origin,other_extent,other_reflected=canonical_features(mask[:,::-1])
    assert a == pytest.approx(b)
    assert origin == pytest.approx(other_origin)
    assert extent == pytest.approx(other_extent)
    assert reflected is not other_reflected
    class Fixed:
        def predict(self,x): return np.tile([[.2,.2,.7,.5,.3,.3,.6,.4]],(len(x),1))
    bundle={'model':Fixed(),'feature_contract':'canonical_binary_mask_bbox_48x48_area_and_aspect_v2'}
    first=predict_mask(bundle,mask)
    second=predict_mask(bundle,mask[:,::-1]); second[:,0]=99-second[:,0]
    assert first == pytest.approx(second)
    symmetric=np.zeros_like(mask); symmetric[10:80,20:60]=255
    with pytest.raises(ValueError,match='ambiguous'):canonical_features(symmetric)
