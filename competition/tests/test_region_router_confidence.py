import numpy as np
import pytest
from types import SimpleNamespace

from dxaqc.dicom_io import DicomReadError
from dxaqc.model import RegionRouter
from dxaqc.pipeline import Analyzer


class FixedProba:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, embeddings):
        return np.array([[.168, .369, .463], [.7, .2, .1], [.05, .8, .15]])


def test_router_separates_anatomical_group_from_hip_side_confidence():
    router = RegionRouter(FixedProba())
    regions, group_confidence, side_confidence = router.predict_detailed(np.zeros((3, 2)))
    assert regions == ['hip_left', 'spine', 'hip_right']
    assert np.allclose(group_confidence, [.832, .7, .95])
    assert np.allclose(side_confidence, [.463 / .832, 1, .8 / .95])
    # Legacy three-class API remains available for existing training reports.
    legacy_regions, legacy_confidence = router.predict(np.zeros((3, 2)))
    assert legacy_regions == ['hip_left', 'spine', 'hip_right']
    assert np.allclose(legacy_confidence, [.463, .7, .8])


def test_detailed_router_rejects_missing_side_class():
    class IncompleteProba:
        classes_ = np.array([0, 2])

        def predict_proba(self, embeddings):
            return np.array([[.2, .8]])

    with pytest.raises(ValueError, match='fixed order'):
        RegionRouter(IncompleteProba()).predict_detailed(np.zeros((1, 2)))


@pytest.mark.parametrize(('group_confidence', 'side_confidence', 'code'), [
    (.55, .9, 'UNCERTAIN_REGION'),
    (.85, .55, 'UNCERTAIN_LATERALITY'),
])
def test_analyzer_refuses_uncertain_anatomy_and_side_separately(
        monkeypatch, group_confidence, side_confidence, code):
    class Router:
        def predict_detailed(self, embedding):
            return ['hip_left'], np.array([group_confidence]), np.array([side_confidence])

    monkeypatch.setattr('dxaqc.embedding.embed', lambda pixels: np.zeros(4))
    analyzer = Analyzer(SimpleNamespace(router=Router()))
    with pytest.raises(DicomReadError) as exc:
        analyzer.analyze(SimpleNamespace(pixels=np.zeros((32, 32), np.uint8)))
    assert exc.value.code == code


@pytest.mark.parametrize(('declared', 'expected_basis'), [
    ('L', 'model_agrees_with_Laterality'),
    ('R', None),
    ('', None),
])
def test_low_confidence_hip_side_requires_agreeing_dicom_declaration(
        monkeypatch, declared, expected_basis):
    class Router:
        def predict_detailed(self, embedding):
            return ['hip_left'], np.array([.85]), np.array([.55])

    class Group:
        quality_threshold = .5
        criterion_thresholds = {}

        def predict(self, features, embedding):
            return np.array([.2]), {}

    monkeypatch.setattr('dxaqc.embedding.embed', lambda pixels: np.zeros(4))
    monkeypatch.setattr('dxaqc.geometry.measure_image',
                        lambda *args: SimpleNamespace(features={}, overlay={}))
    monkeypatch.setattr('dxaqc.pipeline.decide', lambda *args: {'quality': 0})
    bundle = SimpleNamespace(router=Router(), groups={'hip': Group()})
    image = SimpleNamespace(pixels=np.zeros((64, 64), np.uint8), pixel_mm=1., pixel_mm_x=1.,
                            declared_laterality=declared, declared_laterality_source='Laterality')
    analyzer = Analyzer(bundle)
    if expected_basis is None:
        with pytest.raises(DicomReadError) as error:
            analyzer.analyze(image)
        assert error.value.code == 'UNCERTAIN_LATERALITY'
    else:
        result = analyzer.analyze(image)
        assert result['region'] == 'hip_left'
        assert result['laterality_confidence'] == .55
        assert result['laterality_basis'] == expected_basis
