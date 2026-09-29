"""Regression contracts for the final-delivery path, including withheld verdicts."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import synthetic_spine, write_dicom
from dxaqc.dicom_io import DicomReadError, read_dxa
from dxaqc.pipeline import Analyzer


def test_shipped_workflow_profile_matches_current_inference_code(monkeypatch):
    """A passing test suite must not coexist with an unbuildable Docker image."""
    from pathlib import Path
    from dxaqc.pipeline import MODEL_PATH
    from dxaqc.workflow_profile import configure_profile
    profile = Path(__file__).resolve().parents[1] / 'models/workflow_1_11_5/profile.json'
    for name in ('DXAQC_WORKFLOW_PROFILE', 'DXAQC_ANATOMY_MODEL',
                 'DXAQC_QUALITY_REVIEW_MODEL', 'DXAQC_PROJECTION_MODEL'):
        monkeypatch.delenv(name, raising=False)
    configured = configure_profile(profile, MODEL_PATH)
    assert configured['profile_id'] == 'tz-delivery-1.11.5'


@pytest.mark.parametrize('view,code', [('lateral', 'UNSUPPORTED_PROJECTION'),
                                     ('unknown', 'UNCERTAIN_PROJECTION')])
def test_pixel_projection_gates_quality_before_geometry_and_classification(tmp_path, monkeypatch, view, code):
    from dxaqc import embedding, geometry
    monkeypatch.setattr(embedding, 'embed', lambda pixels: np.zeros(512))
    def forbidden(*args):
        pytest.fail('frontal geometry must not run on a rejected projection')
    monkeypatch.setattr(geometry, 'measure_image', forbidden)
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.bundle = SimpleNamespace(router=SimpleNamespace(
        predict_detailed=lambda emb: (['spine'], [1.], [1.])))
    analyzer.projection_model = SimpleNamespace(predict=lambda pixels, region: {'value': view})
    path = write_dicom(tmp_path/'source.dcm', synthetic_spine(), study_uid='1.2', sop_uid='1.3')
    image = read_dxa(path)
    assert not image.view_position  # no metadata shortcut
    with pytest.raises(DicomReadError) as error:
        analyzer.analyze(image)
    assert error.value.code == code


def test_submission_gate_preserves_positive_alarm_and_reports_rejection(tmp_path, monkeypatch, bundle_available, trusted_synthetic_router):
    from dxaqc.cli import main
    from dxaqc.decision import decide
    original = Analyzer.analyze
    def untyped(self, image):
        result = original(self, image)
        result.update(decide('spine', .9, {'spine_artifact': .1}, result['features'], .5, {}))
        return result
    monkeypatch.setattr(Analyzer, 'analyze', untyped)
    source = write_dicom(tmp_path/'a.dcm', synthetic_spine(), study_uid='1.2', sop_uid='1.3')
    output = tmp_path/'output'
    assert main(['--input', str(source), '--output', str(output), '--no-explanations',
                 '--acceptance', 'submission']) == 3
    with (output/'submission.csv').open(encoding='utf-8-sig') as stream:
        row = next(csv.DictReader(stream))
    assert row['quality_class'] == '1' and row['violation_type'] == ''
    assert (output/'requirements.json').is_file()


def test_completion_gate_rejects_candidate_anatomy_even_with_valid_submission(tmp_path, monkeypatch, bundle_available, trusted_synthetic_router):
    from dxaqc.cli import main
    from dxaqc.decision import decide
    original = Analyzer.analyze
    def normal(self, image):
        result = original(self, image)
        result.update(decide('spine', .1, {'spine_artifact': .1}, result['features'], .5, {}))
        return result
    monkeypatch.setattr(Analyzer, 'analyze', normal)
    source = write_dicom(tmp_path/'a.dcm', synthetic_spine(), study_uid='1.2', sop_uid='1.3')
    assert main(['--input', str(source), '--output', str(tmp_path/'output'), '--no-explanations',
                 '--acceptance', 'complete']) == 3


def test_lateral_candidate_is_present_in_export_wording():
    from dxaqc.result_context import assessment_notes
    notes = assessment_notes({'projection_assessment': {'value': 'lateral'}})
    assert any('гипотеза боковой' in note for note in notes)


@pytest.mark.parametrize(('sides', 'expected_calls', 'duplicate_of'), [
    (('', ''), ['1.3'], '1.3'),
    (('L', 'R'), ['1.3', '1.4'], ''),
])
def test_duplicate_pixels_keep_independent_source_roi_decisions(
        tmp_path, monkeypatch, sides, expected_calls, duplicate_of):
    import pydicom
    from test_source_anatomy import add_overlay
    from dxaqc.decision import decide
    from dxaqc.pipeline import Options, run_batch
    source = tmp_path/'input'; source.mkdir()
    pixels = synthetic_spine()
    for (name, uid), side in zip([('a.dcm', '1.3'), ('b.dcm', '1.4')], sides):
        path = write_dicom(source/name, pixels, study_uid='1.2', sop_uid=uid)
        ds = pydicom.dcmread(path); ds.PixelSpacing = [1., 1.]
        if side:
            ds.Laterality = side
        if name == 'a.dcm':
            mask = np.zeros(pixels.shape, np.uint8); mask[10:220, 40:210] = 1
            add_overlay(ds, mask)
            ds.add_new((0x6000, 0x1500), 'LO', 'ROI_SCAN_COVERAGE')
        ds.save_as(path, enforce_file_format=True)
    calls = []
    def inference(self, image):
        calls.append(image.image_uid)
        return {'region': 'hip_right', 'region_confidence': 1., 'laterality_confidence': 1.,
                'laterality_basis': 'model',
                'features': {}, 'criteria': {'hip_roi_coverage': .1}, 'overlay': {},
                **decide('hip', .1, {'hip_roi_coverage': .1}, {}, .5, {})}
    monkeypatch.setattr(Analyzer, 'analyze', inference)
    summary = run_batch(source, tmp_path/'output', Options(explanations=False))
    with (tmp_path/'output/results_extended.csv').open(encoding='utf-8-sig') as stream:
        a, b = list(csv.DictReader(stream))
    assert summary['success'] == 2 and calls == expected_calls
    assert a['quality_class'] == '1' and a['violation_type'] == 'Некорректная область интереса'
    assert b['quality_class'] == '0' and not b['violation_type'] and b['duplicate_of'] == duplicate_of
    assert a['decision_reason'] == 'source_roi_scan_margin_failure'


def test_neck_roi_or_uncalibrated_coverage_does_not_invent_anatomical_failure():
    from dxaqc.anatomy import apply_source_roi_findings
    from dxaqc.decision import decide
    for purpose, margin in [('femoral_neck', 'not_applicable'), ('scan_coverage', 'unavailable')]:
        result = {'region': 'hip_right', **decide('hip', .1, {}, {}, .5, {})}
        apply_source_roi_findings(result, {'checks': [{'purpose': purpose, 'margin_status': margin}]})
        assert result['quality'] == 0 and not result['violations']


def test_packaged_workflow_contains_verified_nested_weights_and_preserves_environment(tmp_path, monkeypatch):
    import json
    import os
    from pack_workflow import pack, digest
    from dxaqc.pipeline import MODEL_PATH
    from dxaqc.workflow_profile import ARTIFACT_ENV, configure_profile
    source = MODEL_PATH.parent/'workflow/profile.json'
    monkeypatch.setenv('DXAQC_WORKFLOW_PROFILE', 'unchanged')
    for name in ARTIFACT_ENV:
        monkeypatch.setenv(name, 'unchanged')
    output = tmp_path/'workflow'
    profile = pack(source, output, 'test-delivery')
    assert os.environ['DXAQC_WORKFLOW_PROFILE'] == 'unchanged'
    for name in ('spine.pt', 'hip.pt', 'quality_review.joblib', 'projection.joblib'):
        assert (output/name).is_file()
        assert (output/name).stat().st_mode & 0o777 == 0o644
    assert output.stat().st_mode & 0o777 == 0o755
    configure_profile(output/'profile.json', MODEL_PATH)
    anatomy = json.loads((output/'anatomy.json').read_text())
    assert all(digest(output/spec['path']) == spec['sha256'] for spec in anatomy['models'].values())
    assert profile['source_profile_sha256'] == digest(source)
    with pytest.raises(ValueError, match='new workflow'):
        pack(source, output, 'overwrite')
    (output/'projection.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        configure_profile(output/'profile.json', MODEL_PATH)
