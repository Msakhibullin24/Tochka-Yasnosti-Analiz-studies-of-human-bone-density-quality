"""Independent regression checks for the organiser's DOCX V2 contract."""
import csv
import numpy as np
import pydicom
import pytest
from conftest import synthetic_spine, write_dicom
from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import measure_image
from dxaqc.model import official_violation_type
from dxaqc.report import write_csv
from dxaqc.validate_results import validate
from test_result_context import untyped_row


def test_organiser_scale_overrides_exposed_area_not_explicit_spacing(tmp_path):
    path = write_dicom(tmp_path / 'x.dcm', synthetic_spine(), exposed_area=(180, 180))
    ds = pydicom.dcmread(path)
    ds.ManufacturerModelName = 'Lunar Prodigy Advance'
    ds.save_as(path)
    img = read_dxa(path)
    assert (img.pixel_mm, img.pixel_mm_x, img.pixel_mm_source) == (1.05, .6, 'organiser_v2')
    ds.PixelSpacing = [.8, .7]
    ds.save_as(path)
    img = read_dxa(path)
    assert (img.pixel_mm, img.pixel_mm_x) == (.8, .7)
    assert 'EXPLICIT_SCALE_DIFFERS_FROM_ORGANISER_V2' in img.warnings


def test_exact_docx_vocabulary_and_strict_unknown_rejection(tmp_path):
    assert official_violation_type(['spine_axis']) == 'Не выравнена ось позвоночника'
    row = untyped_row(); path = tmp_path/'submission.csv'
    write_csv([row], path, True)
    assert not validate(path, competition=True)['valid']
    row['violation_type'] = 'Не выравнена ось позвоночника'
    write_csv([row], path, True)
    assert validate(path, competition=True)['valid']
    row['violation_type'] = 'Не выровнена ось позвоночника'
    write_csv([row], path, True)
    assert not validate(path, competition=True)['valid']
    write_csv([row], path)
    assert any('columns' in e for e in validate(path, competition=True)['errors'])


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_synthetic_rotation_is_in_physical_coordinates(seed):
    from train import synthetic_spine as augment
    pixels = synthetic_spine(4)
    angle = measure_image(pixels, 'spine', 1.05, .6).features['spine_angle_deg']
    expected_rng = np.random.default_rng(seed)
    target = expected_rng.uniform(7., 12.) * expected_rng.choice([-1., 1.])
    changed = augment(pixels, 'rotate', np.random.default_rng(seed), angle, 1.05, .6)
    measured = measure_image(changed, 'spine', 1.05, .6).features['spine_angle_deg']
    assert abs(measured - target) < 1.0


def test_pipeline_keeps_diagnostics_beside_submission(tmp_path):
    from dxaqc.pipeline import write_tables, Options
    import json
    row = untyped_row()
    write_tables([row], tmp_path, Options(strict_columns=True))
    with open(tmp_path/'results.csv', encoding='utf-8-sig') as f:
        assert len(next(csv.reader(f))) == 9
    with open(tmp_path/'results_extended.csv', encoding='utf-8-sig') as f:
        assert next(csv.DictReader(f))['violation_type_status'] == 'undetermined'
    acceptance = json.loads((tmp_path/'submission_validation.json').read_text())
    assert not acceptance['valid'] and acceptance['profile'] == 'competition_v2'
    assert (tmp_path/'submission.csv').read_bytes() == (tmp_path/'results.csv').read_bytes()


def test_old_weights_cannot_silently_use_new_calibration(monkeypatch):
    from types import SimpleNamespace
    from dxaqc.pipeline import load_bundle
    import joblib
    monkeypatch.setattr(joblib, 'load', lambda path: SimpleNamespace(meta={}))
    with pytest.raises(ValueError, match='calibration mismatch'):
        load_bundle()
