import csv

from dxaqc.review import Region
from dxaqc.roi import evaluate_roi
from dxaqc.validate_results import validate
from dxaqc.report import write_csv


def polygon():
    return Region(name='roi', kind='polygon', roi_edge='left', points=[
        {'x': .01, 'y': .01}, {'x': .21, 'y': .01}, {'x': .21, 'y': .21}, {'x': .01, 'y': .21}])


def test_roi_translation_preserves_shape_and_passes_anisotropic_margins():
    detail = dict(region='hip_left', width=201, height=101, pixel_mm_x=.5, pixel_mm_y=2, pixel_mm_source='PixelSpacing')
    g = polygon()
    result = evaluate_roi(g, detail)
    assert result['margins_mm'] == {'top': 2., 'bottom': 158., 'side': 1.}
    moved = Region(**result['proposal'])
    assert evaluate_roi(moved, detail)['status'] == 'pass'
    assert abs((moved.points[1].x-moved.points[0].x) - .2) < 1e-9
    assert g.points[0].x == .01
    assert evaluate_roi(g, dict(detail, pixel_mm_source='device_default'))['proposal'] is None
    assert evaluate_roi(g, dict(detail, pixel_mm_y=.1))['proposal'] is None
    assert evaluate_roi(g, dict(detail, region='spine'))['status'] == 'unavailable'


def test_validator_detects_contract_manifest_and_repeat_regressions(tmp_path):
    from dxaqc.model import REGION_LABEL
    row = dict(path_to_study='study', path_to_file='a.dcm', study_uid='1', image_uid='2',
               anatomical_region=REGION_LABEL['spine'], quality_class=0, violation_type='',
               processing_status='Success', time_of_processing=.1, quality_prob=.2)
    path = tmp_path/'results.csv'
    write_csv([row], path)
    assert validate(path, manifest=['a.dcm'])['valid']
    assert not validate(path, manifest=['a.dcm','b.dcm'])['valid']
    repeat = tmp_path/'repeat.csv'
    write_csv([dict(row, time_of_processing=.2)], repeat)
    assert validate(path, repeat)['valid']
    write_csv([dict(row, quality_class=1)], repeat)
    assert not validate(repeat)['valid']
    assert not validate(path, repeat)['valid']
    write_csv([dict(row, time_of_processing=181)], repeat)
    assert not validate(repeat)['valid']
    write_csv([dict(row, anatomical_region=REGION_LABEL['hip_left'], quality_class=1,
                    violation_type='Не выравнена ось позвоночника')], repeat)
    assert not validate(repeat)['valid']


def test_multiframe_and_declared_nonfrontal_are_reported_without_decoding(tmp_path):
    import pydicom
    import pytest
    from conftest import synthetic_spine, write_dicom
    from dxaqc.dicom_io import read_dxa, DicomReadError
    path = write_dicom(tmp_path/'a.dcm', synthetic_spine())
    ds = pydicom.dcmread(path)
    ds.NumberOfFrames = 2
    ds.save_as(path)
    with pytest.raises(DicomReadError, match='multi-frame'):
        read_dxa(path)
    ds.NumberOfFrames = 1
    for view in ('LL', 'RL', 'LLD', 'RLD', 'LLO', 'RLO', 'LAT', 'LATERAL'):
        ds.ViewPosition = view
        ds.save_as(path)
        with pytest.raises(DicomReadError) as error:
            read_dxa(path)
        assert error.value.code == 'UNSUPPORTED_PROJECTION'
    for view in ('AP', 'PA', ''):
        ds.ViewPosition = view
        ds.save_as(path)
        assert read_dxa(path).view_position == view


def test_dicom_laterality_uses_only_consistent_paired_side(tmp_path):
    import pydicom
    from conftest import synthetic_spine, write_dicom
    from dxaqc.dicom_io import read_dxa

    path = write_dicom(tmp_path/'hip.dcm', synthetic_spine())
    ds = pydicom.dcmread(path)
    ds.Laterality = 'L'
    ds.save_as(path)
    assert (read_dxa(path).declared_laterality, read_dxa(path).declared_laterality_source) == ('L', 'Laterality')
    ds.ImageLaterality = 'L'
    ds.save_as(path)
    assert (read_dxa(path).declared_laterality, read_dxa(path).declared_laterality_source) == ('L', 'ImageLaterality')
    ds.ImageLaterality = 'R'
    ds.save_as(path)
    assert read_dxa(path).declared_laterality == ''
    ds.ImageLaterality = 'B'
    ds.save_as(path)
    assert read_dxa(path).declared_laterality == ''
