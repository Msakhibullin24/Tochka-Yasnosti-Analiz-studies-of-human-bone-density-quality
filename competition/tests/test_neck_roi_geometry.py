from types import SimpleNamespace

import numpy as np
import pytest

from dxaqc.neck_roi_geometry import neck_geometry, assess_neck_rois
from dxaqc.mask_raster import encode_raster


def masks():
    roi = np.zeros((70, 80), bool); roi[20:35, 10:70] = True
    femur = np.zeros_like(roi); femur[5:60, 30:45] = True
    return roi, femur


def test_perpendicular_roi_with_known_axis_and_two_candidate_soft_tissue_sides():
    roi, femur = masks()
    result = neck_geometry(roi, femur, .6, 1.05, [[37, 10], [37, 50]])
    assert result['roi_to_neck_axis_angle_deg'] == pytest.approx(90)
    assert result['deviation_from_perpendicular_deg'] == pytest.approx(0)
    assert result['soft_tissue_sides']['both_sides_have_candidate_pixels']
    assert result['clinical_verdict'] == 'undetermined'
    assert result['clinical_validation'] is False


def test_physical_spacing_is_used_and_axis_is_not_invented_from_clipped_bone():
    roi, femur = masks()
    without_axis = neck_geometry(roi, femur, .6, 1.05)
    assert without_axis['roi_to_neck_axis_angle_deg'] is None
    assert without_axis['status'] == 'partial'
    result = neck_geometry(roi, femur, .6, 1.05, [[20, 10], [40, 30]])
    assert result['roi_to_neck_axis_angle_deg'] == pytest.approx(np.degrees(np.arctan(1.05/.6)))
    assert result['deviation_from_perpendicular_deg'] != pytest.approx(45)


def test_absent_bone_and_isotropic_roi_do_not_yield_an_invented_axis():
    roi, femur = masks(); femur[:] = False; femur[:3, :3] = True
    assert neck_geometry(roi, femur, 1., 1.)['roi_to_neck_axis_angle_deg'] is None
    square = np.zeros_like(roi); square[10:30, 10:30] = True
    assert neck_geometry(square, femur, 1., 1., [[1, 1], [2, 2]])['soft_tissue_sides'] is None


def test_missing_soft_tissue_on_one_side_is_measured_without_clinical_threshold():
    roi, femur = masks(); roi[:, :30] = False
    result = neck_geometry(roi, femur, 1., 1.)
    assert not result['soft_tissue_sides']['both_sides_have_candidate_pixels']
    assert result['clinical_verdict'] == 'undetermined'


def test_source_roi_is_checked_against_whole_structure_masks_and_purpose():
    roi, femur = masks()
    source = {'rois': [{'id': 'a', 'purpose': 'femoral_neck', 'raster': encode_raster(roi)}]}
    img = SimpleNamespace(pixels=np.zeros(roi.shape, np.uint8), pixel_mm=1.05, pixel_mm_x=.6, pixel_mm_source='PixelSpacing')
    trochanter = np.zeros_like(roi); trochanter[22:26, 15:18] = True
    regions = [{'name': 'femur', 'raster': encode_raster(femur)},
               {'name': 'greater_trochanter', 'raster': encode_raster(trochanter)}]
    result = assess_neck_rois(source, regions, img)['checks'][0]
    assert result['structure_exclusions']['greater_trochanter']['candidate_intersection_pixels'] == 12
    assert result['structure_exclusions']['ischium']['status'] == 'unavailable'
    img.pixel_mm_source = 'device_default'
    assert assess_neck_rois(source, regions, img)['checks'][0]['status'] == 'unavailable'
    source['rois'][0]['purpose'] = 'scan_coverage'
    assert assess_neck_rois(source, regions, img)['status'] == 'not_applicable'


@pytest.mark.parametrize('scale', [0, float('nan'), True])
def test_invalid_scale_is_rejected(scale):
    roi, femur = masks()
    with pytest.raises(ValueError): neck_geometry(roi, femur, scale, 1.)


def test_batch_reports_explicit_neck_geometry_without_retyping_it_as_scan_coverage(tmp_path, monkeypatch):
    import csv
    import json
    import pydicom
    from conftest import write_dicom
    from test_source_anatomy import add_overlay
    from dxaqc.pipeline import Analyzer, Options, run_batch
    from dxaqc.decision import decide
    roi, femur = masks()
    path = write_dicom(tmp_path/'a.dcm', roi.astype(np.uint8)*180 + femur.astype(np.uint8)*60, study_uid='1.2', sop_uid='1.3')
    ds = pydicom.dcmread(path); ds.PixelSpacing = [1.05, .6]
    add_overlay(ds, roi.astype(np.uint8)); ds.add_new((0x6000, 0x1500), 'LO', 'ROI_FEMORAL_NECK')
    ds.save_as(path, enforce_file_format=True)
    def inference(self, image):
        return {'region': 'hip_right', 'region_confidence': 1., 'laterality_confidence': 1.,
                'laterality_basis': 'model',
                'features': {}, 'criteria': {}, 'overlay': {},
                'learned_anatomy': {'status': 'evaluated', 'model_sha256': 'synthetic', 'regions': [
                    {'name': 'femur', 'raster': encode_raster(femur)}]},
                **decide('hip', .1, {}, {}, .5, {})}
    monkeypatch.setattr(Analyzer, 'analyze', inference)
    run_batch(path, tmp_path/'out', Options(explanations=False))
    with (tmp_path/'out/results_extended.csv').open(encoding='utf-8-sig') as stream:
        row = next(csv.DictReader(stream))
    assert row['processing_status'] == 'Success' and row['quality_class'] == '0'
    assert row['violation_type'] == '' and row['anatomical_checks_complete'] == 'false'
    check = json.loads(row['source_roi_assessment'])['neck_geometry']['checks'][0]
    assert check['status'] == 'partial' and check['roi_to_neck_axis_angle_deg'] is None
    assert check['soft_tissue_sides']['both_sides_have_candidate_pixels']
