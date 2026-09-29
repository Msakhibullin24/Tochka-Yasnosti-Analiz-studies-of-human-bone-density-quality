import csv
import json
from pathlib import Path

import cv2
import pytest

from conftest import synthetic_spine, write_dicom
from prepare_anatomy_review import prepare


def results_file(path, uid='1.2.3', source='a.dcm', width=300):
    row = {'image_uid': uid, 'path_to_file': source, 'processing_status': 'Success',
           'duplicate_of': '', 'image_width': width, 'image_height': 300,
           'anatomy_assessment': json.dumps({'landmarks': [{'name': 'Th12', 'points': [[123, 234]]}]}),
           'quality_class': 1, 'violation_type': 'model decision must stay hidden'}
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=row); writer.writeheader(); writer.writerow(row)
    return path


def test_blind_package_checks_source_and_preserves_geometry(tmp_path):
    source = tmp_path/'in'; source.mkdir()
    write_dicom(source/'a.dcm', synthetic_spine(), sop_uid='1.2.3')
    results = results_file(tmp_path/'results.csv')
    output = tmp_path/'review'
    assert prepare(results, source, output)['cases'] == 1
    reference = json.loads((output/'template.json').read_text())
    case = reference['cases'][0]
    assert case['landmarks']['Th12'] == {'visible': None, 'point': None}
    assert cv2.imread(str(output/case['image']), 0).shape == (300, 300)
    assert len(case['source_sha256']) == len(case['pixel_sha256']) == 64
    page = (output/'index.html').read_text()
    assert 'model decision must stay hidden' not in page and '123, 234' not in page
    assert 'quality_class' not in page and 'points' not in reference
    assert reference['independent_of_predictions'] is False
    with pytest.raises(ValueError): prepare(results, source, output)


@pytest.mark.parametrize('uid,width', [('wrong', 300), ('1.2.3', 299)])
def test_source_mismatch_does_not_publish_partial_package(tmp_path, uid, width):
    source = tmp_path/'in'; source.mkdir()
    write_dicom(source/'a.dcm', synthetic_spine(), sop_uid='1.2.3')
    results = results_file(tmp_path/'results.csv', uid=uid, width=width)
    with pytest.raises(ValueError, match='identity or image geometry'):
        prepare(results, source, tmp_path/'review')
    assert not (tmp_path/'review').exists()
    assert not list(tmp_path.glob('.anatomy-review-*'))


def test_source_symlink_escape_is_rejected(tmp_path):
    source = tmp_path/'in'; source.mkdir()
    write_dicom(tmp_path/'outside.dcm', synthetic_spine(), sop_uid='1.2.3')
    (source/'a.dcm').symlink_to(tmp_path/'outside.dcm')
    with pytest.raises(ValueError, match='escapes'):
        prepare(results_file(tmp_path/'results.csv'), source, tmp_path/'review')


def test_reference_evaluation_requires_exact_source_and_results_provenance(tmp_path):
    from evaluate_anatomy_review import evaluate_review
    source = tmp_path/'in'; source.mkdir()
    write_dicom(source/'a.dcm', synthetic_spine(), sop_uid='1.2.3')
    results = results_file(tmp_path/'results.csv')
    prepare(results, source, tmp_path/'review')
    reference = tmp_path/'review/template.json'
    with pytest.raises(ValueError, match='independently reviewed'):
        evaluate_review(results, source, reference)
    labels = json.loads(reference.read_text())
    labels['cases'][0]['source_sha256'] = 'incorrect'
    reference.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match='Source DICOM differs'):
        evaluate_review(results, source, reference)
    results.write_text(results.read_text()+'\n')
    with pytest.raises(ValueError, match='reviewed results artifact'):
        evaluate_review(results, source, reference)


def test_confirmed_synthetic_reference_reports_missing_landmarks_and_abstention(tmp_path):
    from evaluate_anatomy_review import evaluate_review
    source = tmp_path/'in'; source.mkdir()
    write_dicom(source/'a.dcm', synthetic_spine(), sop_uid='1.2.3')
    results = results_file(tmp_path/'results.csv')
    with results.open() as stream:
        row = next(csv.DictReader(stream))
    row.update(pixel_mm_x='0.6', pixel_mm='1.05', pixel_mm_source='organizer_ge_lunar',
               projection_assessment=json.dumps({'value': 'unknown'}))
    with results.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=row); writer.writeheader(); writer.writerow(row)
    prepare(results, source, tmp_path/'review')
    reference = tmp_path/'review/template.json'
    labels = json.loads(reference.read_text())
    labels.update(reviewer='synthetic test reviewer', annotation_source='synthetic fixture', independent_of_predictions=True)
    case = labels['cases'][0]; case.update(status='confirmed', projection='frontal')
    case['th12_full_body'] = {'status': 'unassessable', 'quadrilateral': None}
    for name in case['named_regions']:
        case['named_regions'][name] = {'visible': False, 'polygon': None}
    for name in case['landmarks']:
        case['landmarks'][name] = {'visible': True, 'point': [123, 234]}
    reference.write_text(json.dumps(labels))
    report = evaluate_review(results, source, reference)
    assert report['source_provenance_verified'] is True
    assert report['landmarks']['Th12']['median_mm'] == 0
    assert report['landmarks']['iliac_crest_left']['detected'] == 0
    assert report['projection']['abstained'] == 1
    assert report['requirements_complete'] is False


def test_named_mask_evaluation_keeps_missing_and_wrong_numbering_in_metrics():
    from evaluate_anatomy_review import evaluate_regions
    from dxaqc.mask_raster import encode_raster
    import numpy as np
    mask = np.zeros((20, 20), bool); mask[2:9, 2:9] = True
    row = {'image_uid': '1.2', 'image_width': '20', 'image_height': '20',
           'learned_anatomy': json.dumps({'status': 'evaluated', 'regions': [
               {'name': 'L2', 'raster': encode_raster(mask)}]})}
    case = {'image_uid': '1.2', 'landmarks': {'Th12': {}}, 'named_regions': {
        'L1': {'visible': True, 'polygon': [[2, 2], [8, 2], [8, 8], [2, 8]]},
        'L2': {'visible': True, 'polygon': [[11, 11], [17, 11], [17, 17], [11, 17]]},
        'L3': {'visible': False, 'polygon': None}, 'L4': {'visible': False, 'polygon': None}}}
    metrics, _ = evaluate_regions([row], [case])
    assert metrics['L1']['missing_predictions'] == 1
    assert metrics['L1']['mean_dice_including_missing'] == 0
    assert metrics['L2']['mean_dice_including_missing'] == 0
    case['named_regions']['L2']['polygon'] = case['named_regions']['L1']['polygon']
    metrics, _ = evaluate_regions([row], [case])
    assert metrics['L2']['mean_dice_including_missing'] == 1


def test_hip_rotation_requires_explicit_subtype_and_does_not_invent_accuracy():
    from evaluate_anatomy_review import evaluate_regions
    row = {'image_uid': '1.2', 'image_width': '20', 'image_height': '20'}
    case = {'image_uid': '1.2', 'landmarks': {'femoral_neck': {}},
            'named_regions': {'femoral_neck_roi': {'visible': False, 'polygon': None}},
            'rotation': None}
    with pytest.raises(ValueError, match='rotation'): evaluate_regions([row], [case])
    case['rotation'] = 'insufficient'
    _, rotation = evaluate_regions([row], [case])
    assert rotation['expert_label_counts'] == {'insufficient': 1}
    assert rotation['prediction_accuracy'] is None


def test_concave_bone_contours_preserve_notch_and_reject_crossings():
    from review_geometry import contour_mask
    contour = [[2, 2], [12, 2], [12, 12], [8, 12], [8, 6], [6, 6], [6, 12], [2, 12]]
    mask = contour_mask(contour, (20, 20))
    assert mask[4, 7] and not mask[10, 7]
    for invalid in ([[2, 2], [12, 12], [2, 12], [12, 2]],
                    [[2, 2], [12, 2], [6, 2], [12, 12], [2, 12]],
                    [[2, 2], [12, 2], [12, 12], [2, 2]]):
        with pytest.raises(ValueError): contour_mask(invalid, (20, 20))


def test_axis_evaluation_is_undirected_and_uses_anisotropic_physical_scale():
    from evaluate_anatomy_review import evaluate_axes
    row = {'image_uid': '1.2', 'image_width': '20', 'image_height': '20',
           'pixel_mm_x': '2', 'pixel_mm': '1', 'pixel_mm_source': 'PixelSpacing',
           'learned_anatomy': json.dumps({'status': 'evaluated', 'regions': [
               {'name': 'femoral_neck_axis', 'points': [[8, 8], [2, 2]]}]})}
    case = {'image_uid': '1.2', 'landmarks': {'femoral_neck': {}},
            'axes': {'femoral_neck_axis': {'visible': True, 'points': [[2, 2], [8, 8]]}}}
    metric = evaluate_axes([row], [case])
    assert metric['mean_endpoint_errors_mm'] == 0
    assert metric['mean_angle_errors_deg'] == pytest.approx(0, abs=1e-6)
    row['learned_anatomy'] = json.dumps({'status': 'evaluated', 'regions': [
        {'name': 'femoral_neck_axis', 'points': [[2, 2], [8, 2]]}]})
    assert evaluate_axes([row], [case])['mean_angle_errors_deg'] == pytest.approx(26.565051177)
    row['pixel_mm_source'] = 'device_default'
    assert evaluate_axes([row], [case])['unscaled_predictions'] == 1
    row['learned_anatomy'] = '{}'
    metric = evaluate_axes([row], [case])
    assert metric['missing_predictions'] == 1 and metric['prediction_coverage'] == 0
    assert metric['mean_angle_errors_deg'] is None
    case['axes']['femoral_neck_axis']['points'] = [[2, 2], [2, 2]]
    with pytest.raises(ValueError, match='endpoints'): evaluate_axes([row], [case])


def test_v4_package_has_independent_axis_named_contours_and_th12_status(tmp_path):
    source = tmp_path/'in'; source.mkdir()
    write_dicom(source/'a.dcm', synthetic_spine(), sop_uid='1.2.3')
    results = results_file(tmp_path/'results.csv')
    with results.open() as stream: row = next(csv.DictReader(stream))
    row['anatomy_assessment'] = json.dumps({'landmarks': [{'name': 'femoral_neck', 'points': [[20, 30]]}]})
    with results.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=row); writer.writeheader(); writer.writerow(row)
    prepare(results, source, tmp_path/'review')
    reference = json.loads((tmp_path/'review/template.json').read_text())
    assert reference['review_package_version'] == 4
    case = reference['cases'][0]
    assert set(case['named_regions']) == {'femoral_neck_roi', 'femur', 'greater_trochanter', 'lesser_trochanter', 'ischium'}
    assert case['axes'] == {'femoral_neck_axis': {'visible': None, 'points': None}}


def test_th12_full_body_reference_separates_area_height_and_unassessable():
    from evaluate_anatomy_review import evaluate_th12_full_body
    rows = [{'image_uid': 'spine', 'image_width': '20', 'image_height': '20'},
            {'image_uid': 'other', 'image_width': '20', 'image_height': '20'}]
    visible = {'image_uid': 'spine', 'landmarks': {'Th12': {}},
               'named_regions': {'Th12': {'visible': True}},
               'th12_full_body': {'status': 'assessable', 'quadrilateral': [[2, -10], [12, -10], [12, 10], [2, 10]]}}
    absent = {'image_uid': 'other', 'landmarks': {'Th12': {}},
              'named_regions': {'Th12': {'visible': False}},
              'th12_full_body': {'status': 'unassessable', 'quadrilateral': None}}
    result = evaluate_th12_full_body(rows, [visible, absent])
    assert result['assessable'] == result['unassessable'] == 1
    assert result['cases'][0]['visible_vertical_extent_fraction'] == pytest.approx(.525)
    assert result['cases'][1]['visible_vertical_extent_fraction'] is None
    assert result['prediction_accuracy'] is None
    for broken in ({'status': 'unassessable', 'quadrilateral': [[2, -10], [12, -10], [12, 10], [2, 10]]},
                   {'status': 'assessable', 'quadrilateral': [[2, 2], [12, 12], [2, 12], [12, 2]]},
                   {'status': None, 'quadrilateral': None}):
        visible['th12_full_body'] = broken
        with pytest.raises(ValueError): evaluate_th12_full_body(rows, [visible])
    visible['th12_full_body'] = {'status': 'assessable', 'quadrilateral': [[2, 21], [12, 21], [12, 30], [2, 30]]}
    with pytest.raises(ValueError, match='conflicts'): evaluate_th12_full_body(rows, [visible])


def test_projection_reference_reports_unsafe_frontal_errors_by_region():
    from dxaqc.validate_anatomy import evaluate
    rows = []
    cases = []
    for uid, group, prediction, truth in [('spine', 'spine', 'unknown', 'frontal'),
                                           ('hip', 'hip', 'frontal', 'lateral')]:
        names = (['Th12', 'iliac_crest_left', 'iliac_crest_right'] if group == 'spine'
                 else ['greater_trochanter', 'lesser_trochanter', 'femoral_neck', 'ischium'])
        rows.append({'image_uid': uid, 'anatomy_assessment': json.dumps({'landmarks': [
            {'name': name, 'points': []} for name in names]}),
            'projection_assessment': json.dumps({'value': prediction}),
            'pixel_mm_x': '1', 'pixel_mm': '1', 'pixel_mm_source': 'PixelSpacing',
            'image_width': '40', 'image_height': '40'})
        cases.append({'image_uid': uid, 'status': 'confirmed', 'projection': truth,
                      'landmarks': {name: {'visible': False, 'point': None} for name in names}})
    reference = {'version': 1, 'coordinate_system': 'original_pixel_centres',
                 'reviewer': 'fixture', 'annotation_source': 'fixture',
                 'independent_of_predictions': True, 'cases': cases}
    projection = evaluate(rows, reference)['projection']
    assert projection['false_frontal_on_unsupported'] == 1
    assert projection['by_region']['hip']['confusion']['lateral']['frontal'] == 1
    assert projection['by_region']['spine']['abstained'] == 1
    assert projection['by_region']['hip']['accuracy_including_abstentions'] == 0


def test_expert_neck_geometry_uses_independent_axis_and_named_contours():
    from evaluate_anatomy_review import evaluate_expert_neck_roi_geometry
    row = {'image_uid': 'hip', 'image_width': '40', 'image_height': '40',
           'pixel_mm_x': '2', 'pixel_mm': '1', 'pixel_mm_source': 'PixelSpacing'}
    case = {'image_uid': 'hip', 'landmarks': {'femoral_neck': {}},
            'named_regions': {
                'femoral_neck_roi': {'visible': True, 'polygon': [[5, 10], [24, 10], [24, 16], [5, 16]]},
                'femur': {'visible': True, 'polygon': [[10, 8], [19, 8], [19, 18], [10, 18]]},
                'greater_trochanter': {'visible': True, 'polygon': [[8, 11], [12, 11], [12, 14], [8, 14]]},
                'lesser_trochanter': {'visible': False, 'polygon': None},
                'ischium': {'visible': True, 'polygon': [[25, 20], [30, 20], [30, 25], [25, 25]]}},
            'axes': {'femoral_neck_axis': {'visible': True, 'points': [[14, 5], [14, 25]]}}}
    report = evaluate_expert_neck_roi_geometry([row], [case])
    assert report['measured'] == 1 and report['unassessable'] == 0
    measured = report['cases'][0]
    assert measured['deviation_from_perpendicular_deg'] == pytest.approx(0, abs=1e-6)
    assert measured['soft_tissue_sides']['both_sides_have_candidate_pixels'] is True
    assert measured['reference_structure_overlap_pixels']['greater_trochanter'] > 0
    assert measured['reference_structure_overlap_pixels']['ischium'] == 0
    assert measured['clinical_verdict'] == 'undetermined'
    case['axes']['femoral_neck_axis']['visible'] = False
    case['axes']['femoral_neck_axis']['points'] = None
    assert evaluate_expert_neck_roi_geometry([row], [case])['unassessable'] == 1
    case['axes']['femoral_neck_axis'] = {'visible': True, 'points': [[14, 5], [14, 25]]}
    row['pixel_mm_source'] = 'device_default'
    assert evaluate_expert_neck_roi_geometry([row], [case])['unscaled'] == 1
