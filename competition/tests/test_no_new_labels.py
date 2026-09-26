import json

import numpy as np
import pytest
import torch

from anatomy_training_data import aasce_target, dxa_spine_target, split_records
from dxaqc.anatomical_roi import compare_roi, propose_roi, compare_source_rois
from dxaqc.decision import decide
from dxaqc.joint_quality import JointQualityModel, select_quality_threshold
from dxaqc.learned_anatomy import decode_masks, mask_regions, numbered_axis, roi_proposals, AnatomyPortfolio
from dxaqc.synthetic_defects import rotate_pair, crop_pair, artifact_pair, rotate_expand
from render_femur_curriculum import rotation_matrix, project
from train_hip_masks import loss_with_unknown


def test_published_corner_order_becomes_nonintersecting_named_targets():
    points = np.array([
        p for i in range(17) for p in ([20, i*12+2], [35, i*12+2], [20, i*12+10], [35, i*12+10])])
    target = aasce_target(points, (220, 60))
    assert target[3, 25] == 1 and target[147, 25] == 13
    with pytest.raises(ValueError):
        aasce_target(points[::-1], (220, 60))


def test_dxa_unannotated_anatomy_remains_ignored():
    masks = {}
    for i in range(1, 5):
        a = np.zeros((100, 70), bool); a[i*15:i*15+8, 25:40] = True
        masks[f'Lumbar_{i}_bone_area'] = a
    target = dxa_spine_target(masks, (100, 70))
    assert target[0, 0] == -100 and target[16, 26] == 13
    rows = [{'source': source, 'group': f'{source}:{i}'} for source in ('a', 'b') for i in range(10)]
    held = split_records(rows)
    assert held.sum() == 4
    assert not {r['group'] for r, t in zip(rows, held) if t} & {r['group'] for r, t in zip(rows, held) if not t}


def test_paired_transforms_keep_geometry_and_do_not_invent_other_targets():
    pixels = np.zeros((80, 60), np.uint8); pixels[20:55, 25:35] = 160
    masks = {'L1': pixels > 0}
    altered, mask, meta = rotate_pair(pixels, masks, 15, (.5, 2.))
    assert altered.shape == pixels.shape and mask['L1'].any() and meta['qc_targets'] == {}
    _, cropped, meta = crop_pair(pixels, masks, (0, 0, 60, 35))
    assert 0 < meta['retained_mask_fraction']['L1'] < 1
    assert cropped['L1'].shape == (35, 60)
    altered, mask, meta = artifact_pair(pixels, masks, (2, 2, 10, 10))
    assert mask['injected_artifact'].sum() == 64
    assert meta['qc_targets'] == {'spine_artifact': 1} and not meta['implant_ground_truth']
    expanded, affine = rotate_expand(pixels, 15, (.5, 2.))
    corners = np.c_[np.array([[0, 0], [59, 0], [0, 79], [59, 79]]), np.ones(4)] @ affine.T
    assert (corners >= 0).all()
    assert (corners[:, 0] < expanded.shape[1]).all() and (corners[:, 1] < expanded.shape[0]).all()


def test_roi_measurements_detect_displacement_and_keep_proposals_unconfirmed():
    mask = np.zeros((60, 60), bool); mask[15:35, 20:40] = True
    proposal = propose_roi(mask, purpose='L1', reference_origin='published')
    correct = compare_roi(proposal['polygon'], mask, purpose='L1', reference_origin='published')
    moved = np.asarray(proposal['polygon'])+[8, 0]
    shifted = compare_roi(moved, mask, purpose='L1', reference_origin='published')
    assert correct['reference_coverage'] == 1 and shifted['iou'] < correct['iou']
    assert proposal['requires_confirmation'] and correct['clinical_verdict'] == 'undetermined'
    with pytest.raises(ValueError):
        compare_roi(moved, mask, purpose='', reference_origin='published')


def test_3d_rotation_is_rigid_and_preserves_shaft_axis():
    axis = np.array([1., 2., 8.]); axis /= np.linalg.norm(axis)
    rotation = rotation_matrix(axis, 30)
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.allclose(rotation @ axis, axis)
    mask = np.zeros((25, 25, 40), bool); mask[10:15, 10:15, 4:36] = True
    volume = np.full(mask.shape, -1000., dtype=np.float32); volume[mask] = 800
    image, projection, meta = project(volume, mask, 15, 'frontal')
    assert image.shape == projection.shape and projection.any()
    assert 'shaft_axis_ras' in meta


def test_missing_hip_labels_contribute_no_gradient():
    logits = torch.zeros((1, 3, 4, 4), requires_grad=True)
    target = torch.full_like(logits, -1.); target[:, 2] = 1.
    loss_with_unknown(logits, target).backward()
    assert not logits.grad[:, :2].any() and logits.grad[:, 2].abs().sum() > 0


def test_named_decoding_handles_empty_masks_and_order_ambiguity():
    logits = np.zeros((18, 40, 40)); logits[0] = 1
    assert not mask_regions(decode_masks(logits, (80, 60)))
    regions = [{'name': f'L{i}', 'center': [20+i, 15*i], 'polygon': [[1,1],[5,1],[5,5],[1,5]]} for i in range(1, 5)]
    assert numbered_axis(regions, 2., 1.) > 0
    assert numbered_axis(regions[:2], 2., 1.) is None
    assert all(p['requires_confirmation'] for p in roi_proposals(regions))


def test_joint_review_is_supervised_and_preserves_identified_baseline():
    model = JointQualityModel('spine', 'advisory')
    model.fit([{}]*8, np.array([[0.]]*4+[[1.]]*4), ['normal']*4+['spine_artifact']*4)
    baseline = decide('spine', .8, {'spine_axis': 0., 'spine_coverage': 0., 'spine_artifact': 0.},
                      {'spine_abs_angle_deg': 2.}, .5, {})
    reviewed = model.review_untyped(baseline, {}, [[0.]])
    assert reviewed['quality'] == 0 and not reviewed['violations']
    assert baseline['quality'] == 1
    positive = model.review_untyped(baseline, {}, [[1.]])
    assert positive['quality'] == 1 and positive['violations'] == ['spine_artifact']
    identified = {**baseline, 'violations': ['spine_artifact']}
    assert model.review_untyped(identified, {}, [[0.]]) is identified


def test_profile_rejects_model_escape_and_threshold_arrays_are_checked(tmp_path):
    path = tmp_path/'anatomy.json'
    path.write_text(json.dumps({'schema_version': 1, 'models': {'spine': {'path': '../outside.pt', 'sha256': 'bad'}}}))
    with pytest.raises(ValueError, match='escapes'):
        AnatomyPortfolio(path)
    with pytest.raises(ValueError):
        select_quality_threshold([1], [float('nan')], [False])
    threshold = select_quality_threshold([0, 1], [.1, .8], [False, False])
    assert .1 < threshold <= .8


def test_source_roi_raster_preserves_holes_and_unknown_purpose():
    region = {'name': 'L1', 'polygon': [[10,10],[29,10],[29,29],[10,29]]}
    roi = {'id': 'roi', 'purpose': 'L1', 'raster': {'shape': [20,20], 'origin': [10,10], 'runs': [[0,200],[220,400]]}}
    source = {'status': 'present', 'rois': [roi]}
    result = compare_source_rois(source, [region], (60,60))
    assert result['checks'][0]['reference_coverage'] == .95
    assert result['checks'][0]['clinical_verdict'] == 'undetermined'
    assert compare_source_rois({'status':'absent','rois':[]}, [], (60,60))['status'] == 'not_applicable'
    roi['purpose'] = 'unknown'
    assert compare_source_rois(source, [region], (60,60))['checks'][0]['status'] == 'unavailable'


def test_lumbar_crop_preserves_published_level_names():
    from refine_lumbar_masks import lumbar_crop
    image = np.arange(120*60).reshape(120, 60)
    target = np.zeros((120, 60), np.int64)
    target[10:20, 20:30] = 1
    for level in range(12, 18):
        target[35+(level-12)*10:40+(level-12)*10, 20:30] = level
    cropped, annotation = lumbar_crop(image, target)
    assert cropped.shape == annotation.shape
    assert set(np.unique(annotation)) == {0, 12, 13, 14, 15, 16, 17}
    assert (annotation == 13).sum() == (target == 13).sum()
    with pytest.raises(ValueError):
        lumbar_crop(image, np.zeros_like(target))


def test_polarity_stability_is_not_an_accuracy_metric():
    from evaluate_anatomy_transfer import paired_stability
    a = np.zeros((6, 6), np.uint8)
    assert paired_stability(a, a) is None
    a[1:3, 1:3] = 13
    assert paired_stability(a, a.copy()) == 1
    b = np.zeros_like(a); b[3:5, 3:5] = 13
    assert paired_stability(a, b) == 0
    b = a.copy(); b[b == 13] = 14
    assert paired_stability(a, b) == 0
    with pytest.raises(ValueError):
        paired_stability(a, a[:3])


def test_transfer_audit_source_paths_stay_inside_dataset(tmp_path):
    from evaluate_anatomy_transfer import resolve_source
    root = tmp_path/'dataset'; root.mkdir()
    source = root/'image.dcm'; source.write_bytes(b'dicom')
    assert resolve_source(root, 'image.dcm') == source
    outside = tmp_path/'outside.dcm'; outside.write_bytes(b'dicom')
    with pytest.raises(ValueError):
        resolve_source(root, '../outside.dcm')
    with pytest.raises(ValueError):
        resolve_source(root, 'missing.dcm')
    (root/'link.dcm').symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_source(root, 'link.dcm')


def test_hip_roi_curriculum_keeps_parent_split_and_unknown_qc():
    from build_defect_curriculum import roi_displacements
    mask = np.zeros((80, 90), bool); mask[20:45, 20:40] = True
    cases = roi_displacements({'femoral_neck_roi': mask}, 'patient-a', 'test', 90, 'source-sha', 'annotation-sha')
    assert len(cases) == 1
    assert cases[0]['parent_group'] == 'patient-a' and cases[0]['split'] == 'test'
    assert cases[0]['qc_targets'] == {}
    assert cases[0]['shifted']['iou'] < cases[0]['original']['iou']
    edge = np.zeros_like(mask); edge[20:45, 75:89] = True
    assert roi_displacements({'femoral_neck_roi': edge}, 'patient-a', 'test', 90, 'sha', 'sha') == []


def test_transfer_coverage_deduplicates_pixels_without_hiding_disagreement():
    from evaluate_anatomy_transfer import coverage_summary
    a = {'pixel_sha256': 'same-pixels', 'axis_candidates_deg': {'original': 2., 'inverted': None}}
    b = {'pixel_sha256': 'other-pixels', 'axis_candidates_deg': {'original': None, 'inverted': 3.}}
    result = coverage_summary([a, a.copy(), b])
    assert result['unique_pixel_rasters'] == 2
    assert result['unique_pixel_axis_available'] == {'original': 1, 'inverted': 1}
    conflicting = {'pixel_sha256': 'same-pixels', 'axis_candidates_deg': {'original': None, 'inverted': None}}
    with pytest.raises(ValueError):
        coverage_summary([a, conflicting])


def test_anatomy_photometry_preserves_pixel_positions_and_source():
    from refine_anatomy_photometry import photometric_variant
    raster = np.array([[0, 32, 128], [64, 192, 255]], np.uint8)
    original = raster.copy()
    for kind in ('inverse', 'dark', 'bright'):
        result = photometric_variant(raster, kind)
        assert result.shape == raster.shape and result.dtype == np.uint8
        assert np.array_equal(raster, original)
    assert np.array_equal(photometric_variant(photometric_variant(raster, 'inverse'), 'inverse'), raster)
    assert photometric_variant(raster, 'dark')[0, 2] < raster[0, 2]
    assert photometric_variant(raster, 'bright')[0, 2] > raster[0, 2]
    with pytest.raises(ValueError):
        photometric_variant(raster.astype(float), 'dark')
    with pytest.raises(ValueError):
        photometric_variant(raster, 'unknown')


def test_anatomy_candidate_review_cannot_hide_source_regression_or_change_cohort():
    from review_anatomy_candidate import compare_original_cases
    before = [dict(source='a', view='original', group='patient1', polarity='original', dice=.4),
              dict(source='dxa', view='original', group='patient2', polarity='original', dice=.9)]
    after = [dict(before[0], dice=.8), dict(before[1], dice=.8)]
    result = compare_original_cases(before, after)
    assert result['ordinary_field_regressions'] == ['dxa/original']
    assert result['ordinary_fields_non_decreasing'] is False
    assert compare_original_cases(before, before)['ordinary_fields_non_decreasing'] is True
    with pytest.raises(ValueError):
        compare_original_cases(before, [dict(after[0], group='different'), after[1]])
    with pytest.raises(ValueError):
        compare_original_cases(before, after+[after[0]])
    with pytest.raises(ValueError):
        compare_original_cases(before, [dict(after[0], dice=float('nan')), after[1]])


def test_learned_mask_and_roi_comparison_preserve_holes_and_concavity():
    from dxaqc.mask_raster import decode_raster
    labels = np.zeros((50, 50), np.uint8)
    labels[10:40, 10:40] = 13
    labels[20:30, 20:30] = 0
    labels[10:20, 30:40] = 0
    region = mask_regions(labels)[0]
    exact = decode_raster(region['raster'], labels.shape)
    assert np.array_equal(exact, labels == 13)
    assert region['pixels'] == int(exact.sum())
    source = {'status': 'extracted', 'rois': [{'id': 'L1', 'purpose': 'L1', 'raster': region['raster']}]}
    check = compare_source_rois(source, [region], labels.shape)['checks'][0]
    assert check['iou'] == 1 and check['outside_reference_fraction'] == 0
    assert check['clinical_verdict'] == 'undetermined'


def test_raster_codec_checks_bounds_before_allocation_and_clips_origins():
    from dxaqc.mask_raster import encode_raster, decode_raster
    raster = encode_raster(np.ones((3, 4), bool), (-1, 2))
    assert decode_raster(raster, (4, 4)).sum() == 4
    for altered in (dict(raster, shape=[100000, 100000]), dict(raster, runs=[[-1, 2]]),
                    dict(raster, runs=[[0, 5], [4, 8]]), dict(raster, runs=[[0, 13]]),
                    dict(raster, origin=[.5, 2])):
        with pytest.raises(ValueError):
            decode_raster(altered, (4, 4))


def test_largest_named_component_uses_pixel_area_and_matching_contour():
    from dxaqc.mask_raster import decode_raster
    labels = np.zeros((120,160),np.uint8)
    labels[10:90,10:90] = 13; labels[12:88,12:88] = 0
    labels[30:70,100:130] = 13
    region = mask_regions(labels)[0]
    assert region['bbox_xyxy'] == [100,30,130,70]
    assert region['pixels'] == 1200
    assert decode_raster(region['raster'], labels.shape).sum() == 1200
    assert min(p[0] for p in region['polygon']) == 100


def test_hip_bundle_rejects_wrong_feature_grid_and_irrelevant_regions(tmp_path):
    from dxaqc.learned_anatomy import LearnedHipAnatomy, HIP_CLASSES
    from dxaqc.embedding import WEIGHTS_SHA256
    path = tmp_path/'invalid.pt'
    torch.save({'schema_version':1,'classes':HIP_CLASSES,'encoder_sha256':WEIGHTS_SHA256,
                'input_size':320,'grid':20,'clinical_validation':False,'state_dict':{}},path)
    with pytest.raises(ValueError):
        LearnedHipAnatomy(path)
    model = LearnedHipAnatomy.__new__(LearnedHipAnatomy)
    assert model.predict(np.zeros((10,10),np.uint8),'spine')['status'] == 'not_applicable'


def test_workflow_profile_commits_environment_only_after_all_checks(tmp_path,monkeypatch):
    import hashlib
    import os
    from dxaqc.workflow_profile import configure_profile, ARTIFACT_ENV
    baseline=tmp_path/'baseline'; baseline.write_bytes(b'baseline')
    for name in ARTIFACT_ENV | {'DXAQC_WORKFLOW_PROFILE'}:
        monkeypatch.setenv(name,'before')
    artifact=tmp_path/'model'; artifact.write_bytes(b'model')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    profile={'schema_version':1,'profile_id':'test','clinical_validation':False,
             'baseline_bundle_sha256':digest(baseline),
             'artifacts':{name:{'path':'model','sha256':digest(artifact)} for name in ARTIFACT_ENV}}
    path=tmp_path/'profile.json'; path.write_text(json.dumps(profile))
    configure_profile(path,baseline)
    assert os.environ['DXAQC_WORKFLOW_PROFILE']==str(path)
    assert all(os.environ[name]==str(artifact) for name in ARTIFACT_ENV)
    for name in ARTIFACT_ENV | {'DXAQC_WORKFLOW_PROFILE'}:
        monkeypatch.setenv(name,'before')
    profile['artifacts'][next(iter(ARTIFACT_ENV))]['sha256']='wrong'
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='checksum'):
        configure_profile(path,baseline)
    assert all(os.environ[name]=='before' for name in ARTIFACT_ENV | {'DXAQC_WORKFLOW_PROFILE'})
    profile['artifacts']={name:{'path':'../escape','sha256':'wrong'} for name in ARTIFACT_ENV}
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='escapes'):
        configure_profile(path,baseline)
    assert all(os.environ[name]=='before' for name in ARTIFACT_ENV)
