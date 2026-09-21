import csv
import json

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.pixels.utils import pack_bits

from conftest import synthetic_spine, write_dicom
from dxaqc.source_roi import extract_roi, contains
from dxaqc.dicom_io import read_dxa
from dxaqc.pipeline import run_batch, Options


def add_overlay(ds, mask, origin=(1,1), kind='R'):
    for element,vr,value in [(0x0010,'US',mask.shape[0]),(0x0011,'US',mask.shape[1]),(0x0040,'CS',kind),
                              (0x0050,'SS',list(origin)),(0x0100,'US',1),(0x0102,'US',0),(0x3000,'OW',pack_bits(mask))]:
        ds.add_new((0x6000,element),vr,value)


def graphic_state(uid, study_uid):
    ds=Dataset();ds.SOPClassUID='1.2.840.10008.5.1.4.1.1.11.1';ds.SOPInstanceUID='1.2.3.99';ds.StudyInstanceUID=study_uid
    ref=Dataset();ref.ReferencedSOPClassUID='1.2.840.10008.5.1.4.1.1.1';ref.ReferencedSOPInstanceUID=uid
    annotation=Dataset();annotation.GraphicLayer='ROI';annotation.ReferencedImageSequence=[ref]
    graphic=Dataset();graphic.GraphicAnnotationUnits='PIXEL';graphic.GraphicDimensions=2;graphic.GraphicType='POLYLINE'
    graphic.GraphicData=[10.5,20.5,60.5,20.5,60.5,80.5,10.5,80.5,10.5,20.5];graphic.NumberOfGraphicPoints=5
    annotation.GraphicObjectSequence=[graphic];ds.GraphicAnnotationSequence=[annotation]
    return ds


def test_overlay_origin_holes_and_non_roi_graphics():
    ds=Dataset();mask=np.ones((20,30),np.uint8);mask[5:15,10:20]=0
    add_overlay(ds,mask,origin=(11,21))
    result=extract_roi(ds,(100,100),'')
    assert result['status']=='extracted'
    roi=result['rois'][0]
    assert roi['bounds']==[20,10,49,29]
    assert roi['area_pixels']==500
    assert contains(roi,(22,12)) and not contains(roi,(35,20))
    ds[0x6000,0x0040].value='G'
    assert not extract_roi(ds,(100,100),'')['rois']
    ds[0x6000,0x0040].value='R';ds[0x6000,0x0050].value=[95,95]
    invalid=extract_roi(ds,(100,100),'')
    assert invalid['status']=='unavailable' and not invalid['rois']


def test_graphics_coordinates_source_reference_and_unsupported_units():
    ds=graphic_state('1.2.3','1.2')
    result=extract_roi(ds,(100,100),'1.2.3')
    assert result['rois'][0]['bounds']==[10,20,60,80]
    assert not extract_roi(ds,(100,100),'1.2.4')['rois']
    ds.GraphicAnnotationSequence[0].GraphicObjectSequence[0].GraphicAnnotationUnits='DISPLAY'
    assert extract_roi(ds,(100,100),'1.2.3')['status']=='unavailable'
    ds.GraphicAnnotationSequence[0].GraphicLayer='DECORATION'
    assert not extract_roi(ds,(100,100),'1.2.3')['rois']


def test_pixel_duplicates_do_not_share_source_roi_and_gsps_is_not_an_image(tmp_path,bundle_available):
    src=tmp_path/'input';src.mkdir()
    first=write_dicom(src/'a.dcm',synthetic_spine(),study_uid='1.2',sop_uid='1.2.3')
    write_dicom(src/'b.dcm',synthetic_spine(),study_uid='1.2',sop_uid='1.2.4')
    state=graphic_state('1.2.3','1.2');state.file_meta=pydicom.dataset.FileMetaDataset()
    state.file_meta.TransferSyntaxUID=pydicom.uid.ExplicitVRLittleEndian
    state.save_as(src/'state.dcm',enforce_file_format=True)
    summary=run_batch(src,tmp_path/'out',Options(explanations=False,keep_explanation_dir=True))
    assert summary['files']==summary['success']==2
    with open(tmp_path/'out'/'results.csv',encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
    a,b=rows
    assert json.loads(a['source_roi_assessment'])['status']=='extracted'
    assert json.loads(b['source_roi_assessment'])['status']=='absent'
    assert b['duplicate_of']=='1.2.3' and a['quality_class']==b['quality_class']
    detail=json.loads((tmp_path/'out'/'images'/f"{a['row_id']}.json").read_text())
    assert detail['assessment']['complete'] is False
    assert all(not x['verified'] for x in detail['assessment']['anatomy']['landmarks'])
    assert read_dxa(first).source_roi['status']=='absent'


def test_hip_candidates_are_mirrored_back_and_missing_evidence_abstains():
    from dxaqc.anatomy import detect_landmarks,assess_projection
    pixels=np.zeros((120,100),np.uint8)
    overlay={'greater_trochanter_top':[(10,20)],'ischium_bottom':[(80,80)]}
    right=detect_landmarks(pixels,'hip_right',overlay)
    left=detect_landmarks(pixels[:,::-1],'hip_left',overlay)
    for a,b in zip(right['landmarks'],left['landmarks']):
        assert b['points']==[[99-x,y] for x,y in a['points']]
    assert assess_projection(pixels,'hip_right',right)['status']=='undetermined'
    assert all(not x['verified'] for x in right['landmarks'])


def test_spine_bodies_are_not_silently_numbered_from_crop_edge():
    from dxaqc.anatomy import detect_landmarks
    result=detect_landmarks(synthetic_spine(), 'spine', {})
    th12=next(x for x in result['landmarks'] if x['name']=='Th12')
    assert not th12['points'] and th12['status']=='not_localized'
    assert th12['verified'] is False


def test_reference_evaluator_rejects_templates_and_counts_abstention_and_missing_landmark():
    import pytest
    from dxaqc.validate_anatomy import template,evaluate
    row={'image_uid':'1.2','path_to_file':'a.dcm','image_width':'100','image_height':'100','pixel_mm_x':'2','pixel_mm':'1','pixel_mm_source':'PixelSpacing',
         'anatomy_assessment':json.dumps({'landmarks':[{'name':'Th12','points':[]},{'name':'iliac_crest_left','points':[[12,20]]},{'name':'iliac_crest_right','points':[[50,80]]}]}),
         'projection_assessment':json.dumps({'value':'unknown'})}
    reference=template([row])
    with pytest.raises(ValueError):evaluate([row],reference)
    reference.update(reviewer='test reviewer',annotation_source='synthetic fixture',independent_of_predictions=True)
    case=reference['cases'][0];case.update(status='confirmed',projection='frontal',landmarks={
        'Th12':{'visible':True,'point':[30,10]},'iliac_crest_left':{'visible':True,'point':[10,20]},'iliac_crest_right':{'visible':False,'point':None}})
    result=evaluate([row],reference)
    assert result['projection']['accuracy_including_abstentions']==0
    assert result['landmarks']['Th12']['detected']==0
    assert result['landmarks']['iliac_crest_left']['median_mm']==4
    assert result['landmarks']['iliac_crest_right']['false_positive_on_invisible']==1


def test_api_exposes_source_roi_and_anatomy_without_claiming_full_pass(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from dxaqc import api
    monkeypatch.setattr(api,'DATA_DIR',tmp_path/'jobs')
    path=write_dicom(tmp_path/'image.dcm',synthetic_spine(),study_uid='1.2',sop_uid='1.2.3')
    ds=pydicom.dcmread(path);mask=np.zeros((300,300),np.uint8);mask[50:200,60:220]=1
    add_overlay(ds,mask);ds.save_as(path,enforce_file_format=True)
    with TestClient(api.app) as client:
        response=client.post('/api/v1/analyze',files={'file':('image.dcm',path.read_bytes())})
        assert response.status_code==200,response.text
        assessment=response.json()['assessment']
        assert assessment['source_roi']['status']=='extracted'
        assert assessment['complete'] is False
        assert assessment['projection']['clinical_validation'] is False


def test_trochanter_candidate_follows_distal_femur_component_not_isolated_bright_blob():
    from dxaqc.anatomy import detect_landmarks
    pixels=np.zeros((150,120),np.uint8)
    pixels[60:,35:65]=180
    pixels[35:85,25:50]=180
    pixels[5:15,5:15]=255
    result=detect_landmarks(pixels,'hip_right',{'shaft_axis':[(50,120),(50,149)]})
    cap=next(x for x in result['landmarks'] if x['name']=='greater_trochanter')
    assert cap['status']=='candidate'
    assert 30 <= cap['points'][0][1] <= 40
    assert cap['verified'] is False


def test_raster_roi_membership_preserves_every_pixel_including_hole_boundary_and_singletons():
    masks=[]
    hole=np.ones((10,10),np.uint8);hole[4:6,4:6]=0;masks.append(hole)
    sparse=np.zeros((10,10),np.uint8);sparse[1,1]=1;sparse[5,2:8]=1;masks.append(sparse)
    for mask in masks:
        ds=Dataset();add_overlay(ds,mask,origin=(11,21))
        roi=extract_roi(ds,(40,40),'')['rois'][0]
        for y in range(10):
            for x in range(10):
                assert contains(roi,(x+20,y+10))==bool(mask[y,x])
        assert not contains(roi,(19,10))
        assert not contains(roi,(30,10))


def test_boundary_connected_bone_cannot_be_named_trochanter():
    from dxaqc.anatomy import detect_landmarks,display_overlay
    pixels=np.zeros((150,120),np.uint8);pixels[:,35:65]=180
    overlay={'shaft_axis':[(50,120),(50,149)],'greater_trochanter_top':[(40,0)]}
    candidates=detect_landmarks(pixels,'hip_right',overlay)
    cap=next(x for x in candidates['landmarks'] if x['name']=='greater_trochanter')
    assert cap['points']==[] and cap['status']=='not_localized'
    result={'region':'hip_right','overlay':overlay,'anatomy_candidates':candidates}
    assert display_overlay(result,120)['greater_trochanter_top']==[]
    assert overlay['greater_trochanter_top']==[(40,0)]


def test_medial_pelvis_at_top_does_not_hide_lateral_trochanter_candidate():
    from dxaqc.anatomy import detect_landmarks
    pixels=np.zeros((150,120),np.uint8)
    pixels[60:,35:65]=180
    pixels[35:85,25:50]=180
    pixels[:65,80:110]=180
    pixels[60:65,45:90]=180  # joined head/pelvis component reaches the frame edge
    result=detect_landmarks(pixels,'hip_right',{'shaft_axis':[(50,120),(50,149)]})
    cap=next(x for x in result['landmarks'] if x['name']=='greater_trochanter')
    assert cap['status']=='candidate' and 30 <= cap['points'][0][1] <= 40
    assert cap['verified'] is False


def test_trochanter_fallback_at_boundary_also_abstains_without_shaft_axis():
    from dxaqc.anatomy import detect_landmarks
    result=detect_landmarks(np.zeros((100,100),np.uint8),'hip_right',{'greater_trochanter_top':[(20,0)]})
    cap=next(x for x in result['landmarks'] if x['name']=='greater_trochanter')
    assert cap['status']=='not_localized' and cap['points']==[]
