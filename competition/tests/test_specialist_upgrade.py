import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dxaqc.specialist_qc import OUTPUTS,SpecialistQC,digest
from train_detector import annotation


def test_pose_contract_validates_visibility_and_shape():
    assert annotation('0 .5 .5 .8 .8 .2 .3 2 0 0 0','pose',1,2)==[0]
    for row in ('0 .5 .5 .8 .8 .2 .3 3 0 0 0', '0 .5 .5 .8 .8 0 0 0 0 0 0',
                '0 .5 .5 .8 .8 1.2 .3 2 0 0 0'):
        with pytest.raises(ValueError):annotation(row,'pose',1,2)


@pytest.mark.parametrize('views',['full','full-center'])
def test_last_stage_training_freezes_prefix_and_exports_matching_scores(tmp_path,monkeypatch,views):
    pytest.importorskip('safetensors')
    import train_specialist as training
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem=torch.nn.Conv2d(3,4,1)
            self.stages=torch.nn.Sequential(*[torch.nn.Conv2d(4,4,1) for _ in range(4)])
            self.norm_pre=torch.nn.Identity()
            self.head=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
        def forward(self,x):return self.head(self.stages(self.stem(x)))
    encoder=TinyEncoder().eval()
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    n=80
    labels=pd.DataFrame({'study_key':np.repeat(np.arange(40),2),'quality_class':np.tile([0,1],40),
                         'region':['spine']*n,'first_source_path':[f'{i}.dcm' for i in range(n)],
                         'pixel_sha256':[str(i) for i in range(n)],'rows':[32]*n,'columns':[32]*n,
                         **{k:np.tile([0,1],40) for k in OUTPUTS[1:]}})
    label_path=tmp_path/'labels.csv';labels.to_csv(label_path,index=False)
    weights=tmp_path/'weights.safetensors'
    catalog=json.loads((Path(training.__file__).resolve().parents[1]/'docs/competition/specialist_sources.json').read_text())
    expected=next(a['sha256'] for a in catalog['artifacts'] if a['id']=='convnextv2_tiny_weights')
    real_digest=training.digest
    monkeypatch.setattr(training,'digest',lambda p:expected if p==weights else real_digest(p))
    monkeypatch.setattr(training,'create_encoder',lambda *a:encoder)
    def read(path):
        i=int(path.stem)
        return SimpleNamespace(pixels=np.full((32,32),20+200*(i%2),np.uint8),pixel_sha256=str(i))
    monkeypatch.setattr(training,'read_any',read)
    args=SimpleNamespace(output=tmp_path/'run',epochs=2,size=32,threads=1,labels=label_path,
                          weights=weights,dataset=tmp_path,backbone='convnextv2_tiny',loss='bce',
                          region='spine',train_mode='last-stage',views=views)
    report=training.run(args)
    assert torch.equal(before['stem.weight'],encoder.stem.weight)
    assert not torch.equal(before['stages.3.weight'],encoder.stages[3].weight)
    assert report['encoder_sha256']==digest(args.output/'encoder.safetensors')
    assert 1<=report['selected_epoch']<=2
    candidate=SpecialistQC(args.output)
    rows=json.loads((args.output/'predictions.json').read_text())
    p=candidate.predict(read(tmp_path/'0.dcm').pixels,'spine')['predictions']['quality']['score']
    assert p==pytest.approx(rows[0]['quality_score'],abs=1e-5)
    with pytest.raises(ValueError,match='different anatomical'):candidate.predict(read(tmp_path/'0.dcm').pixels,'hip_left')


def test_regional_suite_rejects_metadata_tampering(tmp_path):
    from test_specialist_qc import write_candidate
    from pack_specialists import pack
    from dxaqc.specialist_qc import load_specialist
    for region in ('spine','hip'):
        p=tmp_path/region;p.mkdir();write_candidate(p)
        meta=json.loads((p/'model.json').read_text());meta.update(region_scope=region,labels_sha256='same')
        (p/'model.json').write_text(json.dumps(meta))
    output=tmp_path/'suite';pack(tmp_path/'spine',tmp_path/'hip',output)
    suite=load_specialist(output)
    for region,scope in [('spine','spine'),('hip_left','hip'),('hip_right','hip')]:
        assert suite.predict(np.zeros((64,64),np.uint8),region)['region_scope']==scope
    (output/'hip/model.json').write_text('{}')
    with pytest.raises(ValueError,match='metadata checksum'):load_specialist(output)


def test_inventory_does_not_equate_sources_with_weights_or_validation(tmp_path):
    from dxaqc.specialist_catalog import inventory
    f=tmp_path/'source';f.write_bytes(b'abc')
    catalog={'artifacts':[{'id':'source','path':'source','bytes':3,'sha256':digest(f)}],
             'capabilities':[{'id':'blocked','protocol':'whole-body','task':'qc','artifacts':['source'],
                             'readiness':'weights_unavailable','limitation':'no checkpoint'}]}
    p=tmp_path/'catalog.json';p.write_text(json.dumps(catalog))
    result=inventory(tmp_path,p)
    assert result['capabilities'][0]['artifacts_verified']
    assert result['capabilities'][0]['readiness']=='weights_unavailable'
    assert not result['clinical_models_ready']
    f.write_bytes(b'bad')
    assert not inventory(tmp_path,p)['capabilities'][0]['artifacts_verified']


def test_fusion_selection_uses_validation_values_only():
    from benchmark_specialists import choose_fusion
    y=np.array([0,1,0,1]);baseline=np.array([.1,.8,.2,.9]);regional=1-baseline
    weight,threshold=choose_fusion(y,baseline,regional)
    assert weight==0 and np.isfinite(threshold)


def test_reviewed_coco_import_is_atomic_and_preserves_study_split(tmp_path):
    import cv2,csv
    from import_annotations import convert
    packet=tmp_path/'packet';packet.mkdir();(packet/'images').mkdir()
    samples=[];images=[];annotations=[]
    for i,split in enumerate(('train','val','test')):
        name=f'images/{i}.png';cv2.imwrite(str(packet/name),np.random.default_rng(i).integers(0,256,(64,64),dtype=np.uint8))
        samples.append({'image_id':str(i),'image':name,'study':str(i),'split':split,'png_sha256':digest(packet/name)})
        images.append({'id':i,'file_name':name,'width':64,'height':64})
        annotations.append({'id':i,'image_id':i,'category_id':1,'bbox':[10,10,20,20]})
    (packet/'review-manifest.json').write_text(json.dumps({'samples':samples}))
    coco=tmp_path/'coco.json';coco.write_text(json.dumps({'images':images,'annotations':annotations,'categories':[{'id':1,'name':'implant'}]}))
    ledger=tmp_path/'ledger.csv'
    def write_review(reviewed):
        with ledger.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['image_id','reviewer_id','reviewed']);w.writeheader()
            for i in range(3):w.writerow({'image_id':str(i),'reviewer_id':'expert-A','reviewed':reviewed})
    write_review('false')
    with pytest.raises(ValueError,match='reviewed=true'):convert(packet,coco,ledger,'detect',tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
    write_review('true');audit=convert(packet,coco,ledger,'detect',tmp_path/'accepted')
    assert audit['study_counts']=={'train':1,'val':1,'test':1}
    (packet/'images/0.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Image file changed'):convert(packet,coco,ledger,'detect',tmp_path/'corrupt')
    assert not (tmp_path/'corrupt').exists()


def test_independent_reviews_keep_disagreement_unresolved(tmp_path):
    import csv
    from reconcile_reviews import reconcile
    samples=[{'image_id':'a','region':'spine'}]
    (tmp_path/'review-manifest.json').write_text(json.dumps({'samples':samples,'labels_sha256':'source'}))
    for reviewer in ('A','B'):
        with (tmp_path/f'{reviewer}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['image_id','reviewer_id','reviewed',*OUTPUTS]);w.writeheader()
            w.writerow({'image_id':'a','reviewer_id':reviewer,'reviewed':'true',**{n:'0' for n in OUTPUTS},'spine_axis':'1' if reviewer=='B' else '0'})
    result=reconcile(tmp_path,tmp_path/'A.csv',tmp_path/'B.csv',tmp_path/'result.json')
    assert result['counts']['requires_adjudication']==1
    assert 'spine_axis' in result['images'][0]['disagreements']
    assert not result['original_labels_changed']
