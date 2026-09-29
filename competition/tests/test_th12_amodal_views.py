"""Derived visibility labels preserve parent splits and never use model predictions."""
import json
import numpy as np
import pytest
from experiments import prepare_th12_amodal_views as prepare


def test_visibility_views_preserve_parent_split_and_unclipped_geometry(tmp_path,monkeypatch):
    points=[];target=np.zeros((220,100),np.uint8)
    for level in range(1,18):
        y=5+(level-1)*12
        points.extend([[25,y],[65,y],[25,y+7],[65,y+7]])
        target[y:y+8,25:66]=level
    rows=[{'source':'aasce','group':str(i),'split':split,'annotation':'fixture.mat',
           'image_sha256':f'parent-{i}','annotation_sha256':'reference'} for i,split in enumerate(('train','test'))]
    source=tmp_path/'inventory.json';source.write_text(json.dumps({'rows':rows}))
    monkeypatch.setattr(prepare,'load_record',lambda row:(np.full((220,100),int(row['group'])*30,np.uint8),target.copy()))
    monkeypatch.setattr(prepare,'loadmat',lambda path:{'p2':np.asarray(points,float)})
    output=tmp_path/'views';prepare.run(source,output)
    report=json.loads((output/'dataset.json').read_text())
    assert report['derived_images']==10 and report['parent_images']==2
    train={c['parent_group'] for c in report['cases'] if c['split']=='train'}
    test={c['parent_group'] for c in report['cases'] if c['split']=='test'}
    assert not train&test
    for parent in ('0','1'):
        cases=[c for c in report['cases'] if c['parent_group']==parent]
        assert cases[0]['reference']['visible_projected_area_fraction']==0
        assert np.max(np.array(cases[0]['full_body_quad'])[:,1])<-.5
        assert cases[-1]['reference']['visible_projected_area_fraction']==1
    assert not list(tmp_path.glob('.th12-amodal-*'))


def test_duplicate_parent_inventory_is_rejected_before_publication(tmp_path):
    row={'source':'aasce','group':'same','split':'train'}
    source=tmp_path/'inventory.json';source.write_text(json.dumps({'rows':[row,row]}))
    with pytest.raises(ValueError,match='one AASCE view'):prepare.run(source,tmp_path/'views')
    assert not (tmp_path/'views').exists()
