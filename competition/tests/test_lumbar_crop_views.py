"""Derived anatomical views keep original target semantics and parent split."""
import numpy as np
import pytest

from anatomy_training_data import lumbar_crop_bounds, derive_lumbar_views
from experiments.audit_numbered_spine_axis import score


def target():
    t=np.zeros((200,100),np.int16)
    for i in range(12,18):t[20+(i-12)*20:30+(i-12)*20,30:60]=i
    return t


def test_lumbar_view_keeps_half_th12_and_every_lumbar_level():
    t=target();x0,y0,x1,y1=lumbar_crop_bounds(t)
    crop=t[y0:y1,x0:x1]
    assert set(range(12,18))<=set(np.unique(crop))
    assert (crop==12).sum()==(t==12).sum()/2
    for level in range(13,18):assert (crop==level).sum()==(t==level).sum()
    assert 0<=x0<x1<=100 and 0<=y0<y1<=200
    t[t==17]=0
    with pytest.raises(ValueError,match='Th12'):lumbar_crop_bounds(t)


def test_derived_view_inherits_parent_identity_and_holdout(monkeypatch):
    import anatomy_training_data as adapter
    monkeypatch.setattr(adapter,'load_record',lambda row:(np.zeros((200,100),np.uint8),target()))
    parents=[{'source':'aasce','group':'one','image_sha256':'a'}, {'source':'aasce','group':'two','image_sha256':'b'}, {'source':'ramathibodi','group':'three','image_sha256':'c'}]
    rows,flags=derive_lumbar_views(parents,np.array([True,False,True]))
    assert len(rows)==5 and flags.tolist()==[True,True,False,False,True]
    assert [r['group'] for r in rows]==['one','one','two','two','three']
    assert 'crop_xyxy' in rows[1] and 'crop_xyxy' not in rows[0]
    assert 'crop_xyxy' not in parents[0]


def test_axis_audit_includes_unavailable_positive_in_sensitivity():
    cases=[{'axis_violation':1,'decision':True},{'axis_violation':1,'decision':None},{'axis_violation':0,'decision':False}]
    result=score(cases,'decision')
    assert result['sensitivity_including_unavailable']==.5
    assert result['positive_unavailable']==1 and result['references']==3
