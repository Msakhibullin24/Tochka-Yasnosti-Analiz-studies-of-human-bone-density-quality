"""Reusing image features preserves training initialization and source/view identity."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

import train_anatomy_masks as training
from dxaqc.embedding import WEIGHTS_SHA256


def cache(tmp_path):
    row={'source':'aasce','group':'one','view':'lumbar_half_th12','image_sha256':'file',
         'pixel_sha256':'pixels','crop_xyxy':[0,3,20,30]}
    (tmp_path/'inventory.json').write_text(json.dumps({'rows':[row]}))
    np.save(tmp_path/'features.npy',np.zeros((1,896,40,40),np.float16))
    torch.save({'encoder_sha256':WEIGHTS_SHA256,'input_size':320,'grid':40},tmp_path/'anatomy_masks.pt')
    return row


def test_reuse_rejects_changed_pixels_or_crop(tmp_path):
    row=cache(tmp_path)
    _,ids=training.reuse_features(tmp_path,[row])
    assert ids==[0]
    for changes in ({'pixel_sha256':'changed'},{'crop_xyxy':[0,4,20,30]}):
        with pytest.raises(ValueError,match='identity differs'):
            training.reuse_features(tmp_path,[{**row,**changes}])


def test_reuse_rejects_different_encoder(tmp_path):
    row=cache(tmp_path)
    torch.save({'encoder_sha256':'different','input_size':320,'grid':40},tmp_path/'anatomy_masks.pt')
    with pytest.raises(ValueError,match='encoder'):
        training.reuse_features(tmp_path,[row])


def test_fresh_and_reused_features_train_identical_heads(tmp_path,monkeypatch):
    rows=[{'source':'aasce','group':str(i),'image_sha256':str(i),'pixel_sha256':str(i)} for i in range(6)]
    monkeypatch.setattr(training,'records',lambda *args:([dict(r) for r in rows],[]))
    monkeypatch.setattr(training,'split_records',lambda rows:np.array([False]*4+[True]*2))
    def load(row):
        pixels=np.full((40,40),int(row['group']),np.uint8)
        target=np.zeros((40,40),np.int16);target[10:20,10:20]=13
        return pixels,target
    monkeypatch.setattr(training,'load_record',load)
    def encode(pixels):
        torch.manual_seed(0)  # Real encoder initialization has this side effect.
        return np.full((896,40,40),int(pixels[0,0])/10,np.float16)
    monkeypatch.setattr(training,'spatial_features',encode)
    fresh=tmp_path/'fresh';reused=tmp_path/'reused'
    for output,previous in ((fresh,None),(reused,fresh)):
        training.run(SimpleNamespace(aasce=None,dxa=None,output=output,epochs=1,
                                     lumbar_crops=False,lumbar_only=False,reuse_features=previous))
    a=torch.load(fresh/'anatomy_masks.pt',weights_only=True)['state_dict']
    b=torch.load(reused/'anatomy_masks.pt',weights_only=True)['state_dict']
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
