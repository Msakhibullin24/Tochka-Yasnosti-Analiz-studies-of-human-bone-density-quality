"""Projection training excludes mixed views and keeps transitive duplicates together."""
import numpy as np
import pytest
from experiments.train_fracatlas_hip_projection import projection_label, duplicate_groups


def row(**changes):
    result=dict(frontal='1',lateral='0',oblique='0',mixed='0',multiscan='0')
    result.update(changes)
    return result


def test_only_single_author_view_is_eligible():
    assert projection_label(row())==0
    assert projection_label(row(frontal='0',lateral='1'))==1
    assert projection_label(row(frontal='0',oblique='1'))==1
    assert projection_label(row(mixed='1'))==0  # Multiple body regions, one view.
    for changes in ({'multiscan':'1'},{'lateral':'1'},{'frontal':'0'}):
        assert projection_label(row(**changes)) is None
    with pytest.raises(ValueError,match='tag'):
        projection_label(row(lateral='unknown'))


def test_near_duplicate_components_are_transitive(monkeypatch):
    import experiments.train_fracatlas_hip_projection as module
    values=iter([0,15,255,65535])
    monkeypatch.setattr(module,'dhash',lambda image:next(values))
    images=[np.full((32,32),i,dtype=np.uint8) for i in range(4)]
    groups,edges,_=duplicate_groups(images)
    assert groups[0]==groups[1]==groups[2]
    assert groups[3]!=groups[2]
    assert [0,1] in edges and [1,2] in edges
    assert [0,2] not in edges


def test_research_hip_bundle_cannot_be_loaded_as_runtime_spine_gate(tmp_path):
    import joblib
    from dxaqc.embedding import WEIGHTS_SHA256
    from dxaqc.projection_model import ProjectionModel
    path=tmp_path/'research-hip.joblib'
    joblib.dump({'schema_version':2,'status':'research_only','scope':'hip_frontal_vs_nonfrontal',
                 'encoder_sha256':WEIGHTS_SHA256,'clinical_validation':False,'model':None},path)
    with pytest.raises(ValueError,match='Incompatible'):
        ProjectionModel(path)
