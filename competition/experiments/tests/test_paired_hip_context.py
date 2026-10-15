from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from experiments.paired_hip_context import partners_from_routes,fit_pair,predict_pair


def test_partner_mapping_uses_study_and_opposite_route_only():
    studies=np.array(['a','a','a','b','c','c'])
    routes=np.array(['hip_left','spine','hip_right','hip_left','hip_left','hip_right'])
    np.testing.assert_array_equal(partners_from_routes(studies,routes),[2,-1,0,-1,5,4])
    with pytest.raises(ValueError):partners_from_routes(np.array(['a','a']),np.array(['hip_left','hip_left']))


def test_context_training_excludes_external_targets_and_missing_partner_falls_back():
    rng=np.random.default_rng(17);n=30
    studies=np.repeat(np.arange(15).astype(str),2);routes=np.tile(['hip_left','hip_right'],15)
    partners=partners_from_routes(studies,routes)
    geometry=rng.normal(size=(n,4));embedding=rng.normal(size=(n,20));train=np.arange(20);query=np.arange(20,30)
    targets={'quality':np.tile([0.,1.,1.,0.,0.],6)}
    one=fit_pair(geometry,embedding,partners,targets,train)
    a=predict_pair(one,geometry,embedding,partners,query)
    targets['quality'][query]=np.nan
    b=predict_pair(fit_pair(geometry,embedding,partners,targets,train),geometry,embedding,partners,query)
    np.testing.assert_allclose(a[True]['quality'],b[True]['quality'])
    singleton=predict_pair(one,geometry,embedding,partners,query[:1])
    np.testing.assert_allclose(singleton[True]['quality'],a[True]['quality'][:1])
    missing=partners.copy();missing[query]=-1
    fallback=predict_pair(one,geometry,embedding,missing,query)
    np.testing.assert_array_equal(fallback[True]['quality'],fallback[False]['quality'])
