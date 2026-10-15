from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from experiments.nested_model_council import (fit_experts,predict_experts,fit_arbiter,
    predict_arbiter,quality_cut,POLICIES)


def test_external_targets_cannot_change_experts_or_predictions():
    rng=np.random.default_rng(17)
    matrices=[rng.normal(size=(24,4))]
    regions=np.tile([0,1],12)
    target=np.tile([0,0,1,1],6).astype(float)
    train,query=np.arange(16),np.arange(16,24)
    one=predict_experts(fit_experts(matrices,regions,target,train),matrices,regions,query)
    target[query]=np.nan
    two=predict_experts(fit_experts(matrices,regions,target,train),matrices,regions,query)
    np.testing.assert_array_equal(one,two)


def test_arbiters_are_pointwise_and_return_bounded_scores():
    rng=np.random.default_rng(17);raw=rng.uniform(.01,.99,size=(40,7))
    regions=np.tile([0,1],20);target=(raw[:,0]>.5).astype(int)
    for policy in POLICIES:
        model=fit_arbiter(policy,raw,regions,target)
        full=predict_arbiter(model,raw,regions)
        assert np.all((full>=0)&(full<=1))
        singleton=predict_arbiter(model,raw[:1],regions[:1])
        np.testing.assert_allclose(singleton,full[:1],atol=1e-12,rtol=0)
        if policy=='simplex':
            assert np.min(model['weights'])>=0
            np.testing.assert_allclose(model['weights'].sum(),1,atol=1e-8)


def test_cut_respects_forced_physical_axis_and_all_negative_option():
    target=np.array([0,0,1,1]);q=np.array([.1,.2,.3,.4]);forced=np.array([False,False,True,False])
    cut=quality_cut(target,q,forced)
    np.testing.assert_array_equal((q>=cut)|forced,target.astype(bool))
    cut=quality_cut(np.zeros(4,int),q,np.zeros(4,bool))
    assert cut>q.max()
