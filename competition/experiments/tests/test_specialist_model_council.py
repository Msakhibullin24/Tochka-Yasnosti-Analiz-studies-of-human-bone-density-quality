from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.specialist_model_council import fit_raw,predict_raw,fit_policy,predict_policy,select_operating_point


def test_specialist_models_do_not_read_external_quality_or_type_targets():
    rng=np.random.default_rng(12);n=24
    embeddings=[rng.normal(size=(n,3)) for _ in range(6)]
    geometries={'artifact':(rng.normal(size=(n,4)),rng.normal(size=(n,4)))}
    targets={'artifact':np.tile([0.,0.,1.,1.],6)}
    train,query=np.arange(16),np.arange(16,24)
    one=predict_raw(fit_raw(embeddings,geometries,targets,train),embeddings,geometries,query)
    targets['artifact'][query]=np.nan
    two=predict_raw(fit_raw(embeddings,geometries,targets,train),embeddings,geometries,query)
    np.testing.assert_array_equal(one['artifact'],two['artifact'])


def test_legacy_controls_reproduce_expert_weights_without_claiming_old_decoder():
    x=np.arange(24).reshape(3,8)/24
    for policy,index in [('legacy_reference',7),('legacy_anatomy',6)]:
        models=fit_policy(policy,{'coverage':x},{},np.arange(3))
        output=predict_policy(models,{'coverage':x})['coverage']
        np.testing.assert_allclose(output,.25*x[:,0]+.25*x[:,1]+.5*x[:,index])


def test_missing_type_labels_are_filtered_before_training_arbiter():
    rng=np.random.default_rng(2);x=rng.uniform(.05,.95,(10,8));target=np.tile([0.,1.],5)
    target[3]=np.nan
    model=fit_policy('stack_shrink',{'coverage':x},{'coverage':target},np.arange(10))
    expected=fit_policy('stack_shrink',{'coverage':x[np.isfinite(target)]},
        {'coverage':target[np.isfinite(target)]},np.arange(9))
    np.testing.assert_allclose(predict_policy(model,{'coverage':x})['coverage'],
                               predict_policy(expected,{'coverage':x})['coverage'])


def test_operating_point_is_reproducible_with_grouped_resampling():
    prediction={'quality':np.array([.1,.8,.2,.9,.3,.7,.4,.6]),
                'coverage':np.array([.1,.8,.2,.9,.3,.7,.4,.6])}
    target=np.tile([0,1],4);groups=np.repeat(np.arange(4),2);forced=np.zeros(8,bool)
    a=select_operating_point(prediction,['coverage'],target,forced,groups,100)
    b=select_operating_point(prediction,['coverage'],target,forced,groups,100)
    assert a==b and a['inner_f1']==1
