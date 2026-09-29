"""Decision and grouped calibration contracts for the research half-height rule."""
import numpy as np
import pytest

from experiments.train_th12_half_classifier import calibration_threshold, candidate_rule, assess
from dxaqc.embedding import WEIGHTS_SHA256
from experiments.train_th12_visibility import FEATURE_CONTRACT


def test_group_maximum_calibration_and_exact_half_boundary():
    scores=np.array([.1,.9,.8,.7,.8,.9])
    labels=np.array([0,0,1,1,1,1])
    assert calibration_threshold(scores,labels,['a','a','b','b','c','c'],alpha=.25)==.9
    assert candidate_rule(.1,.4)=='candidate_less_than_half_height'
    assert candidate_rule(.9,.4)=='candidate_at_least_half_height'
    assert candidate_rule(.5,.5)=='undetermined'
    assert candidate_rule(.5,.4)=='undetermined'  # empty conformal set abstains


def test_invalid_calibration_and_probabilities_are_rejected():
    with pytest.raises(ValueError,match='Too few'):
        calibration_threshold([.2,.3],[0,1],['a','b'])
    for scores in ([float('nan')],[float('inf')],[-.1],[1.1]):
        with pytest.raises(ValueError,match='Invalid'):
            calibration_threshold(scores,[1],['a'],alpha=.5)
    for value in (float('nan'),float('inf'),-.1,1.1):
        with pytest.raises(ValueError,match='Invalid'):
            candidate_rule(value,.5)


def test_transfer_assessment_preserves_research_status(monkeypatch):
    from experiments import train_th12_half_classifier as module
    monkeypatch.setattr(module,'features',lambda pixels:np.zeros(1792))
    class Fixed:
        def __init__(self,p):self.p=p
        def predict_proba(self,x):return np.array([[1-self.p,self.p]])
    bundle={'schema_version':2,'scope':'th12_source_half_height','status':'research_only',
            'feature_contract':FEATURE_CONTRACT,'encoder_sha256':WEIGHTS_SHA256,
            'clinical_validation':False,'calibration_threshold':.4,'model':Fixed(.9)}
    result=assess(bundle,np.zeros((20,20),np.uint8))
    assert result['candidate_rule']=='candidate_at_least_half_height'
    assert result['clinical_verdict']=='undetermined' and result['clinical_validation'] is False
    with pytest.raises(ValueError,match='Incompatible'):
        assess({**bundle,'scope':'th12_partial_visibility'},np.zeros((20,20),np.uint8))
    with pytest.raises(ValueError,match='Invalid'):
        assess({**bundle,'model':Fixed(float('inf'))},np.zeros((20,20),np.uint8))
