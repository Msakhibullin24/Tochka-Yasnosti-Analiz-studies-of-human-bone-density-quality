"""Calibration covers parent groups, and ambiguous half-body estimates abstain."""
import numpy as np
import pytest
from experiments import train_th12_visibility as visibility
from dxaqc.embedding import WEIGHTS_SHA256


def test_calibration_uses_worst_residual_per_parent():
    radius=visibility.calibration_radius(np.array([.1,.9,.2,.3,.4,.5]),['a','a','b','b','c','c'],alpha=.5)
    assert radius==.5
    with pytest.raises(ValueError,match='Too few'):
        visibility.calibration_radius(np.array([.1,.2]),['a','b'])
    with pytest.raises(ValueError,match='residual'):
        visibility.calibration_radius(np.array([float('nan')]),['a'])


class Fixed:
    def __init__(self,value):self.value=value
    def predict(self,x):return np.array([self.value])


def bundle(value,radius=.2):
    return {'schema_version':2,'scope':'th12_partial_visibility','encoder_sha256':WEIGHTS_SHA256,
            'feature_contract':visibility.FEATURE_CONTRACT,'clinical_validation':False,
            'model':Fixed(value),'calibration_radius':radius}


def test_half_body_interval_crossing_threshold_abstains(monkeypatch):
    monkeypatch.setattr(visibility,'features',lambda pixels:np.zeros(1792))
    image=np.zeros((40,40),np.uint8)
    result=visibility.assess(bundle(.55),image)
    assert result['candidate_rule']=='undetermined' and result['clinical_verdict']=='undetermined'
    assert visibility.assess(bundle(.1),image)['candidate_rule']=='candidate_less_than_half_height'
    assert visibility.assess(bundle(.9),image)['candidate_rule']=='candidate_at_least_half_height'
    with pytest.raises(ValueError,match='Incompatible'):
        visibility.assess({**bundle(.9),'feature_contract':'different'},image)
    for invalid in (float('inf'),float('-inf'),float('nan')):
        with pytest.raises(ValueError,match='Invalid visibility prediction'):
            visibility.assess(bundle(invalid),image)
