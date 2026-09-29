"""DXA criterion audit refuses duplicated study identities before fitting."""
import numpy as np
import pandas as pd
import pytest

from experiments.dxa_spine_coverage_ablation import counts, run


def test_binary_report_keeps_all_six_positives_in_denominator():
    truth=np.array([1]*6+[0]*93)
    guess=np.array([1,0,0,0,0,0]+[1]*3+[0]*90)
    result=counts(truth,guess)
    assert result['tn_fp_fn_tp']==[90,3,5,1]
    assert result['sensitivity']==pytest.approx(1/6)


def test_study_duplicate_rejected_before_training():
    labels=pd.DataFrame({'region':['spine']*99,'study_key':[str(i) for i in range(98)]+['0'],
                         'spine_coverage':[1]*6+[0]*93})
    with pytest.raises(ValueError,match='cohort'):
        run(labels,[],np.empty((99,1792)))
