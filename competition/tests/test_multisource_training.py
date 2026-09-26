import numpy as np
import pytest

from train_multisource_projection import grouped_folds, source_weights
from train_buu_anatomy import numbered_metrics


def test_source_balancing_equalizes_mass_not_sample_counts():
    sources=np.array(['large']*20+['small']*2)
    weights=source_weights(sources)
    assert weights[sources=='large'].sum()==pytest.approx(weights[sources=='small'].sum())


def test_views_stay_grouped_and_shared_pixels_are_rejected():
    rows=[{'source':s,'group':f'{s}:{p}','pixel_sha256':f'{s}:{p}:{v}','target':v}
          for s in ['a','b'] for p in range(10) for v in range(2)]
    folds=grouped_folds(rows)
    assert all(folds[i]==folds[i+1] for i in range(0,len(rows),2))
    i=next(i for i,f in enumerate(folds) if f!=folds[0])
    rows[i]['pixel_sha256']=rows[0]['pixel_sha256']
    with pytest.raises(ValueError,match='leakage'):
        grouped_folds(rows)


def test_numbered_evaluation_does_not_hide_missed_bodies_or_swap_levels():
    actual=np.array([[[0.,float(i)] for i in range(5)]])
    predicted=actual.copy();predicted[:,0]+=10
    report=numbered_metrics(predicted,actual,np.ones((1,5)))
    assert report['within_half_body_height']==4
    swapped=actual[:,::-1].copy()
    assert numbered_metrics(swapped,actual,np.ones((1,5)))['within_half_body_height']==1
    with pytest.raises(ValueError):
        numbered_metrics(actual,actual,np.zeros((1,5)))
