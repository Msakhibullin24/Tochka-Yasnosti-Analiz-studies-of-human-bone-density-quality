import numpy as np
import pytest

from experiments.joint_quality_cuts import fit_joint_quality_cuts
from experiments.quality_gated_decoder import fit_quality_cut


def group(code, truth, quality):
    truth = np.asarray(truth, float)
    state = {'codes': [code], 'inner': {code: np.full(len(truth), .9)},
             'quality_truth': truth, 'criterion_truth': truth[:, None],
             'angles': np.full(len(truth), np.nan)}
    spec = {'thresholds': {code: .5}}
    return state, spec, np.asarray(quality)


def test_coupled_f1_uses_population_counts_instead_of_mean_group_f1():
    spine = group('spine_coverage', [1, 0, 0, 0], [.2, .3, .4, .5])
    hip = group('hip_roi_coverage', [1] * 6, [.8] * 6)
    assert fit_quality_cut(*spine, type_weight=0) == .2
    chosen = fit_joint_quality_cuts({'spine': spine, 'hip': hip}, type_weight=0)
    assert chosen['spine'] > 1
    assert chosen['hip'] == .8
    # The selector requires neither outer scores nor outer labels.
    assert all('outer' not in state for state, _, _ in (spine, hip))


def test_missing_group_cannot_silently_change_the_training_objective():
    with pytest.raises(ValueError, match='both anatomical groups'):
        fit_joint_quality_cuts({'spine': group('spine_coverage', [1, 0], [.8, .2])})
