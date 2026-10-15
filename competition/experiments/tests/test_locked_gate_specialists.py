from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.locked_gate_specialists import choose_expert


def test_margin_and_ap_guard_keep_default_on_small_or_ranking_regression():
    truth = np.array([0., 0., 1., 1.])
    incumbent = np.array([.1, .8, .7, .9])
    perfect = np.array([.1, .2, .8, .9])
    gate = np.ones(4, bool)
    chosen = choose_expert(truth, incumbent, .5, {'better': perfect}, gate, .05)
    assert chosen['source'] == 'better'
    assert chosen['inner_f1'] == 1.
    # A higher fixed margin must not silently choose the replacement.
    assert choose_expert(truth, incumbent, .5, {'better': perfect}, gate, .3)['source'] == 'incumbent'
    # Identical information retains the incumbent even with zero margin.
    assert choose_expert(truth, incumbent, .5, {'same': incumbent}, gate, 0.)['source'] == 'incumbent'
    # A threshold gain must not override worse ranking information.
    assert choose_expert(truth, perfect, .95, {'worse_ap': incumbent}, gate, 0.)['source'] == 'incumbent'


def test_quality_gate_prevents_credit_for_inaccessible_true_positives():
    truth = np.array([0., 0., 1., 1., np.nan])
    score = np.array([.1, .2, .8, .9, .5])
    chosen = choose_expert(truth, score, .5, {'same': score},
                           np.array([1, 1, 0, 0, 1], bool), .05)
    assert chosen['source'] == 'incumbent'
    assert chosen['inner_f1'] == 0.
