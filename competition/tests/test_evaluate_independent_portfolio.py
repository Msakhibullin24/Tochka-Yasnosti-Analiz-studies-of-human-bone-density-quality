import numpy as np

from evaluate_independent_portfolio import disagreements, paired_ci


def test_paired_interval_is_zero_for_identical_predictions():
    truth = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([.1, .9, .2, .8, .3, .7])
    pred = (score >= .5).astype(int)
    studies = np.array(["a", "a", "b", "b", "c", "c"])
    result = paired_ci(truth, score, pred, score, pred, studies, repeats=100)
    for entry in result.values():
        assert entry["low"] == entry["high"] == 0


def test_paired_interval_keeps_study_members_together():
    truth = np.array([0, 1, 0, 1])
    score = np.array([.1, .9, .2, .8])
    good = np.array([0, 1, 0, 1])
    bad = np.array([1, 0, 1, 0])
    studies = np.array(["a", "a", "b", "b"])
    result = paired_ci(truth, score, good, score, bad, studies, repeats=100)
    assert result["delta_f1"]["low"] == result["delta_f1"]["high"] == 1


def test_disagreement_review_exposes_only_hashed_study():
    review = disagreements(np.array([0, 1]), np.array([0, 1]), np.array([0, 0]),
                           np.array([1, 1]), np.array(["spine", "hip"]),
                           np.array(["private-study", "other-study"]))
    assert review["corrected_count"] == 1
    assert review["regressed_count"] == 1
    assert review["candidate_corrected_baseline"][0]["label_row"] == 0
    assert "private-study" not in str(review)
