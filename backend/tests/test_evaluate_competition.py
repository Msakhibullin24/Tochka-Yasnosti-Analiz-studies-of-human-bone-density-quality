import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
spec = importlib.util.spec_from_file_location("evaluate_competition", Path(__file__).parents[1] / "scripts/evaluate_competition.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def sample():
    truth = [dict(path_to_study=str(i), study_uid=str(i), image_uid=str(i),
                  anatomical_region="spine", quality_class=str(i % 2), group_id=str(i)) for i in range(4)]
    predictions = [{**r, "processing_status": "Success", "time_of_processing": "0.2",
                    "quality_probability": "0.9" if r["quality_class"] == "1" else "0.1", "violation_type": ""} for r in truth]
    return truth, predictions


def test_failures_missing_unknown_and_wrong_region_remain_visible():
    truth, pred = sample()
    truth[2]["quality_class"] = ""
    pred[0]["anatomical_region"] = "hip"
    pred[1]["processing_status"] = "Failure"
    result = evaluator.evaluate(truth, pred[:3], repeats=0)
    assert result["overall"]["statuses"] == {"Success": 2, "Failure": 1, "Missing": 1}
    assert result["overall"]["labeled_coverage"] == 1 / 3
    assert result["overall"]["unscored_positive"] == 2
    assert result["overall"]["metrics_on_successful_labeled_images"]["roc_auc"] is None
    assert result["overall"]["metrics_on_successful_labeled_images"]["f1"] is None
    assert result["region_accuracy_on_success"] == 0.5


def test_known_metrics_and_reproducible_group_intervals():
    truth, pred = sample()
    result = evaluator.evaluate(truth, pred, repeats=30)
    assert result == evaluator.evaluate(truth, pred, repeats=30)
    metrics = result["overall"]["metrics_on_successful_labeled_images"]
    assert metrics["f1"] == metrics["roc_auc"] == 1
    for p in pred:
        p.update(quality_class="1", quality_probability="0.5")
    metrics = evaluator.evaluate(truth, pred, repeats=0)["overall"]["metrics_on_successful_labeled_images"]
    assert metrics["f1"] == 2 / 3
    assert metrics["roc_auc"] == metrics["balanced_accuracy"] == 0.5


def test_duplicate_nonfinite_and_inconsistent_group_rejected():
    truth, pred = sample()
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.evaluate(truth, pred + pred[:1], repeats=0)
    pred[0]["quality_probability"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate(truth, pred, repeats=0)
    truth[1]["study_uid"] = truth[0]["study_uid"]
    with pytest.raises(ValueError, match="multiple"):
        evaluator.evaluate(truth, [], repeats=0)
