"""Numbering audit distinguishes spatially correct bodies from swapped names."""
import numpy as np
from experiments.audit_spine_numbering import numbering_metrics,summarize


def reference():
    labels=np.zeros((12,12),np.int16)
    labels[1:4,4:8]=13;labels[7:10,4:8]=14
    return labels


def test_swapped_levels_are_localized_but_never_count_as_correct_numbering():
    truth=reference();prediction=truth.copy()
    prediction[truth==13]=14;prediction[truth==14]=13
    result=numbering_metrics(prediction,truth,truth.shape)
    assert result['localized_iou50']==2 and result['correct_names_iou50']==0
    assert result['false_predictions_iou50']==0
    assert [(c['reference_name'],c['matched_prediction_name']) for c in result['cases']]==[('L1','L2'),('L2','L1')]


def test_missing_bodies_remain_in_named_accuracy_and_boundary_coverage():
    truth=reference();result=numbering_metrics(np.zeros_like(truth),truth,truth.shape)
    summary=summarize([{'candidate':result}],'candidate')
    assert summary['reference_bodies']==2 and summary['correct_named_fraction_including_misses']==0
    assert summary['boundary_measured_bodies']==0 and summary['boundary_unavailable_bodies']==2
    assert summary['boundary_hd95_mean_source_pixels_on_present_names'] is None


def test_unknown_area_is_not_scored_as_false_detection():
    truth=reference();truth[:4]=-100
    prediction=np.zeros_like(truth);prediction[:3,:3]=12;prediction[7:10,4:8]=14
    result=numbering_metrics(prediction,truth,truth.shape)
    assert result['references']==1 and result['predictions_in_known_area']==1
    assert result['correct_names_iou50']==1 and result['false_predictions_iou50']==0


def test_boundary_distance_uses_both_original_resize_dimensions():
    truth=np.zeros((8,8),np.int16);truth[1:3,1:3]=13
    shifted=np.zeros_like(truth);shifted[1:3,2:4]=13
    result=numbering_metrics(shifted,truth,(40,80))
    assert result['cases'][0]['same_name_boundary_hd95_source_pixels']==10


def test_original_reference_lost_on_resize_remains_in_denominator():
    truth=reference();truth[truth==14]=0
    result=numbering_metrics(truth,truth,truth.shape,source_levels=[13,14])
    summary=summarize([{'candidate':result}],'candidate')
    assert summary['reference_bodies']==2 and summary['correct_names_iou50']==1
    assert summary['correct_named_fraction_including_misses']==.5
    assert summary['references_lost_by_grid_quantization']==1
    assert summary['boundary_unavailable_bodies']==1
