import numpy as np
import pytest

from multilabel_typifier import MultilabelTypifier


def data():
    features = [{'spine_abs_angle_deg': 2., 'rib_signal': float(i % 2)} for i in range(18)]
    embeddings = np.array([[i % 2, i / 20.] for i in range(18)])
    targets = ['spine_artifact' if i % 2 else 'normal' for i in range(18)]
    studies = [f'study-{i//2}' for i in range(18)]
    return features, embeddings, targets, studies


def test_inner_threshold_folds_keep_whole_studies_and_unlearnable_types_unavailable():
    f, e, y, s = data()
    model = MultilabelTypifier('spine').fit(f, e, y, s)
    for split in model.inner_splits:
        assert not set(split['training_studies']) & set(split['test_studies'])
    assert model.models['spine_axis'] is None
    assert model.thresholds['spine_axis'] > 1
    for row in model.predict(f, e):
        assert 'spine_axis' not in row['codes']
        assert 'spine_coverage' not in row['codes']


def test_sparse_positive_without_inner_positive_evidence_is_not_always_predicted():
    f, e, y, s = data()
    y = ['normal']*18;y[0] = 'spine_artifact'
    model = MultilabelTypifier('spine').fit(f, e, y, s)
    assert model.thresholds['spine_artifact'] > 1


def test_axis_guard_cannot_be_bypassed_by_high_probability():
    f, e, y, s = data()
    y = ['spine_axis' if i % 2 else 'normal' for i in range(18)]
    model = MultilabelTypifier('spine').fit(f, e, y, s)
    model.thresholds['spine_axis'] = 0
    assert 'spine_axis' in model.predict(f[:1], e[:1], axis_guard=False)[0]['codes']
    guarded = model.predict(f[:1], e[:1])[0]
    assert 'spine_axis' not in guarded['codes']
    assert guarded['conflicts']
    f[0]['spine_abs_angle_deg'] = 5.01
    assert 'spine_axis' in model.predict(f[:1], e[:1])[0]['codes']


def test_wrong_region_and_insufficient_groups_are_rejected():
    f, e, y, s = data()
    with pytest.raises(ValueError):
        MultilabelTypifier('hip').fit(f, e, y, s)
    with pytest.raises(ValueError):
        MultilabelTypifier('spine').fit(f, e, y, ['same']*len(s))
