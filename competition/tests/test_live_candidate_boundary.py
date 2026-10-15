import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from experiments.live_anatomy_candidate import load_bank, anatomical_axis_features
from experiments.build_replay_manifest import bind_identity
import pytest


def test_numbered_axis_changes_only_new_bank_input_and_preserves_reference_geometry():
    reference = {'spine_abs_angle_deg': 8., 'other': 1.}
    current = anatomical_axis_features(reference, -3., True)
    assert current['spine_abs_angle_deg'] == 3.
    assert reference['spine_abs_angle_deg'] == 8.
    assert anatomical_axis_features(reference, None, True)['spine_abs_angle_deg'] is None
    assert anatomical_axis_features(reference, -3., False) == reference


def test_inference_bank_loading_retains_models_and_fold_identities_without_qc_targets(tmp_path):
    state = {'fold': 0, 'group': 'spine', 'heads': {'quality': ['fitted-model']},
             'columns': {'quality': ['x']}, 'codes': ['spine_artifact'],
             'inner': {'quality': np.array([.1, .9])}, 'outer': {'quality': np.array([.2])},
             'quality_truth': np.array([0, 1]), 'criterion_truth': np.array([[0], [1]]),
             'train': np.array([0, 1]), 'test': np.array([2])}
    payload = {'folds': [state], 'features': [{'x': 999}],
               'baseline': pd.DataFrame({'study': ['held-study'], 'fold': [0], 'quality_true': [1]})}
    path = tmp_path / 'bank.joblib'
    joblib.dump(payload, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({'bank_sha256': checksum}))
    _, folds, studies = load_bank(tmp_path)
    assert studies == {'held-study': 0}
    assert set(folds[0, 'spine']) == {'fold', 'group', 'heads', 'columns', 'codes'}
    assert folds[0, 'spine']['heads'] == {'quality': ['fitted-model']}


def test_dicom_identity_binding_rejects_aliases_across_source_groups_or_folds():
    mapping = {}
    bind_identity(mapping, 'current-study-uid', 'source-folder-group', 2)
    bind_identity(mapping, 'current-study-uid', 'source-folder-group', 2)
    with pytest.raises(ValueError, match='crosses'):
        bind_identity(mapping, 'current-study-uid', 'other-source-folder', 2)
    with pytest.raises(ValueError, match='crosses'):
        bind_identity(mapping, 'current-study-uid', 'source-folder-group', 3)
