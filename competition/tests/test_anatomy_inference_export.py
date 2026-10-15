import hashlib
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.export_anatomy_candidate import export_bank
from experiments.live_anatomy_candidate import load_bank


def test_export_removes_training_data_and_preserves_fitted_predictions(tmp_path):
    x = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
    target = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(x, target)
    state = {'fold': 0, 'group': 'hip', 'heads': {'quality': model}, 'columns': {'quality': ['x', 'y']},
             'codes': ['hip_roi_coverage'], 'quality_truth': target, 'criterion_truth': target,
             'inner': {'quality': x[:, 0]}, 'outer': {'quality': x[:, 1]}, 'train': [0, 1], 'test': [2, 3]}
    source = tmp_path / 'source'
    source.mkdir()
    joblib.dump({'baseline': {'quality_true': target}, 'features': x, 'folds': [state]}, source / 'bank.joblib')
    sha = hashlib.sha256((source / 'bank.joblib').read_bytes()).hexdigest()
    (source / 'manifest.json').write_text(json.dumps({'bank_sha256': sha, 'embeddings': []}))
    exported = tmp_path / 'exported'
    manifest = export_bank(source, exported)
    payload = joblib.load(exported / 'bank.joblib')
    assert set(payload) == {'schema_version', 'inference_only', 'folds'}
    _, loaded, studies = load_bank(exported)
    assert not studies
    assert set(loaded[0, 'hip']) == {'fold', 'group', 'heads', 'columns', 'codes'}
    query = np.array([[.5, .4], [4., 3.]])
    assert np.array_equal(model.predict_proba(query), loaded[0, 'hip']['heads']['quality'].predict_proba(query))
    assert manifest['training_bank_sha256'] == sha
