import json

import pytest

from build_final_materials import release_facts


def test_slides_cannot_use_metrics_from_a_previous_decision_version(tmp_path):
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps({'decision_versions': ['4']}))
    with pytest.raises(ValueError, match='current decision version'):
        release_facts(path, tmp_path / 'oof.csv', tmp_path, tmp_path, 'sha256:' + 'a' * 64)


def test_profile_cannot_be_presented_with_baseline_only_metrics(tmp_path):
    import hashlib
    from dxaqc import __version__
    predictions = tmp_path/'oof.csv'
    predictions.write_text('decision_version\n5\n')
    metrics = tmp_path/'metrics.json'
    metrics.write_text(json.dumps({'decision_versions': ['5'], 'labelled_images': 1,
                                  'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest()}))
    (tmp_path/'summary.json').write_text(json.dumps({'version': __version__,
                                                   'workflow_profile_id': 'tz-delivery-1.11.0'}))
    with pytest.raises(ValueError, match='baseline metrics are insufficient'):
        release_facts(metrics, predictions, tmp_path, tmp_path, 'sha256:'+'a'*64)
