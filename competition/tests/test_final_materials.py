import json

import pytest

from build_final_materials import release_facts


def test_slides_cannot_use_metrics_from_a_previous_decision_version(tmp_path):
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps({'decision_versions': ['4']}))
    with pytest.raises(ValueError, match='current decision version'):
        release_facts(path, tmp_path / 'oof.csv', tmp_path, tmp_path, 'sha256:' + 'a' * 64)
