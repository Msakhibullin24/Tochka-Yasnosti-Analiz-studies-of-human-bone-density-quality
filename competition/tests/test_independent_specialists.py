import json

import numpy as np
import pytest
import torch

from dxaqc.specialist_qc import (IndependentSpecialists, OUTPUTS, PREPROCESS_VERSION,
                                 digest)
from pack_independent_specialists import pack


def _candidate(path, scope, bias):
    path.mkdir()
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(3, len(OUTPUTS)))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.fill_(bias)
    torch.jit.trace(model.eval(), torch.zeros(1, 3, 32, 32)).save(str(path / 'model.ts'))
    meta = {'schema_version': 1, 'outputs': list(OUTPUTS),
            'preprocess_version': PREPROCESS_VERSION, 'mode': 'shadow',
            'input_size': 32, 'model_sha256': digest(path / 'model.ts'),
            'region_scope': scope, 'labels_sha256': 'a' * 64,
            'thresholds': {name: .5 for name in OUTPUTS}, 'backbone': f'test-{scope}'}
    (path / 'model.json').write_text(json.dumps(meta))


def test_each_specialist_has_own_model_scope_and_output(tmp_path):
    for name, scope, bias in [('spine', 'spine', -2.), ('hip', 'hip', 2.),
                              ('general', 'all', 0.)]:
        _candidate(tmp_path / name, scope, bias)
    output = tmp_path / 'portfolio'
    pack(tmp_path / 'spine', tmp_path / 'hip', tmp_path / 'general', output)
    portfolio = IndependentSpecialists(output)
    pixels = np.full((40, 20), 100, np.uint8)

    spine = portfolio.predict(pixels, 'spine')
    assert set(spine) == {'spine_convnext', 'hip_convnext', 'general_efficientnet'}
    assert spine['hip_convnext']['status'] == 'not_applicable'
    assert spine['spine_convnext']['status'] == spine['general_efficientnet']['status'] == 'ok'
    assert spine['spine_convnext']['predictions']['quality']['score'] < spine['general_efficientnet']['predictions']['quality']['score']
    assert all(not item['affects_decision'] for item in spine.values())

    hip = portfolio.predict(pixels, 'hip_left')
    assert hip['spine_convnext']['status'] == 'not_applicable'
    assert hip['hip_convnext']['predictions']['quality']['score'] > hip['general_efficientnet']['predictions']['quality']['score']

    portfolio.models['spine_convnext'].predict = lambda *_: (_ for _ in ()).throw(RuntimeError('broken'))
    isolated = portfolio.predict(pixels, 'spine')
    assert isolated['spine_convnext']['status'] == 'unavailable'
    assert isolated['general_efficientnet']['status'] == 'ok'


def test_legacy_and_independent_configuration_cannot_be_mixed(monkeypatch):
    from dxaqc.pipeline import Analyzer

    monkeypatch.setenv('DXAQC_SPECIALIST_PATH', '/legacy')
    monkeypatch.setenv('DXAQC_SPECIALIST_PORTFOLIO', '/independent')
    with pytest.raises(ValueError, match='either one legacy specialist or an independent portfolio'):
        Analyzer(bundle=object())
