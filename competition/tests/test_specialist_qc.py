import json

import numpy as np
import pandas as pd
import pytest
import torch

from dxaqc.specialist_qc import (OUTPUTS, PREPROCESS_VERSION, SpecialistQC, digest,
                               image_tensor, masked_loss)


@pytest.mark.parametrize('kind', ['bce', 'asl'])
def test_unknown_labels_have_no_gradient(kind):
    logits = torch.tensor([[.2, -.1, .4]], requires_grad=True)
    loss = masked_loss(logits, torch.tensor([[1., float('nan'), 0.]]), kind)
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert logits.grad[0, 0] < 0 and logits.grad[0, 2] > 0
    empty = masked_loss(logits, torch.full_like(logits, float('nan')), kind)
    assert empty == 0 and torch.isfinite(empty)


def test_masked_bce_matches_known_labels_only():
    logits = torch.tensor([[1., 8., -2.]])
    y = torch.tensor([[1., float('nan'), 0.]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, [0, 2]], y[:, [0, 2]])
    assert torch.allclose(masked_loss(logits, y), expected)
    with pytest.raises(ValueError):
        masked_loss(logits, torch.ones_like(logits)*2)
    with pytest.raises(ValueError):
        masked_loss(logits, torch.full_like(logits, float('inf')))


def write_candidate(path):
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Linear(3, len(OUTPUTS)))
    torch.jit.trace(model.eval(), torch.zeros(1, 3, 32, 32)).save(str(path/'model.ts'))
    meta = {'schema_version': 1, 'outputs': list(OUTPUTS), 'preprocess_version': PREPROCESS_VERSION,
            'mode': 'shadow', 'input_size': 32, 'model_sha256': digest(path/'model.ts'),
            'thresholds': {name: .5 for name in OUTPUTS}}
    meta['thresholds']['spine_artifact'] = None
    (path/'model.json').write_text(json.dumps(meta))


def test_offline_artifact_checks_and_protocol_mask(tmp_path, monkeypatch):
    write_candidate(tmp_path)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', lambda *a, **k: pytest.fail('Network'))
    candidate = SpecialistQC(tmp_path)
    result = candidate.predict(np.full((40, 20), 100, np.uint8), 'spine')
    assert not result['affects_decision']
    assert result['predictions']['hip_roi_coverage']['status'] == 'not_applicable'
    assert result['predictions']['spine_artifact']['status'] == 'undetermined'
    with pytest.raises(ValueError, match='Unsupported specialist region'):
        candidate.predict(np.zeros((20, 20), np.uint8), 'total-body')
    (tmp_path/'model.ts').write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        SpecialistQC(tmp_path)


def test_study_split_and_not_applicable_labels():
    from train_specialist import split_indices, targets_for
    labels = pd.DataFrame({'study_key': np.repeat(np.arange(40), 2),
                           'quality_class': np.tile([0, 1], 40), 'region': ['spine']*80,
                           **{name: [1]*80 for name in OUTPUTS[1:]}})
    a, b, c = split_indices(labels)
    assert not set(labels.study_key.iloc[a]) & set(labels.study_key.iloc[b])
    assert not set(labels.study_key.iloc[a]) & set(labels.study_key.iloc[c])
    targets = targets_for(labels)
    assert np.isnan(targets[:, -2:]).all()
    assert np.isfinite(targets[:, :4]).all()


def test_shadow_does_not_change_product_verdict(tmp_path, monkeypatch, bundle_available, trusted_synthetic_router):
    from conftest import synthetic_spine, write_dicom
    from dxaqc.dicom_io import read_dxa
    from dxaqc.pipeline import Analyzer, Options, process_files
    write_candidate(tmp_path)
    image_path = write_dicom(tmp_path/'input.dcm', synthetic_spine())
    img = read_dxa(image_path)
    monkeypatch.delenv('DXAQC_SPECIALIST_PATH', raising=False)
    base = Analyzer().analyze(img)
    monkeypatch.setenv('DXAQC_SPECIALIST_PATH', str(tmp_path))
    analyzer = Analyzer()
    shadow = analyzer.analyze(img)
    for key in ('quality', 'score', 'violations', 'criterion_states'):
        assert base[key] == shadow[key]
    output = tmp_path/'output'
    output.mkdir()
    rows = process_files([image_path], tmp_path, output, analyzer,
                         Options(explanations=False, keep_explanation_dir=True))
    assert json.loads(rows[0]['specialist_qc'])['mode'] == 'shadow'
    detail = json.loads(next((output/'images').glob('*.json')).read_text())
    assert detail['assessment']['specialist_qc']['affects_decision'] is False
    def broken(*args):
        raise RuntimeError('test inference failure')
    monkeypatch.setattr(analyzer.specialist, 'predict', broken)
    unavailable = analyzer.analyze(img)
    assert unavailable['quality'] == base['quality']
    assert unavailable['specialist_qc']['status'] == 'unavailable'


def test_training_exports_an_offline_loadable_candidate(tmp_path, monkeypatch):
    pytest.importorskip('safetensors.torch', reason='Training export is tested in the specialist environment')
    from types import SimpleNamespace
    import train_specialist as training
    labels = pd.DataFrame({'study_key': np.repeat(np.arange(40), 2),
                           'quality_class': np.tile([0, 1], 40), 'region': ['spine']*80,
                           'first_source_path': [f'{i}.dcm' for i in range(80)],
                           'pixel_sha256': [str(i) for i in range(80)], 'rows': [32]*80, 'columns': [32]*80,
                           **{name: [1]*80 for name in OUTPUTS[1:]}})
    label_path = tmp_path/'labels.csv'
    labels.to_csv(label_path, index=False)
    weights = tmp_path/'weights.safetensors'
    # The real checkpoint path is covered by check_specialists. Here use a small
    # encoder to test the entire training/export/runtime contract offline.
    root = training.Path(training.__file__).resolve().parents[1]
    catalog = json.loads((root/'docs/competition/specialist_sources.json').read_text())
    expected = next(a['sha256'] for a in catalog['artifacts'] if a['id'] == 'convnextv2_tiny_weights')
    real_digest = training.digest
    monkeypatch.setattr(training, 'digest', lambda p: expected if p == weights else real_digest(p))
    monkeypatch.setattr(training, 'create_encoder', lambda *a: torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()))
    def read(path):
        i = int(path.stem)
        return SimpleNamespace(pixels=np.full((32,32), i+1, np.uint8), pixel_sha256=str(i))
    monkeypatch.setattr(training, 'read_any', read)
    output = tmp_path/'candidate'
    report = training.run(SimpleNamespace(output=output, epochs=1, size=32, threads=1,
                                          labels=label_path, weights=weights, dataset=tmp_path,
                                          backbone='convnextv2_tiny', loss='bce'))
    history = json.loads((output/'history.json').read_text())
    assert len(history) == 1 and np.isfinite(history[0]['validation_loss'])
    assert report['thresholds']['spine_coverage'] is None
    candidate = SpecialistQC(output)
    assert candidate.predict(np.zeros((32,32),np.uint8), 'spine')['mode'] == 'shadow'
    with pytest.raises(ValueError, match='new output directory'):
        training.run(SimpleNamespace(output=output))
