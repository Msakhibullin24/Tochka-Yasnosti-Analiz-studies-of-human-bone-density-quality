"""Research boundary checks: frame preservation, no leaked features, safe transfer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from gpu_research.common import digest, letterbox, load_feature_bundle, representations
from gpu_research.transfer import build, verify


def test_research_entrypoints_import_in_clean_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(root)}
    # A fresh interpreter cannot inherit sys.path changes from another test.
    for script in ('gpu_research/finetune.py', 'experiments/nested_criterion_upgrade.py',
                   'experiments/verify_metric_candidate.py'):
        completed = subprocess.run([sys.executable, str(root / script), '--help'],
                                   cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr


def test_full_frame_letterbox_preserves_borders_and_aspect():
    pixels = np.full((20, 40), 255, np.uint8)
    image, transform = letterbox(pixels, 80)
    assert image.shape == (80, 80, 3)
    assert transform['width'] == 80 and transform['height'] == 40
    assert np.all(image[20:60] == 255)
    assert not image[:20].any() and not image[60:].any()


def test_feature_bundle_rejects_wrong_order_tampering_and_finetuned_encoder(tmp_path):
    path = tmp_path / 'features.npy'
    np.save(path, np.ones((2, 3), np.float32))
    labels = pd.DataFrame({'first_source_path': ['a', 'b']})
    metadata = {'schema_version': 1, 'paths': ['a', 'b'], 'labels_sha256': 'labels',
                'source_fingerprint': 'pixels', 'features_sha256': digest(path), 'encoder_finetuned': False}
    path.with_suffix('.json').write_text(json.dumps(metadata))
    values, _ = load_feature_bundle(path, labels, 'pixels', 'labels')
    assert values.shape == (2, 3)
    with pytest.raises(ValueError, match='identities'):
        load_feature_bundle(path, labels.iloc[::-1], 'pixels', 'labels')
    metadata['encoder_finetuned'] = True
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='frozen external'):
        load_feature_bundle(path, labels, 'pixels', 'labels')
    metadata['encoder_finetuned'] = False
    path.with_suffix('.json').write_text(json.dumps(metadata))
    np.save(path, np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match='checksum'):
        load_feature_bundle(path, labels, 'pixels', 'labels')


def test_private_transfer_roundtrip_and_corruption(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'test.dcm').write_bytes(b'original-pixels')
    output = tmp_path / 'private.zip'
    report = build(root, output)
    assert report['files'] == 1
    verify(output, tmp_path / 'unpacked')
    assert (tmp_path / 'unpacked/dataset/test.dcm').read_bytes() == b'original-pixels'
    with pytest.raises(ValueError, match='new'):
        verify(output, tmp_path / 'unpacked')
    with zipfile.ZipFile(tmp_path / 'bad.zip', 'w') as archive:
        manifest = {'schema_version': 1, 'files': {'dataset/test.dcm': {'sha256': 'bad', 'bytes': 7}}}
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('dataset/test.dcm', b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify(tmp_path / 'bad.zip', tmp_path / 'must-not-exist')
    assert not (tmp_path / 'must-not-exist').exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'dataset\\escape', 'C:/escape', 'dataset/./escape'])
def test_transfer_rejects_unsafe_paths_before_writing(tmp_path, name):
    path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', '{}')
        archive.writestr(name, 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        verify(path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_dinov3_pooling_excludes_register_tokens():
    from types import SimpleNamespace
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.config = SimpleNamespace(num_register_tokens=2)
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[1., 1.], [99., 99.], [99., 99.],
                                                                    [2., 2.], [4., 4.]]]))
    vector, patches = representations(Encoder(), torch.zeros(1, 3, 32, 32), 'dinov3_vit', 'global-mean')
    assert patches.shape == (1, 2, 2)
    assert torch.equal(vector, torch.tensor([[1., 1., 3., 3.]]))


def test_masked_labels_and_class_weights_never_use_validation_targets():
    from gpu_research.finetune import class_weights, masked_bce
    targets = np.array([[0., np.nan], [1., 1.], [1., 0.]], dtype=np.float32)
    weights, supported = class_weights(targets, np.array([0, 1]))
    assert np.array_equal(supported, [True, False])
    changed = targets.copy()
    changed[2] = np.nan
    assert np.array_equal(class_weights(changed, np.array([0, 1]))[0], weights)
    logits = torch.zeros((3, 2), requires_grad=True)
    loss = masked_bce(logits, torch.tensor(targets), torch.tensor(weights), torch.tensor(supported))
    loss.backward()
    assert torch.isfinite(loss) and torch.equal(logits.grad[:, 1], torch.zeros(3))


@pytest.mark.parametrize('kind', ['siglip', 'dinov3_vit'])
def test_real_tiny_foundation_forward_backward_and_tail_selection(kind):
    transformers = pytest.importorskip('transformers', minversion='5.17.0')
    from gpu_research.finetune import FoundationQC
    if kind == 'siglip':
        config = transformers.SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                                  num_attention_heads=4, image_size=28, patch_size=14)
        encoder = transformers.SiglipVisionModel(config)
        size = 28
    else:
        config = transformers.DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                              num_attention_heads=4, image_size=32, patch_size=16)
        encoder = transformers.DINOv3ViTModel(config)
        size = 32
    model = FoundationQC(encoder, kind, 'global-mean', 64, 1)
    logits = model(torch.randn(2, 3, size, size))
    logits.square().mean().backward()
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    blocks = encoder.encoder.layers if kind == 'siglip' else encoder.model.layer
    assert all(p.grad is None for p in blocks[0].parameters())
    assert any(p.grad is not None for p in blocks[1].parameters())
    assert torch.isfinite(model.head.weight.grad).all()


def test_complete_cpu_training_smoke_reads_real_dicom_and_keeps_test_studies_out(tmp_path, monkeypatch):
    transformers = pytest.importorskip('transformers', minversion='5.17.0')
    from argparse import Namespace
    from conftest import write_dicom
    from dxaqc.dicom_io import read_any
    from gpu_research import finetune
    rows, predictions = [], []
    dataset = tmp_path / 'dicom'
    dataset.mkdir()
    rng = np.random.default_rng(17)
    for index in range(16):
        study = f'study-{index//2}'
        name = f'{index:02d}.dcm'
        path = write_dicom(dataset / name, rng.integers(0, 120, (84, 84), dtype=np.uint8),
                           study_uid=f'1.2.3.{index//2+1}', sop_uid=f'1.2.4.{index+1}')
        image = read_any(path)
        truth = index % 2
        rows.append({'first_source_path': name, 'study_key': study, 'region': 'spine', 'quality_class': truth,
                     'pixel_sha256': image.pixel_sha256, 'rows': 84, 'columns': 84,
                     'spine_coverage': 0, 'spine_axis': 0, 'spine_artifact': truth,
                     'hip_position_rotation': np.nan, 'hip_roi_coverage': np.nan})
        predictions.append({'index': index, 'repeat': 0, 'fold': (index//2) % 5, 'study': study,
                            'source_path': name, 'true_region': 'spine', 'predicted_region': 'spine',
                            'quality_true': truth, 'quality_pred': truth, 'quality_score': float(truth),
                            'violation_type': 'Присутствуют посторонние предметы' if truth else '',
                            'criterion_states': json.dumps({'spine_axis': {'angle_deg': 0.}})})
    labels, baseline = tmp_path / 'labels.csv', tmp_path / 'baseline.csv'
    pd.DataFrame(rows).to_csv(labels, index=False)
    pd.DataFrame(predictions).to_csv(baseline, index=False)
    def tiny_encoder(*args):
        config = transformers.SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                                  num_attention_heads=4, image_size=42, patch_size=14)
        return transformers.SiglipImageProcessor(), transformers.SiglipVisionModel(config)
    monkeypatch.setattr(finetune, 'load_encoder', tiny_encoder)
    monkeypatch.setattr(finetune, 'checkpoint_identity', lambda *args: {'weights_sha256': 'tiny-test-only'})
    args = Namespace(output=tmp_path / 'run', epochs=1, batch_size=2, last_blocks=0, patience=1,
                     head_lr=.001, encoder_lr=.00001, device='cpu', precision='fp32', model='medsiglip',
                     size=42, labels=labels, baseline=baseline, dataset=dataset, model_dir=tmp_path,
                     seed=17, pooling='global-mean', fold=0)
    finetune.run(args)
    report = json.loads((args.output / 'evaluation.json').read_text())
    assert report['partial'] is True and 'metrics' not in report and 'screen' not in report
    fold = report['folds'][0]
    assert not set(fold['training_studies']) & set(fold['test_studies'])
    assert not set(fold['validation_studies']) & set(fold['test_studies'])
    assert 'spine_coverage' in fold['unvalidated_thresholds']
    assert (args.output / 'fold-0/adapter.pt').exists()
    assert len(pd.read_csv(args.output / 'partial_predictions.csv')) == 4


def test_anatomy_partial_targets_and_mirroring_keep_unknown_padding():
    from gpu_research.anatomy import reviewed_target, to_patch_grid, loss_known
    case = {'height': 4, 'width': 8, 'named_regions': {
        'L1': {'visible': True, 'polygon': [[0, 0], [2, 0], [2, 3], [0, 3]]},
        'L2': {'visible': False, 'polygon': None}}}
    target = reviewed_target(case, ['L1', 'L2', 'L3'])
    assert np.all(target[2] == -1) and not target[1].any()
    transformed = to_patch_grid(target, {'canvas_size': 8, 'x': 0, 'y': 2, 'width': 8, 'height': 4}, 8, True)
    assert np.all(transformed[:, :2] == -1)
    assert transformed[0, 3, 7] == 1 and transformed[0, 3, 0] == 0
    logits = torch.zeros_like(torch.tensor(transformed), requires_grad=True)
    loss_known(logits, torch.tensor(transformed)).backward()
    assert not logits.grad[2].any() and not logits.grad[:, :2].any()


def test_anatomy_blank_template_is_rejected_before_training(tmp_path):
    from argparse import Namespace
    from gpu_research.anatomy import train
    reference = tmp_path / 'template.json'
    reference.write_text(json.dumps({'review_package_version': 4, 'reviewer': '', 'cases': []}))
    with pytest.raises(ValueError, match='Expert provenance'):
        train(Namespace(output=tmp_path / 'out', epochs=1, batch_size=1, device='cpu', reference=reference))
    assert not (tmp_path / 'out').exists()


def test_anatomy_training_roundtrip_with_confirmed_contours(tmp_path):
    from argparse import Namespace
    from conftest import write_dicom
    from dxaqc.dicom_io import read_any
    from gpu_research.anatomy import train
    from gpu_research.common import dump
    dataset, features = tmp_path / 'images', tmp_path / 'features'
    dataset.mkdir()
    features.mkdir()
    cases, transforms, paths, dense_sha = [], [], [], {}
    rng = np.random.default_rng(18)
    for i in range(8):
        path = dataset / f'{i}.dcm'
        write_dicom(path, rng.integers(0, 120, (84, 84), dtype=np.uint8),
                    study_uid=f'1.2.3.{i+1}', sop_uid=f'1.2.4.{i+1}')
        image = read_any(path)
        cases.append({'path_to_file': path.name, 'status': 'confirmed', 'image_uid': image.image_uid,
                      'height': 84, 'width': 84, 'study_group': f'g-{i}', 'source_sha256': digest(path),
                      'pixel_sha256': image.pixel_sha256,
                      'named_regions': {'L1': {'visible': True, 'polygon': [[0, 0], [30, 0], [30, 30], [0, 30]]}}})
        paths.append(path.name)
        transforms.append({'path': path.name, 'source_width': 84, 'source_height': 84,
                           'pixel_sha256': image.pixel_sha256, 'mirrored': False,
                           'x': 0, 'y': 0, 'width': 64, 'height': 64})
        patch_path = features / f'patches_{i:03d}.npy'
        np.save(patch_path, rng.normal(size=(16, 4)).astype(np.float16))
        dense_sha[patch_path.name] = digest(patch_path)
    feature_path = features / 'features.npy'
    np.save(feature_path, rng.normal(size=(8, 4)).astype(np.float32))
    dump(features / 'features.json', {'encoder_finetuned': False, 'dense_features': True, 'smoke_only': False,
                                     'features_sha256': digest(feature_path), 'model': 'dinov3-b', 'size': 64,
                                     'paths': paths, 'transforms': transforms, 'dense_sha256': dense_sha,
                                     'weights_sha256': 'synthetic-test-only'})
    reference = tmp_path / 'reference.json'
    dump(reference, {'review_package_version': 4, 'reviewer': 'synthetic-test-only',
                     'annotation_source': 'synthetic-test-only', 'coordinate_system': 'original_pixel_centres', 'cases': cases})
    args = Namespace(output=tmp_path / 'run', epochs=1, batch_size=2, device='cpu', reference=reference,
                     features=features, dataset=dataset, region='spine')
    train(args)
    report = json.loads((args.output / 'evaluation.json').read_text())
    assert not report['clinical_validation'] and (args.output / 'mask_head.pt').exists()
    assert report['metrics']['L1']['known_pixels'] > 0
    assert report['metrics']['L2']['dice_at_patch_grid'] is None
    assert not {r['group'] for r in report['training']} & {r['group'] for r in report['test']}
