import numpy as np
import pytest
import json
from types import SimpleNamespace

import evaluate_projection_candidate as candidate
from evaluate_projection_candidate import assessment, variants


def test_projection_candidate_abstains_instead_of_forcing_lateral_on_uncertain_score():
    assert assessment(.577)['value'] == 'unknown'
    assert assessment(.95)['value'] == 'lateral'
    assert assessment(.03)['value'] == 'frontal'
    assert assessment(.95)['clinical_validation'] is False
    assert assessment(.95)['model_affects_decision'] is False
    for bad in (float('nan'), float('inf'), -1., 1.01):
        with pytest.raises(ValueError):
            assessment(bad)


def test_projection_representation_removes_native_aspect_ratio_and_polarity_shortcut():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    original, inverted = variants(image)
    assert original.shape == inverted.shape == (320, 320)
    assert np.array_equal(inverted, 255-original)


def test_training_collapses_copies_and_keeps_patient_pairs_in_one_fold(tmp_path, monkeypatch):
    root = tmp_path/'external'
    decoded = {}
    for index in range(5):
        for branch, name, lateral in (('BMD', 'spine_image.dcm', 0),
                                      ('VFA', 'image.dcm', 1)):
            for part in ('Annotation', 'Verification'):
                path = root/branch/part/f'{index:04d}'/'images'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f'{index}:{branch}'.encode())
                decoded[path] = SimpleNamespace(pixel_sha256=f'{index}:{branch}',
                                                pixels=np.full((8, 8), lateral, dtype=np.uint8))
    dataset = tmp_path/'organizer'
    dataset.mkdir()
    source = dataset/'source.dcm'
    source.write_bytes(b'GE AP')
    decoded[source] = SimpleNamespace(pixel_sha256='GE pixels', pixels=np.zeros((8, 8), dtype=np.uint8))
    labels = tmp_path/'labels.csv'
    labels.write_text('first_source_path,region,quality_class,study_key\nsource.dcm,spine,0,ge-study\n')
    monkeypatch.setattr(candidate, 'read_dxa', lambda path: decoded[path])
    monkeypatch.setattr(candidate, 'features',
                        lambda pixels: np.repeat([[float(pixels.mean()), 1.]], 2, axis=0))
    report = candidate.run(root, dataset, labels, tmp_path/'result')
    assert report['duplicate_source_copies_collapsed'] == 10
    assert report['source_images'] == 10
    assert report['source_patient_groups'] == 5
    by_patient = {}
    for record in report['source_oof']:
        by_patient.setdefault(record['patient_group'], set()).add(record['fold'])
    assert all(len(folds) == 1 for folds in by_patient.values())
    assert report['model_affects_decision'] is False
    assert json.loads((tmp_path/'result/evaluation.json').read_text())['clinical_validation'] is False
    result = candidate.predict(tmp_path/'result/projection.joblib', source)
    assert result['pixel_sha256'] == 'GE pixels'
    import joblib
    path=tmp_path/'result/projection.joblib'
    bundle=joblib.load(path);bundle['method']='multisource_frozen_resnet18_lr'
    joblib.dump(bundle,path)
    result=candidate.predict(path,source)
    assert result['method']=='multisource_frozen_resnet18_lr'
    assert result['clinical_validation'] is False
