import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from source_integrity import inspect_sources


def write_labels(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'first_source_path', 'study_key', 'pixel_sha256', 'rows', 'columns', 'quality_class'])
        writer.writeheader()
        writer.writerows(rows)


def row(path: str, study: str, quality: str = '0') -> dict:
    return {'first_source_path': path, 'study_key': study, 'pixel_sha256': 'historical',
            'rows': '2', 'columns': '3', 'quality_class': quality}


def test_fresh_pixel_identity_rejects_extra_copy_of_organiser_image(tmp_path):
    primary = tmp_path / 'primary'
    extra = tmp_path / 'extra'
    primary.mkdir()
    extra.mkdir()
    (primary / 'a.dcm').touch()
    (extra / 'b.dcm').touch()
    first = tmp_path / 'primary.csv'
    second = tmp_path / 'extra.csv'
    write_labels(first, [row('a.dcm', 'study-a')])
    write_labels(second, [row('b.dcm', 'study-b')])

    def read(_):
        return SimpleNamespace(pixels=np.zeros((2, 3)), pixel_sha256='current-copy')

    report = inspect_sources([('organiser', first, primary)], reader=read)
    assert report['historical_hash_changed'] == 1
    assert report['patient_disjointness_proven'] is False
    with pytest.raises(ValueError, match='crosses data sources'):
        inspect_sources([('organiser', first, primary), ('extra', second, extra)], reader=read)


def test_duplicate_within_study_is_reported_but_across_studies_is_rejected(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    for name in ('a.dcm', 'b.dcm'):
        (root / name).touch()
    labels = tmp_path / 'labels.csv'
    write_labels(labels, [row('a.dcm', 'study-a'), row('b.dcm', 'study-a')])
    read = lambda _: SimpleNamespace(pixels=np.zeros((2, 3)), pixel_sha256='copy')
    assert inspect_sources([('organiser', labels, root)], reader=read)['within_study_pixel_copies'] == 1
    write_labels(labels, [row('a.dcm', 'study-a'), row('b.dcm', 'study-b')])
    with pytest.raises(ValueError, match='study groups'):
        inspect_sources([('organiser', labels, root)], reader=read)


def test_changed_size_invalid_quality_and_escaping_path_are_rejected(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    (root / 'a.dcm').touch()
    labels = tmp_path / 'labels.csv'
    read = lambda _: SimpleNamespace(pixels=np.zeros((3, 3)), pixel_sha256='current')
    write_labels(labels, [row('a.dcm', 'study-a')])
    with pytest.raises(ValueError, match='size changed'):
        inspect_sources([('organiser', labels, root)], reader=read)
    write_labels(labels, [row('a.dcm', 'study-a', '2')])
    with pytest.raises(ValueError, match='quality_class'):
        inspect_sources([('organiser', labels, root)], reader=read)
    write_labels(labels, [row('../outside.dcm', 'study-a')])
    with pytest.raises(ValueError, match='escapes dataset root'):
        inspect_sources([('organiser', labels, root)], reader=read)


def test_training_cache_changes_when_decoded_source_fingerprint_changes(tmp_path, monkeypatch):
    import pandas as pd
    import train

    current = {'pixel': 1}
    calls = []

    def read(_path, _mm=None):
        calls.append(current['pixel'])
        return SimpleNamespace(pixels=np.full((2, 3), current['pixel'], dtype=np.uint8),
                               pixel_mm=1.05, pixel_mm_x=0.6)

    monkeypatch.setattr(train, 'read_any', read)
    monkeypatch.setattr(train, 'embed', lambda pixels: np.array([pixels[0, 0]], dtype=float))
    monkeypatch.setattr(train, 'measure_image',
                        lambda pixels, *_: SimpleNamespace(features={'mean': int(pixels.mean())}))
    labels = pd.DataFrame([{'first_source_path': 'a.dcm', 'region': 'spine'}])
    first = train.build_table(tmp_path, labels, tmp_path / 'cache', 'fingerprint-1')
    current['pixel'] = 2
    second = train.build_table(tmp_path, labels, tmp_path / 'cache', 'fingerprint-2')
    assert first[0][0]['mean'] == 1 and second[0][0]['mean'] == 2
    train.build_table(tmp_path, labels, tmp_path / 'cache', 'fingerprint-2')
    assert calls == [1, 2]
