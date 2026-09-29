"""Dataset extraction never publishes a partially read or ambiguous hip subset."""
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from experiments import fetch_fracatlas_hips as fetch


def fixture_archive(root,ids=('one.jpg',)):
    root.mkdir()
    path=root/'fixture.zip'
    header='image_id,hip,frontal,lateral,oblique,mixed,multiscan\n'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('dataset.csv',header+''.join(f'{name},1,1,0,0,0,0\n' for name in ids))
        for name in set(ids):z.writestr('images/'+name,b'publisher bytes')
    return path


def test_successful_acquisition_preserves_exact_source_bytes(tmp_path,monkeypatch):
    root=tmp_path/'source';archive=fixture_archive(root)
    monkeypatch.setattr(fetch,'fetch_archive',lambda *args:archive)
    report=fetch.acquire(root)
    assert report['hip_images']==1
    assert (root/'hip-subset/images/one.jpg').read_bytes()==b'publisher bytes'
    assert json.loads((root/'acquisition.json').read_text())['archive_sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    assert not list(root.glob('.fracatlas-hips-*'))


def test_crc_read_failure_does_not_publish_partial_subset(tmp_path,monkeypatch):
    root=tmp_path/'source';archive=fixture_archive(root)
    monkeypatch.setattr(fetch,'fetch_archive',lambda *args:archive)
    original=zipfile.ZipFile.read
    def read(self,item,*args,**kwargs):
        name=item.filename if isinstance(item,zipfile.ZipInfo) else item
        if name.endswith('.jpg'):raise zipfile.BadZipFile('CRC failed')
        return original(self,item,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'read',read)
    with pytest.raises(zipfile.BadZipFile):fetch.acquire(root)
    assert not (root/'hip-subset').exists() and not (root/'acquisition.json').exists()
    assert not list(root.glob('.fracatlas-hips-*'))


def test_duplicate_csv_images_are_rejected(tmp_path,monkeypatch):
    root=tmp_path/'source';archive=fixture_archive(root,('one.jpg','one.jpg'))
    monkeypatch.setattr(fetch,'fetch_archive',lambda *args:archive)
    with pytest.raises(ValueError,match='duplicate'):fetch.acquire(root)
    assert not (root/'hip-subset').exists()
