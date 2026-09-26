import zipfile
import json

import pytest

from competition.audit_downloaded_datasets import archive_inventory


def test_rejects_archive_path_escape(tmp_path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.dcm", b"example")
    with pytest.raises(ValueError, match="Unsafe"):
        archive_inventory(path)


def test_rejects_archive_symlink(tmp_path):
    path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, "outside")
    with pytest.raises(ValueError, match="Symlink"):
        archive_inventory(path)


def test_rejects_corrupt_archive_payload(tmp_path):
    path = tmp_path / "corrupt.zip"
    payload = b"unique-image-content"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("image.dcm", payload)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(payload, b"x" * len(payload)))
    with pytest.raises(ValueError, match="CRC"):
        archive_inventory(path)


def test_audit_does_not_publish_partial_owned_by_archive_downloader(tmp_path, monkeypatch):
    from competition import audit_downloaded_datasets as module

    project = tmp_path / 'project'
    docs = project / 'docs/competition'
    docs.mkdir(parents=True)
    (docs / 'download_sources.json').write_text('{}')
    root = tmp_path / 'downloads'
    directory = root / 'fixture'
    directory.mkdir(parents=True)
    partial = directory / 'images.zip.part'
    with zipfile.ZipFile(partial, 'w') as archive:
        archive.writestr('image.txt', b'fixture')
    spec = {'dataset': 'fixture', 'path': 'images.zip', 'bytes': partial.stat().st_size,
            'source': 'https://example.org/data', 'license': 'fixture'}
    (docs / 'additional_dataset_acquisition.json').write_text(json.dumps({'files': [spec], 'unavailable': []}))
    (directory / 'images.zip.download.lock').touch()
    monkeypatch.setattr(module, 'ROOT', project)
    result = module.audit(root, {})
    assert result['files'][0]['status'] == 'downloading'
    assert partial.exists() and not (directory / 'images.zip').exists()
