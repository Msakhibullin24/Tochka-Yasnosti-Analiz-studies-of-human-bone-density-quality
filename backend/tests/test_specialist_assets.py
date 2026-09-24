import hashlib
import io
from pathlib import Path
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import fetch_specialists as assets


def archive(path, entries):
    with tarfile.open(path, 'w:gz') as stream:
        for name, kind in entries:
            item = tarfile.TarInfo(name)
            if kind == 'link':
                item.type, item.linkname = tarfile.SYMTYPE, '/tmp/outside'
                stream.addfile(item)
            else:
                item.size = 3
                stream.addfile(item, io.BytesIO(b'abc'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_source_extraction_and_link_inventory(tmp_path):
    source = tmp_path/'source.tar.gz'
    checksum = archive(source, [('root/file.py', 'file'), ('root/alias', 'link')])
    dest = tmp_path/'sources'
    receipt = assets.extract_source(source, dest, checksum)
    assert (dest/'file.py').read_bytes() == b'abc'
    assert receipt['skipped_links'] == ['alias']
    assert not (dest/'alias').exists()
    assert assets.extract_source(source, dest, checksum) == receipt


@pytest.mark.parametrize('entries', [[('root/../escape', 'file')], [('/absolute', 'file')],
                                    [('a/x', 'file'), ('b/y', 'file')],
                                    [('a/x', 'file'), ('a/x', 'file')]])
def test_unsafe_archives_leave_no_partial_sources(tmp_path, entries):
    source = tmp_path/'source.tar.gz'
    checksum = archive(source, entries)
    with pytest.raises(ValueError):
        assets.extract_source(source, tmp_path/'dest', checksum)
    assert not (tmp_path/'dest').exists()
    assert not list(tmp_path.glob('.extract-*'))


def test_verify_is_offline_and_missing_weights_are_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'download', lambda *a: pytest.fail('verify attempted network'))
    catalog = {'capabilities': [], 'artifacts': [{'id': 'x', 'path': 'x.pt', 'bytes': 3,
                'sha256': hashlib.sha256(b'abc').hexdigest(), 'kind': 'weights'}]}
    report = assets.prepare(catalog, tmp_path, verify=True)
    assert not report['artifacts_ready'] and not report['clinical_models_ready']
    (tmp_path/'x.pt').write_bytes(b'abc')
    assert assets.prepare(catalog, tmp_path, verify=True)['artifacts_ready']


def test_focused_download_does_not_fetch_or_overwrite_full_inventory(tmp_path, monkeypatch):
    payload = b'dxa checkpoint fixture'
    artifact = {'id': 'dxa_to_3d_weights', 'kind': 'weights', 'path': 'weights/dxa.pt',
                'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    unrelated = {'id': 'unrelated', 'kind': 'weights', 'path': 'weights/other.pt',
                 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    catalog = {'artifacts': [artifact, unrelated], 'capabilities': []}

    def fake_download(output, source, spec):
        target = output / spec['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return 'downloaded'

    monkeypatch.setattr(assets, 'download', fake_download)
    report = assets.prepare(catalog, tmp_path, only={'dxa_to_3d_weights'})
    assert report['artifacts_ready']
    assert [item['id'] for item in report['artifacts']] == ['dxa_to_3d_weights']
    assert report['scope']['selected_artifact_ids'] == ['dxa_to_3d_weights']
    assert (tmp_path / 'selected-assets-status.json').is_file()
    assert not (tmp_path / 'assets-status.json').exists()
    assert not (tmp_path / unrelated['path']).exists()
    with pytest.raises(ValueError, match='Unknown artifact IDs'):
        assets.prepare(catalog, tmp_path / 'unknown', only={'not-in-catalog'})
