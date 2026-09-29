import io
import fcntl
import zipfile
from collections import namedtuple

import pytest

import download_public_archive as module


@pytest.fixture(autouse=True)
def sufficient_disk_space(monkeypatch):
    # Download behavior is tested with in-memory fragments; the production
    # free-space guard depends on the CI host and is outside these scenarios.
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(module.shutil, 'disk_usage',
                        lambda path: Usage(40 * 1024**3, 0, 40 * 1024**3))


def archive_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('image.txt', b'fixture' * 50)
    return stream.getvalue()


def test_parallel_resume_keeps_a_contiguous_prefix_and_checks_zip(tmp_path, monkeypatch):
    body = archive_bytes()
    target = tmp_path / 'images.zip'
    target.with_suffix('.zip.part').write_bytes(body[:17])
    seen = []

    def fetch(url, start, end, size, path):
        assert size == len(body)
        seen.append((start, end))
        path.write_bytes(body[start:end + 1])
        return path

    monkeypatch.setattr(module, 'fetch_range', fetch)
    result = module.download('https://example.org/images.zip', target, len(body), workers=3, chunk_size=20)
    assert target.read_bytes() == body
    assert result['archive']['crc_verified']
    assert min(start for start, _ in seen) == 17
    assert not target.with_suffix('.zip.part').exists()
    assert module.download('https://example.org/images.zip', target, len(body))['status'] == 'verified'


def test_range_failure_preserves_downloaded_prefix(tmp_path, monkeypatch):
    target = tmp_path / 'images.zip'
    partial = target.with_suffix('.zip.part')
    partial.write_bytes(b'prefix')

    def fetch(*args):
        raise OSError('network failure')

    monkeypatch.setattr(module, 'fetch_range', fetch)
    with pytest.raises(OSError, match='network failure'):
        module.download('https://example.org/images.zip', target, 100, chunk_size=10)
    assert partial.read_bytes() == b'prefix'
    assert not target.exists()
    assert not list(tmp_path.glob('.ranges-*'))


@pytest.mark.parametrize('status,content_range', [(200, None), (206, 'bytes 1-9/10')])
def test_ignored_or_wrong_range_never_reaches_partial(tmp_path, monkeypatch, status, content_range):
    class Response(io.BytesIO):
        headers = {'Content-Range': content_range}

    def open_response(*args, **kwargs):
        response = Response(b'0123456789')
        response.status = status
        return response

    monkeypatch.setattr(module, 'urlopen', open_response)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    path = tmp_path / 'fragment'
    with pytest.raises(ValueError, match='exact HTTP range'):
        module.fetch_range('https://example.org/file', 0, 9, 10, path)
    assert not path.exists()


def test_invalid_archive_is_not_published(tmp_path, monkeypatch):
    def fetch(url, start, end, size, path):
        path.write_bytes(b'x' * (end - start + 1))
        return path

    monkeypatch.setattr(module, 'fetch_range', fetch)
    target = tmp_path / 'images.zip'
    with pytest.raises(zipfile.BadZipFile):
        module.download('https://example.org/images.zip', target, 100, chunk_size=20)
    assert not target.exists()
    assert target.with_suffix('.zip.part').stat().st_size == 100


def test_second_writer_is_rejected_before_touching_partial(tmp_path):
    target = tmp_path / 'images.zip'
    partial = target.with_suffix('.zip.part')
    partial.write_bytes(b'prefix')
    with target.with_name(target.name + '.download.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='Another downloader'):
            module.download('https://example.org/images.zip', target, 100)
    assert partial.read_bytes() == b'prefix'
