"""Offline checks: source checksums, interrupted downloads and unsafe destinations."""
import hashlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("download_training_data", Path(__file__).parents[1] / "scripts/download_training_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_download_checks_integrity_and_preserves_existing(tmp_path):
    payload = b"sample image"
    source = {"base_url": "https://example.org/"}
    item = {"path": "images/a.png", "bytes": len(payload), "algorithm": "git-sha1",
            "digest": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
    with patch.object(module, "urlopen", return_value=io.BytesIO(payload)) as request:
        assert module.download(tmp_path, source, item) == "downloaded"
        assert module.download(tmp_path, source, item) == "verified"
        assert request.call_count == 1
    target = tmp_path / item["path"]
    target.write_bytes(b"local changes")
    with pytest.raises(ValueError, match="Existing file"):
        module.download(tmp_path, source, item)
    assert target.read_bytes() == b"local changes"


def test_rejects_corruption_and_path_escape(tmp_path):
    source = {"base_url": "https://example.org/"}
    item = {"path": "a.png", "bytes": 3, "algorithm": "sha256", "digest": hashlib.sha256(b"abc").hexdigest()}
    with patch.object(module, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(b"bad")), patch.object(module.time, "sleep"):
        with pytest.raises(ValueError, match="checksum"):
            module.download(tmp_path, source, item)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="Unsafe"):
        module.download(tmp_path, source, {**item, "path": "../escape.png"})
