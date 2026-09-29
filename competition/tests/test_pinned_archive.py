"""Pinned source retrieval rejects drift and unsafe ZIPs before publication."""
import hashlib
import io
import zipfile
import pytest
from experiments.pinned_archive import fetch_archive,safe_members


def test_existing_archive_is_reverified_without_network(tmp_path):
    content=b'published bytes';p=tmp_path/'archive.zip';p.write_bytes(content)
    assert fetch_archive(p,'https://invalid.example',len(content),hashlib.md5(content).hexdigest())==p
    p.write_bytes(b'changed bytes!!')
    with pytest.raises(ValueError,match='checksum'):
        fetch_archive(p,'https://invalid.example',len(content),hashlib.md5(content).hexdigest())


@pytest.mark.parametrize('name',['../escape.jpg','/absolute.jpg','directory\\escape.jpg'])
def test_archive_traversal_is_rejected(name):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr(name,b'pixels')
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as z:
        with pytest.raises(ValueError,match='Unsafe'):safe_members(z)


def test_archive_budget_and_symlink_are_rejected():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        item=zipfile.ZipInfo('image.jpg');item.create_system=3;item.external_attr=0o120777<<16
        z.writestr(item,b'outside')
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as z:
        with pytest.raises(ValueError,match='Unsafe'):safe_members(z)
        with pytest.raises(ValueError,match='budget'):safe_members(z,byte_budget=1)
