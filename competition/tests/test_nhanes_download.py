import pytest

from download_nhanes_lumbar import inventory, fingerprint, download


def test_cdc_index_selects_only_lumbar_and_excludes_empty_source_files():
    html = '100 <A HREF="/pub/NHANES/XRays/Nhanes2/L00123.tiff">L00123.tiff</A>'
    html += '100 <A HREF="/pub/NHANES/XRays/Nhanes2/C00123.tiff">C00123.tiff</A>'
    html += '0 <A HREF="/pub/NHANES/XRays/Nhanes2/L00452.tiff">L00452.tiff</A>'
    rows = inventory(html)
    assert len(rows) == 1 and rows[0]['path'] == 'L00123.tiff'
    assert rows[0]['bytes'] == 100
    with pytest.raises(ValueError):
        inventory(html + html)


def test_download_verification_rejects_html_and_wrong_source_size(tmp_path):
    p = tmp_path/'image.tiff'
    p.write_bytes(b'<html>failure</html>')
    with pytest.raises(ValueError, match='not a TIFF'):
        fingerprint(p, p.stat().st_size)
    p.write_bytes(b'II*\x00' + b'\x00'*12)
    assert len(fingerprint(p, 16)) == 64
    with pytest.raises(ValueError, match='size differs'):
        fingerprint(p, 17)


def test_complete_partial_is_verified_before_resuming_http(tmp_path, monkeypatch):
    p = tmp_path/'L00123.tiff.part'
    p.write_bytes(b'II*\x00' + b'\x00'*12)
    def unexpected_network(*args, **kwargs):
        raise AssertionError('Complete partial must not issue another range request')
    monkeypatch.setattr('download_nhanes_lumbar.subprocess.run', unexpected_network)
    row = download(tmp_path, {'path': 'L00123.tiff', 'bytes': 16,
                             'url': 'https://ftp.cdc.gov/pub/NHANES/XRays/Nhanes2/L00123.tiff'})
    assert row['status'] == 'downloaded'
    assert len(row['sha256']) == 64
    assert (tmp_path/'L00123.tiff').exists()
    assert not p.exists()
