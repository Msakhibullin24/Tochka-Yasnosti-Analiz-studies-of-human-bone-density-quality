import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from conftest import synthetic_spine, write_dicom
from dxaqc.store import Store, Conflict


def test_store_survives_reopen_and_claims_once(tmp_path):
    first = Store(tmp_path)
    first.add('a', tmp_path / 'in', tmp_path / 'out')
    second = Store(tmp_path)
    assert second.claim()['id'] == 'a'
    assert first.claim() is None
    assert first.interrupted()[0]['id'] == 'a'
    first.update('a', status='finished', summary={'files': 1})
    assert Store(tmp_path).get('a')['summary'] == {'files': 1}
    first.review('a', 'image', 0, {'author': 'doctor'})
    with pytest.raises(Conflict):
        second.review('a', 'image', 0, {'author': 'another'})
    assert len(second.reviews('a', 'image')) == 1


def test_review_roundtrip_restart_conflict_and_machine_immutability(tmp_path, monkeypatch, trusted_synthetic_router):
    from dxaqc import api
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path / 'work')
    image = write_dicom(tmp_path / 'a.dcm', synthetic_spine(4))
    with TestClient(api.app) as client:
        job = client.post('/api/v1/batch?wait=true', files={'files': ('a.dcm', image.read_bytes())}).json()
        assert job['status'] == 'finished'
        base = '/api/v1/jobs/' + job['id']
        original = client.get(base + '/results.csv').content
        row = client.get(base + '/rows').json()[0]
        assert client.get(base + '/worklist').json()[0]['review_status'] == 'unreviewed'
        image_base = base + '/images/' + row['row_id']
        assert client.get(image_base + '/original.png').status_code == 200
        assert client.post(image_base + '/geometry-evaluation', json={'geometry': []}).json() == []
        assert client.get('/api/v1/ready').json()['ready'] is True
        document = {'expected_revision': 0, 'author': 'Эксперт', 'status': 'confirmed', 'quality_class': 0,
                    'violations': [], 'comment': 'Проверено', 'geometry': [
                        {'name': 'axis', 'kind': 'line', 'points': [{'x': 0.5, 'y': 0.1}, {'x': 0.5, 'y': 0.9}]}]}
        result = client.post(image_base + '/reviews', json=document)
        assert result.status_code == 200, result.text
        assert result.json()['revision'] == 1
        assert result.json()['roi_evaluation'] == []
        assert client.get(base + '/validation.json').json()['valid'] is True
        from dxaqc.validate_results import validate
        out = tmp_path / 'series-check'
        out.mkdir()
        (out / 'results.csv').write_bytes(original)
        (out / 'additional_series.zip').write_bytes(client.get(base + '/additional_series.zip').content)
        assert validate(out / 'results.csv', series=out / 'additional_series.zip')['valid'] is True
        import pydicom
        with zipfile.ZipFile(out / 'additional_series.zip') as archive:
            sr_path = next(name for name in archive.namelist() if name.endswith('_sr.dcm'))
            sr = pydicom.dcmread(io.BytesIO(archive.read(sr_path)))
            assert sr.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].SeriesInstanceUID == pydicom.dcmread(image).SeriesInstanceUID
        assert client.get(base + '/worklist').json()[0]['review_status'] == 'confirmed'
        assert result.json()['measurements']['axis']['angle_from_vertical_deg'] == 0
        assert client.post(image_base + '/reviews', json=document).status_code == 409
        assert client.get(base + '/results.csv').content == original

        assert 'Эксперт' in client.get(base + '/reviewed.csv').text
        assert client.get(base + '/reviews.json').json()['images'][0]['reviews'][0]['author'] == 'Эксперт'
        region = client.get(image_base).json()['region']
        wrong_code = 'hip_roi_coverage' if region == 'spine' else 'spine_axis'
        invalid = {**document, 'expected_revision': 1, 'quality_class': 1, 'violations': [wrong_code]}
        assert client.post(image_base + '/reviews', json=invalid).status_code == 422
        api.store().update(job['id'], status='cancelled')
        assert client.post(image_base + '/reviews', json={**document, 'expected_revision': 1}).status_code == 200
        assert client.get(base + '/results.csv').content == original
    with TestClient(api.app) as client:
        assert client.get('/api/v1/jobs').json()[0]['id'] == job['id']
        assert client.get(image_base + '/reviews').json()[0]['comment'] == 'Проверено'
        assert client.get(base + '/results.csv').content == original


def test_worklist_uses_latest_revision_and_isolates_jobs(tmp_path, monkeypatch):
    from dxaqc import api
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path)
    store = Store(tmp_path)
    for job_id in ('one', 'two'):
        out = tmp_path / job_id
        out.mkdir()
        (out / 'results.csv').write_text('row_id,processing_status\na,Success\nb,Failure\n', encoding='utf-8')
        store.add(job_id, tmp_path / 'input', out)
        store.update(job_id, status='finished')
    store.review('one', 'a', 0, {'status': 'confirmed'})
    store.review('one', 'a', 1, {'status': 'draft'})
    store.review('two', 'a', 0, {'status': 'not_evaluable'})
    with TestClient(api.app) as client:
        rows = client.get('/api/v1/jobs/one/worklist').json()
        assert rows[0]['review_status'] == 'draft'
        assert rows[0]['review_revision'] == '2'
        assert rows[1]['review_status'] == 'unavailable'
        assert client.get('/api/v1/jobs/two/worklist').json()[0]['review_status'] == 'not_evaluable'


def test_crash_recovery_has_downloadable_failure(tmp_path, monkeypatch):
    from dxaqc import api
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path)
    out = tmp_path / 'a/out'
    store = Store(tmp_path)
    store.add('a', tmp_path / 'missing', out)
    store.claim()
    with TestClient(api.app) as client:
        assert client.get('/api/v1/jobs/a').json()['status'] == 'failed'
        assert 'BATCH_INPUT_ERROR' in client.get('/api/v1/jobs/a/results.csv').text


def test_api_archive_error_is_downloadable(tmp_path, monkeypatch):
    from dxaqc import api, pipeline
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(pipeline, 'MAX_ARCHIVE_BYTES', 1)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w') as z:
        z.writestr('a.dcm', b'1234')
    with TestClient(api.app) as client:
        job = client.post('/api/v1/batch?wait=true', files={'files': ('a.zip', payload.getvalue())}).json()
        assert job['status'] == 'failed'
        assert client.get('/api/v1/jobs/' + job['id'] + '/results.csv').status_code == 200


def test_review_rejects_nonfinite_geometry():
    from pydantic import ValidationError
    from dxaqc.review import Review
    with pytest.raises(ValidationError):
        Review(expected_revision=0, author='x', status='draft', geometry=[
            {'name': 'x', 'kind': 'point', 'points': [{'x': float('nan'), 'y': 0}]}])


def test_backup_restores_reviews_and_relocates_artifacts(tmp_path):
    from dxaqc.maintenance import backup, restore
    root = tmp_path / 'source'
    store = Store(root)
    out = root / 'a/out'
    out.mkdir(parents=True)
    (out / 'results.csv').write_text('a,b\n1,2\n')
    store.add('a', tmp_path / 'external', out)
    store.update('a', status='finished')
    store.review('a', 'r', 0, {'author': 'expert'})
    archive = tmp_path / 'backup.zip'
    backup(root, archive)
    target = tmp_path / 'restored'
    restore(target, archive)
    restored = Store(target)
    assert restored.get('a')['out'] == str(target / 'a/out')
    assert (target / 'a/out/results.csv').read_text() == 'a,b\n1,2\n'
    assert restored.reviews('a', 'r')[0]['author'] == 'expert'
    with pytest.raises(ValueError):
        restore(target, archive)


def test_upload_name_collisions_do_not_overwrite_inputs(tmp_path, monkeypatch):
    from dxaqc import api
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path / 'work')
    image = write_dicom(tmp_path / 'a.dcm', synthetic_spine(4))
    with TestClient(api.app) as client:
        response = client.post('/api/v1/batch?wait=true', files=[
            ('files', ('a.dcm', image.read_bytes())),
            ('files', ('a.dcm', image.read_bytes())),
            ('files', ('00001_a.dcm', image.read_bytes()))])
        assert response.json()['summary']['files'] == 3
        mixed = client.post('/api/v1/batch', files=[('files', ('a.zip', b'zip')), ('files', ('a.dcm', image.read_bytes()))])
        assert mixed.status_code == 422


def test_annotated_report_is_portable_versioned_and_escaped(tmp_path, monkeypatch, trusted_synthetic_router):
    from dxaqc import api
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path / 'work')
    image = write_dicom(tmp_path / 'a.dcm', synthetic_spine(4))
    with TestClient(api.app) as client:
        job = client.post('/api/v1/batch?wait=true', files={'files': ('a.dcm', image.read_bytes())}).json()
        base = '/api/v1/jobs/' + job['id']
        row = client.get(base + '/rows').json()[0]
        endpoint = base + '/images/' + row['row_id'] + '/reviews'
        document = {'expected_revision': 0, 'author': 'Первый', 'status': 'draft', 'quality_class': 0,
                    'geometry': [{'name': 'note', 'kind': 'point', 'points': [{'x': .5, 'y': .5}], 'note': '<script>alert(1)</script>'}],
                    'followups': [{'id': 'a', 'kind': 'second_opinion', 'question': 'Проверить область', 'state': 'open', 'resolution': ''}]}
        assert client.post(endpoint, json=document).status_code == 200
        queue = client.get(base + '/worklist').json()[0]
        assert queue['open_actions'] == '1' and queue['second_opinion_requested'] == '1'
        report = client.get(endpoint + '/1/report.html')
        assert report.status_code == 200
        assert 'data:image/png;base64,' in report.text
        assert '<script>' not in report.text and '&lt;script&gt;' in report.text
        closed = {**document, 'expected_revision': 1, 'author': 'Второй', 'followups': [dict(document['followups'][0], state='done')]}
        assert client.post(endpoint, json=closed).status_code == 422
        closed['followups'][0]['resolution'] = 'Область проверена'
        assert client.post(endpoint, json=closed).status_code == 200
        assert client.get(base + '/worklist').json()[0]['open_actions'] == '0'
        assert client.get(endpoint + '/1/report.html').content == report.content
        assert 'Область проверена' in client.get(endpoint + '/2/report.html').text
        assert client.get(endpoint + '/999/report.html').status_code == 404
        study = client.get(base + '/study-report.html', params={'study_uid': row['study_uid']})
        assert study.status_code == 200 and 'data:image/png;base64,' in study.text
        assert 'версия 2' in study.text and '<script>' not in study.text
        assert client.get(base + '/study-report.html', params={'study_uid': 'missing'}).status_code == 404


def test_readiness_fails_when_worker_has_stopped(tmp_path, monkeypatch):
    import threading
    from dxaqc import api
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path / 'work')
    monkeypatch.setattr(api.app.state, 'worker', threading.Thread(), raising=False)
    with TestClient(api.app) as client:
        assert client.get('/api/v1/ready').status_code == 200
        live_worker = api.app.state.worker
        api.app.state.worker = threading.Thread()
        try:
            response = client.get('/api/v1/ready')
            assert response.status_code == 503
            assert response.json()['ready'] is False
        finally:
            api.app.state.worker = live_worker
