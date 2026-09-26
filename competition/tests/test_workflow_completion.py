import csv
import io
import json
import time
import zipfile

import numpy as np
import pydicom
from fastapi.testclient import TestClient

from conftest import synthetic_spine, write_dicom
from dxaqc.explain import render_overlay, write_secondary_capture


def test_exported_axis_has_physical_angle_and_square_display_pixels(tmp_path):
    image = np.zeros((300, 200), np.uint8)
    overlay = {'axis': [(60, 30), (90, 200)]}
    bgr = render_overlay(image, 'spine', overlay, {}, [], 0, .1, 1.05, .6)
    anatomy = bgr[:1050]
    yy, xx = np.where((anatomy[:,:,1] > 180) & (anatomy[:,:,0] < 130) & (anatomy[:,:,2] < 130))
    angle = np.degrees(np.arctan(np.polyfit(yy, xx, 1)[0]))
    expected = np.degrees(np.arctan(30*.6/(170*1.05)))
    assert abs(angle-expected) < .3
    path = tmp_path/'sc.dcm'
    write_secondary_capture(bgr, '1.2', '1.2.3', path)
    ds = pydicom.dcmread(path)
    assert list(ds.PixelAspectRatio) == [1,1]


def test_original_snapshot_reprocess_and_expert_package(tmp_path, monkeypatch):
    from dxaqc import api
    from dxaqc.report import write_csv
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path/'jobs')
    path = write_dicom(tmp_path/'input.dcm', synthetic_spine())
    data = path.read_bytes()
    with TestClient(api.app) as client:
        job = client.post('/api/v1/batch?wait=true', files={'files':('input.dcm',data,'application/dicom')}).json()
        base = f"/api/v1/jobs/{job['id']}"
        assert job['status']=='finished' and job['originals_available']
        with zipfile.ZipFile(io.BytesIO(client.get(base+'/source.zip').content)) as archive:
            assert archive.read(archive.namelist()[0]) == data
        original = client.get(base+'/results.csv').content
        new = client.post(base+'/reprocess').json()
        assert new['id'] != job['id']
        deadline = time.monotonic()+20
        while new['status'] in ('running','queued') and time.monotonic()<deadline:
            time.sleep(.05)
            new = client.get(f"/api/v1/jobs/{new['id']}").json()
        assert new['status']=='finished' and new['originals_available']
        assert client.get(base+'/results.csv').content == original
        row = client.get(base+'/rows').json()[0]
        review = {'expected_revision':0,'author':'Reviewer','status':'confirmed','quality_class':0,
                  'violations':[],'comment':'Checked','geometry':[]}
        assert client.post(base+f"/images/{row['row_id']}/reviews",json=review).status_code==200
        package = client.get(base+'/review-package.zip')
        assert package.status_code==200
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            manifest=json.loads(archive.read('manifest.json'))
            assert manifest['complete'] and manifest['images'][0]['revision']==1
            report=archive.read(next(n for n in archive.namelist() if n.endswith('.html'))).decode()
            assert 'Reviewer' in report and 'scale(' in report
        review.update(expected_revision=1,status='draft')
        assert client.post(base+f"/images/{row['row_id']}/reviews",json=review).status_code==200
        with zipfile.ZipFile(io.BytesIO(client.get(base+'/review-package.zip').content)) as archive:
            assert not json.loads(archive.read('manifest.json'))['complete']
            assert not any(n.endswith('.html') for n in archive.namelist())
        # A binary flag with no type must never print as a normal machine result.
        row.update(quality_class='1',violation_type='',violation_type_status='undetermined')
        write_csv([row], api.DATA_DIR/job['id']/'out/results.csv')
        html = client.get(base+'/report.html').text
        assert 'тип не установлен' in html and 'Полная анатомическая проверка не подтверждена' in html
        (api.DATA_DIR/job['id']/'source.zip').write_bytes(b'corrupt')
        assert client.post(base+'/reprocess').status_code==409


def test_anatomical_candidates_return_to_original_pixel_coordinates():
    from dxaqc.anatomy import detect_landmarks
    image = np.zeros((100, 150), np.uint8)
    overlay = {'ischium_bottom': [(20., 80.)], 'lesser_trochanter': [(30.,60.)]}
    candidates = detect_landmarks(image, 'hip_left', overlay, 1.05, .6)
    by_name = {r['name']:r for r in candidates['landmarks']}
    assert np.allclose(by_name['ischium']['points'], [[129.,80.]])
    assert np.allclose(by_name['lesser_trochanter']['points'], [[119.,60.]])
    assert not candidates['clinical_validation']


def test_uncertain_region_abstains_before_quality_model(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from dxaqc.pipeline import Analyzer
    from dxaqc.dicom_io import DicomReadError, read_dxa
    from dxaqc import embedding
    import pytest
    monkeypatch.setattr(embedding, 'embed', lambda image: np.zeros(2))
    router = SimpleNamespace(predict_detailed=lambda values: (['spine'], [.4], [1.0]))
    analyser = Analyzer(SimpleNamespace(router=router, groups={}))
    image = read_dxa(write_dicom(tmp_path/'input.dcm', synthetic_spine()))
    with pytest.raises(DicomReadError, match='region is uncertain'):
        analyser.analyze(image)
