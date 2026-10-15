"""Real HTTP acceptance; synthetic pixels test engineering, never model accuracy."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import uuid
import zipfile


def fixture(path):
    import cv2
    import numpy as np
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    image = np.zeros((300, 300), np.uint8)
    cv2.rectangle(image, (122, 0), (178, 299), 170, -1)
    cv2.circle(image, (25, 290), 45, 140, -1)
    cv2.circle(image, (275, 290), 45, 140, -1)
    image = np.clip(cv2.GaussianBlur(image, (0, 0), 3).astype(np.int16)
                    + np.random.default_rng(0).integers(0, 8, image.shape), 0, 255).astype(np.uint8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.Manufacturer = 'CR', 'GE Healthcare'
    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = image.tobytes()
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('synthetic.dcm', stream.getvalue())


def request(endpoint, path, data=None, headers=None, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(endpoint+path, data=data,
                                headers=headers or {}), timeout=timeout) as response:
        return response.read()


def ready(endpoint, expected_profile, timeout):
    deadline = time.monotonic()+timeout
    error = None
    while time.monotonic() < deadline:
        try:
            value = json.loads(request(endpoint, '/api/v1/ready', timeout=min(5, max(.1, deadline-time.monotonic()))))
            if value.get('ready') is True and value.get('workflow_profile_id') == expected_profile:
                return value
            if value.get('ready') is True:
                raise RuntimeError(f'Ready service has wrong profile: {value.get("workflow_profile_id")}, expected {expected_profile}')
            error = f'Unexpected readiness/profile: {value.get("workflow_profile_id")}'
        except (OSError, ValueError) as exc:
            error = str(exc)
        time.sleep(.5)
    raise TimeoutError(f'Service not ready within {timeout}s: {error}')


def smoke(endpoint, archive, expected_profile, timeout=300, minimum_success=1):
    status = ready(endpoint, expected_profile, timeout)
    html = request(endpoint, '/').decode()
    assets = set(re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html))
    if not assets or any(not request(endpoint, asset) for asset in assets):
        raise ValueError('Compiled UI is missing')
    content = Path(archive).read_bytes()
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        count = sum(n.lower().endswith('.dcm') for n in source.namelist())
    if not count:
        raise ValueError('No DICOM fixtures in ZIP')
    boundary = 'smoke-'+uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="smoke.zip"\r\n'
            'Content-Type: application/zip\r\n\r\n').encode()+content+f'\r\n--{boundary}--\r\n'.encode()
    job = json.loads(request(endpoint, '/api/v1/batch', body,
                     {'Content-Type': 'multipart/form-data; boundary='+boundary}, timeout=60))
    deadline = time.monotonic()+timeout
    while job['status'] in ('queued', 'running') and time.monotonic() < deadline:
        time.sleep(.5)
        job = json.loads(request(endpoint, '/api/v1/jobs/'+job['id']))
    if job['status'] != 'finished':
        raise RuntimeError(f'Batch did not finish: {job["status"]}')
    summary = job['summary']
    if summary['files'] != count or summary['success'] < minimum_success or summary['failure'] != count-summary['success']:
        raise ValueError(f'Incomplete/unsuccessful inference: {summary}')
    if not summary.get('submission_valid'):
        raise ValueError('Submission contract invalid')
    prefix = '/api/v1/jobs/'+job['id']+'/'
    exports = {name: request(endpoint, prefix+name) for name in
               ('submission.csv', 'submission.xlsx', 'results_extended.csv', 'timing.json', 'additional_series.zip')}
    rows = list(csv.DictReader(io.StringIO(exports['submission.csv'].decode('utf-8-sig'))))
    if len(rows) != count:
        raise ValueError('CSV row loss')
    extended = list(csv.DictReader(io.StringIO(exports['results_extended.csv'].decode('utf-8-sig'))))
    if len(extended) != count or any(r['processing_status'] not in ('Success', 'Failure') for r in extended):
        raise ValueError('Invalid processing status')
    with zipfile.ZipFile(io.BytesIO(exports['submission.xlsx'])) as workbook:
        if workbook.testzip() or 'xl/workbook.xml' not in workbook.namelist():
            raise ValueError('Broken XLSX')
    with zipfile.ZipFile(io.BytesIO(exports['additional_series.zip'])) as series:
        names = series.namelist()
        if series.testzip() or sum(n.endswith('.dcm') for n in names) != 2*summary['success']:
            raise ValueError('Missing SC/SR DICOM series')
        if sum(n.endswith('.png') for n in names) != summary['success']:
            raise ValueError('Missing explanation PNG')
    if hashlib.sha256(request(endpoint, prefix+'source.zip')).digest() != hashlib.sha256(content).digest():
        raise ValueError('Input ZIP changed')
    timing = json.loads(exports['timing.json'])
    if timing.get('within_180_seconds') is not True:
        raise ValueError('Study exceeded time limit')
    ready(endpoint, expected_profile, 30)
    return {'ready': True, 'version': status['version'], 'profile': expected_profile,
            'files': count, 'success': summary['success'], 'failure': summary['failure'],
            'ui_assets': len(assets), 'exports_verified': list(exports),
            'source_zip_identical': True, 'within_180_seconds': True,
            'export_sha256': {k: hashlib.sha256(v).hexdigest() for k,v in exports.items()},
            'scope': 'Engineering HTTP/ZIP/export check; not accuracy or full TZ acceptance'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fixture'); f.add_argument('path', type=Path)
    s = sub.add_parser('verify')
    s.add_argument('--url', required=True)
    s.add_argument('--input', type=Path, required=True)
    s.add_argument('--profile', default='tz-delivery-1.12.0')
    s.add_argument('--timeout', type=int, default=300)
    s.add_argument('--minimum-success', type=int, default=1)
    s.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'fixture':
        fixture(a.path)
    else:
        result = smoke(a.url, a.input, a.profile, a.timeout, a.minimum_success)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
