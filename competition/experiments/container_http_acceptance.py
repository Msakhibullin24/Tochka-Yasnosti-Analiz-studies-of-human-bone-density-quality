"""Exercise real HTTP/ZIP/export inside a network-isolated inference container."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import socket
from time import monotonic, sleep
import urllib.error
import urllib.request
import uuid
import warnings
from zipfile import ZipFile

import pydicom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000')
    parser.add_argument('--input-archive', type=Path, required=True)
    parser.add_argument('--comparison-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-images', type=int, default=499)
    parser.add_argument('--scope', default='Frozen workflow 1.11.6; not experimental anatomical QC heads or H200 timing')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    interfaces = [name for _, name in socket.if_nameindex()]
    if set(interfaces) != {'lo'}:
        raise ValueError('Run this acceptance test inside a --network none container')
    warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')

    def request(path, data=None, headers=None, timeout=30):
        req = urllib.request.Request(args.endpoint + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            assert response.status == 200
            return response.read()

    ready = None
    deadline = monotonic() + 60
    while monotonic() < deadline:
        try:
            ready = json.loads(request('/api/v1/ready', timeout=5))
            break
        except (OSError, urllib.error.URLError):
            sleep(.2)
    if ready is None:
        raise TimeoutError('HTTP server was not ready')
    ui = request('/').decode()
    assets = sorted(set(re.findall(r'(?:src|href)="(/assets/[^\"]+)"', ui)))
    if not assets:
        raise ValueError('Compiled UI assets are absent')
    for asset in assets:
        assert request(asset)
    archive = args.input_archive.read_bytes()
    boundary = 'dxaqc-acceptance-' + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="input.zip"\r\n'
            'Content-Type: application/zip\r\n\r\n').encode() + archive + f'\r\n--{boundary}--\r\n'.encode()
    started = monotonic()
    job = json.loads(request('/api/v1/batch', body, {'Content-Type': f'multipart/form-data; boundary={boundary}'}))
    deadline = monotonic() + 600
    while job['status'] in ('queued', 'running') and monotonic() < deadline:
        sleep(.2)
        job = json.loads(request('/api/v1/jobs/' + job['id']))
    assert job['status'] == 'finished', job
    summary = job['summary']
    assert summary['files'] == summary['success'] == args.expected_images and summary['failure'] == 0, summary
    assert summary['submission_valid'] is True
    args.output.mkdir(parents=True)
    paths = ('submission.csv', 'submission.xlsx', 'results_extended.csv', 'additional_series.zip',
             'submission_validation.json', 'requirements.json', 'timing.json')
    hashes = {}
    for name in paths:
        data = request(f'/api/v1/jobs/{job["id"]}/{name}')
        (args.output / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    with ZipFile(BytesIO((args.output / 'submission.xlsx').read_bytes())) as workbook:
        assert 'xl/workbook.xml' in workbook.namelist()
    timing = json.loads((args.output / 'timing.json').read_text())
    assert timing['within_180_seconds'] is True
    # Paths differ between folder input and ZIP wrappers. Technical SOP identity
    # and model answers must nevertheless agree, including repeated SOP IDs.
    def answers(data):
        rows = csv.DictReader(StringIO(data.decode('utf-8-sig')))
        return Counter((r['image_uid'], r['quality_class'], r['violation_type'], r['quality_prob'], r['processing_status']) for r in rows)
    assert answers((args.output / 'results_extended.csv').read_bytes()) == answers(args.comparison_csv.read_bytes())
    classes, png = Counter(), 0
    with ZipFile(args.output / 'additional_series.zip') as series:
        assert series.testzip() is None
        for name in series.namelist():
            if name.endswith('.png'):
                png += 1
            elif name.endswith('.dcm'):
                d = pydicom.dcmread(BytesIO(series.read(name)), stop_before_pixels=True)
                classes[str(d.SOPClassUID)] += 1
    assert png == args.expected_images
    assert classes['1.2.840.10008.5.1.4.1.1.7'] == args.expected_images
    assert sum(classes.values()) == 2 * args.expected_images
    retained = request(f'/api/v1/jobs/{job["id"]}/source.zip')
    assert hashlib.sha256(retained).digest() == hashlib.sha256(archive).digest()
    result = {'ready_response': ready, 'files': summary['files'], 'success': summary['success'], 'failure': summary['failure'],
              'submission_valid': summary['submission_valid'], 'requirements_complete': summary['requirements_complete'],
              'network_interfaces': interfaces, 'network_mode': 'none', 'transport': 'real_loopback_HTTP_multipart_ZIP',
              'compiled_ui_assets_served': len(assets), 'cli_answers_identical': True,
              'original_zip_retained_identically': True, 'export_sha256': hashes,
              'png_explanations': png, 'dicom_sop_class_counts': dict(classes),
              'batch_seconds': timing['batch_seconds'], 'max_study_upper_bound_seconds': timing['max_study_upper_bound_seconds'],
              'within_180_seconds': timing['within_180_seconds'], 'wall_seconds_including_upload_exports': monotonic() - started,
              'clinical_validation': False, 'scope': args.scope}
    (args.output / 'http-acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
