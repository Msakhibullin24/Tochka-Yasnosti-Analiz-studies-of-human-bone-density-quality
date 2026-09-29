"""Exercise the pinned HTTP workflow and validate its batch exports offline."""
import argparse
import json
import time
import warnings
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from dxaqc import api
from dxaqc.validate_results import validate


def verify(input_path, output, repeat=None, base_url=None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = [p.relative_to(input_path).as_posix() for p in input_path.rglob('*.dcm')]
    if not manifest:
        raise ValueError('No DICOM input manifest')
    transport = httpx.Client(base_url=base_url, timeout=60) if base_url else TestClient(api.app)
    with transport as client:
        ready = client.get('/api/v1/ready')
        if ready.status_code != 200 or not ready.json().get('workflow_profile_id'):
            raise RuntimeError('Pinned workflow is not ready: '+ready.text)
        profile_id = ready.json()['workflow_profile_id']
        submitted = client.post('/api/v1/batch/path', json={'input_path': str(input_path)})
        submitted.raise_for_status(); job = submitted.json(); job_id = job['id']
        shown = 0; deadline = time.monotonic()+1800
        while job['status'] in ('queued','running'):
            if time.monotonic() > deadline:
                raise TimeoutError('API batch verification exceeded 30 minutes')
            time.sleep(.5)
            response = client.get(f'/api/v1/jobs/{job_id}'); response.raise_for_status(); job = response.json()
            if job.get('done',0) >= shown+25:
                shown=job['done'];print(f'API {shown}/{len(manifest)}',flush=True)
        if job['status'] != 'finished':
            raise RuntimeError('API job failed: '+json.dumps(job))
        for name in ('results_extended.csv','submission.csv','timing.json','summary.json','additional_series.zip'):
            response = client.get(f'/api/v1/jobs/{job_id}/{name}');response.raise_for_status()
            (output/name).write_bytes(response.content)
        summary=json.loads((output/'summary.json').read_text())
        if summary.get('workflow_profile_id') != profile_id:
            raise ValueError('Batch used a different workflow profile')
        strict=validate(output/'submission.csv',repeat=repeat,competition=True,
                        timing=output/'timing.json',series=output/'additional_series.zip')
        complete=validate(output/'results_extended.csv',manifest=manifest,timing=output/'timing.json')
        if not strict['valid'] or not complete['valid']:
            raise ValueError(json.dumps({'strict':strict,'manifest':complete}))
        report={'profile_id':profile_id,'summary':summary,'strict_repeat_sc_sr':strict,
                'input_manifest':complete,'api_ready':True,'clinical_validation':False,
                'transport': 'http_tcp' if base_url else 'in_process_asgi'}
        (output/'api_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--repeat',type=Path)
    parser.add_argument('--url', help='Optional running HTTP API; input path must exist on the server too')
    args=parser.parse_args()
    warnings.filterwarnings('ignore',message='Invalid value for VR UI:.*')
    warnings.filterwarnings('ignore',message='TypedStorage is deprecated.*')
    verify(args.input,args.output,args.repeat,args.url)
