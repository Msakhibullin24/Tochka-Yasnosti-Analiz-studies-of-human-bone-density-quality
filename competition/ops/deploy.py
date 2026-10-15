#!/usr/bin/env python3
"""Linux Docker deployment with isolated acceptance, ownership and automatic rollback.

Uses Python's standard library on the host; no Docker socket chmod or model download.
Updates have a brief maintenance window. Pause uploads before updating.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from http_smoke import ready, request, smoke

OPS = Path(__file__).resolve().parent
LABEL = 'io.dxaqc.deployment'
HEALTH = 'python -c "import urllib.request; urllib.request.urlopen(\'http://127.0.0.1:8000/api/v1/ready\',timeout=5)"'


def docker(*args, check=True):
    result = subprocess.run(['docker', *map(str, args)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f'Docker {args[0]} failed: {result.stderr.strip()}')
    return result


def connect():
    if docker('info', check=False).returncode:
        if os.environ.get('DOCKER_HOST') or os.environ.get('DOCKER_CONTEXT'):
            raise RuntimeError('Explicit Docker endpoint is unavailable')
        state = OPS.parents[1]/'data/build-rootless/runtime-path.txt'
        if not state.is_file():
            raise RuntimeError('Docker is unavailable; start the Docker service')
        os.environ['DOCKER_HOST'] = 'unix://'+state.read_text().strip()+'/docker.sock'
        docker('info')


def inspect(name, image=False):
    value = docker('image' if image else 'container', 'inspect', name, check=False)
    return json.loads(value.stdout)[0] if value.returncode == 0 else None


def gpu_args(mode):
    if mode == 'cpu':
        return []
    info = json.loads(docker('info', '--format', '{{json .Runtimes}}').stdout)
    if 'nvidia' in info:
        return ['--gpus', 'all']
    result = []
    for name in ('nvidia0', 'nvidiactl', 'nvidia-uvm'):
        path = Path('/dev')/name
        if not path.is_char_device():
            raise RuntimeError(f'GPU device missing: {path}; CPU fallback is not allowed')
        result += ['--device', str(path)]
    libraries = subprocess.check_output(['ldconfig', '-p'], text=True)
    for soname in ('libcuda.so.1', 'libnvidia-ptxjitcompiler.so.1', 'libnvidia-nvvm.so.4'):
        match = re.search(r'^\s*'+re.escape(soname)+r' .* => (.+)$', libraries, re.M)
        if not match or not Path(match[1]).is_file():
            raise RuntimeError(f'Missing GPU driver library: {soname}')
        result += ['-v', f'{match[1]}:/usr/local/nvidia/lib64/{soname}:ro']
    match = re.search(r'^\s*(libnvidia-gpucomp\.so\.\S+) .* => (.+)$', libraries, re.M)
    if match:
        result += ['-v', f'{match[2]}:/usr/local/nvidia/lib64/{match[1]}:ro']
    return result + ['-e', 'LD_LIBRARY_PATH=/usr/local/nvidia/lib64']


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def ensure_owned(info, owner):
    if info and (info['Config'].get('Labels') or {}).get(LABEL) != owner:
        raise RuntimeError('Container is not owned by this deployment; choose another --name/--state')


def idle(endpoint):
    offset = 0
    while True:
        jobs = json.loads(request(endpoint, f'/api/v1/jobs?limit=200&offset={offset}'))
        if any(j['status'] in ('queued', 'running') for j in jobs):
            raise RuntimeError('Jobs are active; wait for completion and pause uploads before updating')
        if len(jobs) < 200:
            break
        offset += len(jobs)


def start(args, image_id, name, work, owner, canary=False):
    work.mkdir(parents=True, exist_ok=True)
    security = docker('info', '--format', '{{json .SecurityOptions}}').stdout
    user = '0:0' if 'rootless' in security else f'{os.getuid()}:{os.getgid()}'
    flags = ['run', '-d', '--name', name, '--label', LABEL+'='+owner,
             '--user', user, '--read-only', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=2g',
             '--stop-timeout', '180', '--log-opt', 'max-size=10m', '--log-opt', 'max-file=3',
             '--health-cmd', HEALTH, '--health-interval', '30s', '--health-timeout', '10s',
             '--health-start-period', '180s', '--health-retries', '3',
             '-v', f'{work}:/work', '-v', f'{OPS}:/ops:ro']
    if canary:
        flags += ['--network', 'none']
    else:
        # Rootless port publishing requires a regular bridge (an internal bridge
        # has no default route and RootlessKit cannot forward its published port).
        network = 'dxaqc-managed-'+owner[:12]+'-api'
        if docker('network', 'inspect', network, check=False).returncode:
            docker('network', 'create', network)
        flags += ['--network', network, '-p', f'127.0.0.1:{args.port}:8000', '--restart', 'unless-stopped']
    flags += gpu_args(args.mode)
    docker(*flags, image_id)


def deploy(args, state, owner):
    image = inspect(args.image, image=True)
    if image is None:
        raise RuntimeError('Image is not installed. Load the offline archive or docker pull the reviewed digest first')
    env = image['Config'].get('Env', [])
    has_gpu = any(e.startswith('DXAQC_ANATOMY_CONFIG=') and e.split('=',1)[1] for e in env)
    if has_gpu != (args.mode == 'gpu'):
        raise RuntimeError('Image does not match selected CPU/GPU mode')
    current = inspect(args.name)
    ensure_owned(current, owner)
    prior_record = json.loads((state/'current.json').read_text()) if (state/'current.json').is_file() else None
    if prior_record and prior_record['mode'] != args.mode:
        raise RuntimeError('CPU/GPU mode change requires a separate deployment state and name')
    if prior_record and prior_record['port'] != args.port:
        raise RuntimeError('Port change requires a separate deployment state and name')
    old_profile = prior_record['profile'] if prior_record else args.profile
    endpoint = f'http://127.0.0.1:{args.port}'
    # Refuse port collisions before model initialization, without changing existing containers.
    if not current or not current['State']['Running']:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
    run_id = time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    evidence = state/'runs'/run_id
    evidence.mkdir(parents=True)
    canary = args.name+'-check-'+uuid.uuid4().hex[:8]
    previous = None
    record = {'run_id': run_id, 'image_id': image['Id'], 'image_requested': args.image,
              'mode': args.mode, 'profile': args.profile, 'port': args.port, 'name': args.name,
              'owner': owner, 'previous_container': None, 'status': 'checking'}
    if prior_record:
        record['previous_release'] = {k: prior_record[k] for k in ('image_id','image_requested','mode','profile','port')}
    save(evidence/'deployment.json', record)
    try:
        if args.mode == 'gpu':
            checked = docker('run', '--rm', '--network', 'none', '--read-only',
                             '-v', f'{OPS}:/ops:ro', image['Id'], 'python', '/ops/verify_assets.py',
                             '--root', '/opt/anatomy-assets', '--code-root', '/app')
            record['model_approval'] = json.loads(checked.stdout)
        start(args, image['Id'], canary, evidence/'canary', owner, canary=True)
        docker('exec', canary, 'python', '/ops/http_smoke.py', 'fixture', '/work/smoke.zip')
        result = docker('exec', canary, 'python', '/ops/http_smoke.py', 'verify',
                        '--url', 'http://127.0.0.1:8000', '--input', '/work/smoke.zip',
                        '--profile', args.profile, '--timeout', args.timeout,
                        '--output', '/work/acceptance.json')
        record['canary'] = json.loads(result.stdout.strip().splitlines()[-1])
    except BaseException as exc:
        record.update(status='canary_failed', error=str(exc))
        save(evidence/'deployment.json', record)
        raise
    finally:
        if inspect(canary):
            (evidence/'canary.log').write_text(docker('logs', canary, check=False).stderr)
            docker('rm', '-f', canary)
    if current:
        if current['State']['Running']:
            ready(endpoint, old_profile, args.timeout)
            idle(endpoint)
        previous = args.name+'-previous-'+run_id.lower()
        docker('stop', '-t', '180', args.name)
        docker('rename', args.name, previous)
        record['previous_container'] = previous
        record['status'] = 'switching'
        save(evidence/'deployment.json', record)
    try:
        database = state/'work/dxaqc-jobs/jobs.sqlite3'
        if database.is_file():
            with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as source:
                with sqlite3.connect(evidence/'jobs-before-update.sqlite3') as target:
                    source.backup(target)
        start(args, image['Id'], args.name, state/'work', owner)
        record['post_deploy'] = smoke(endpoint, evidence/'canary/smoke.zip', args.profile, args.timeout)
        record['status'] = 'ready'
        save(evidence/'deployment.json', record)
        save(state/'current.json', record)
    except BaseException as exc:
        record.update(status='post_deploy_failed', error=str(exc))
        failed = inspect(args.name)
        ensure_owned(failed, owner)
        if failed:
            (evidence/'failed.log').write_text(docker('logs', args.name, check=False).stderr)
            docker('rm', '-f', args.name)
        if previous:
            docker('rename', previous, args.name)
            docker('start', args.name)
            ready(endpoint, old_profile, args.timeout)
            record['rollback_ready'] = True
        save(evidence/'deployment.json', record)
        raise
    return record


def rollback(args, state, owner):
    record = json.loads((state/'current.json').read_text())
    previous = record.get('previous_container')
    if not previous:
        raise RuntimeError('No retained previous container')
    old, current = inspect(previous), inspect(args.name)
    if not old:
        raise RuntimeError('Previous container no longer exists')
    ensure_owned(old, owner); ensure_owned(current, owner)
    endpoint = f'http://127.0.0.1:{record["port"]}'
    if current and current['State']['Running']:
        idle(endpoint)
    retired = args.name+'-retired-'+uuid.uuid4().hex[:8]
    docker('stop', '-t', '180', args.name)
    docker('rename', args.name, retired)
    try:
        docker('rename', previous, args.name); docker('start', args.name)
        ready(endpoint, record.get('previous_release',record)['profile'], args.timeout)
    except BaseException:
        docker('stop', args.name, check=False); docker('rename', args.name, previous, check=False)
        docker('rename', retired, args.name); docker('start', args.name)
        raise
    previous_release = record.get('previous_release', {})
    record['previous_release'] = {k: record[k] for k in ('image_id','image_requested','mode','profile','port')}
    record.update(previous_release)
    record.update(image_id=old['Image'], previous_container=retired, status='rolled_back_ready')
    save(state/'current.json', record)
    return record


def watch(args, state, owner):
    """Only restart an owned, running, Docker-unhealthy container, never starting/stopped."""
    record = json.loads((state/'current.json').read_text())
    info = inspect(args.name)
    ensure_owned(info, owner)
    if not info or not info['State']['Running']:
        raise RuntimeError('Service is stopped; watchdog will not override an intentional stop')
    health = info['State'].get('Health', {}).get('Status')
    if health == 'unhealthy':
        docker('restart', '-t', '180', args.name)
    status = ready(f'http://127.0.0.1:{record["port"]}', record['profile'], args.timeout)
    result = {'ready': status['ready'], 'image_id': info['Image'],
              'restarted_unhealthy': health == 'unhealthy', 'checked_at': time.time()}
    save(state/'last-watch.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('deploy', 'status', 'rollback', 'watch'))
    p.add_argument('--image')
    p.add_argument('--mode', choices=('cpu','gpu'), default='cpu')
    p.add_argument('--name', default='dxaqc-managed')
    p.add_argument('--state', type=Path, default=OPS.parents[1]/'data/deployment')
    p.add_argument('--port', type=int, default=8093)
    p.add_argument('--profile', help='Expected runtime profile; defaults to the reviewed CPU/GPU profile')
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()
    a.profile = a.profile or ('research-live-anatomy-qc-20261015' if a.mode == 'gpu' else 'tz-delivery-1.12.0')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{1,80}', a.name) or not 1 <= a.port <= 65535 or a.timeout < 1:
        p.error('Invalid container name, port or timeout')
    if a.command == 'deploy' and not a.image:
        p.error('--image is required for deploy')
    connect()
    state = a.state.resolve(); state.mkdir(parents=True, exist_ok=True)
    owner = hashlib.sha256(str(state).encode()).hexdigest()
    with (state/'deploy.lock').open('a') as lock:
        if a.command != 'status':
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another deployment or watchdog operation is active for this state') from None
        if a.command == 'deploy': result = deploy(a, state, owner)
        elif a.command == 'rollback': result = rollback(a, state, owner)
        elif a.command == 'watch': result = watch(a, state, owner)
        else:
            info = inspect(a.name); ensure_owned(info, owner)
            result = {'container': a.name, 'running': bool(info and info['State']['Running']),
                      'docker_health': info['State'].get('Health',{}).get('Status') if info else None,
                      'release': json.loads((state/'current.json').read_text()) if (state/'current.json').is_file() else None}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    def terminate(signum, frame):
        raise InterruptedError(f'Deployment interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
