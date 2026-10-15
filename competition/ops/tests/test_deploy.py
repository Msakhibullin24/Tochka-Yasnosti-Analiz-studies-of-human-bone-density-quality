from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deploy as ops


def harness(monkeypatch, tmp_path, failure=None):
    owner = 'test-owner'
    events = []
    containers = {'service': {'Config': {'Labels': {ops.LABEL:owner}}, 'State': {'Running':True}, 'Image':'old-id'}}
    image = {'Config': {'Env': []}, 'Id': 'sha256:tested-id'}
    def inspect(name, image=False):
        return {'Config': {'Env': []}, 'Id': 'sha256:tested-id'} if image else containers.get(name)
    def start(args, image_id, name, work, owner, canary=False):
        events.append(('start', name, image_id, canary))
        containers[name] = {'Config': {'Labels': {ops.LABEL:owner}}, 'State': {'Running':True}, 'Image':image_id}
    def docker(*args, **kwargs):
        events.append(args)
        if args[0] == 'exec' and 'verify' in args and failure == 'canary':
            raise RuntimeError('Bad canary')
        if args[0] == 'rm': containers.pop(args[-1], None)
        if args[0] == 'rename': containers[args[2]] = containers.pop(args[1])
        return subprocess.CompletedProcess(args, 0, stdout='{"ready":true}\n', stderr='')
    monkeypatch.setattr(ops, 'inspect', inspect)
    monkeypatch.setattr(ops, 'start', start)
    monkeypatch.setattr(ops, 'docker', docker)
    monkeypatch.setattr(ops, 'ready', lambda *a, **kw: {})
    monkeypatch.setattr(ops, 'idle', lambda *a: None)
    def smoke(*args):
        if failure == 'post': raise RuntimeError('Bad post deploy')
        return {'ready':True}
    monkeypatch.setattr(ops, 'smoke', smoke)
    args = Namespace(image='mutable:tag', mode='cpu', name='service', port=8093, profile='profile', timeout=1)
    return args, tmp_path, owner, events, containers


def test_failed_canary_never_stops_existing_service(monkeypatch, tmp_path):
    args,state,owner,events,containers = harness(monkeypatch,tmp_path,'canary')
    with pytest.raises(RuntimeError,match='Bad canary'):
        ops.deploy(args,state,owner)
    assert containers['service']['Image'] == 'old-id'
    assert not any(e[0] == 'stop' for e in events)
    assert json.loads(next((state/'runs').glob('*/deployment.json')).read_text())['status'] == 'canary_failed'


def test_failed_post_deploy_restarts_previous_container(monkeypatch, tmp_path):
    args,state,owner,events,containers = harness(monkeypatch,tmp_path,'post')
    with pytest.raises(RuntimeError,match='Bad post deploy'):
        ops.deploy(args,state,owner)
    assert containers['service']['Image'] == 'old-id'
    assert ('start','service') in events
    assert json.loads(next((state/'runs').glob('*/deployment.json')).read_text())['rollback_ready'] is True


def test_success_pins_content_id_and_keeps_previous_for_rollback(monkeypatch, tmp_path):
    args,state,owner,events,containers = harness(monkeypatch,tmp_path)
    record = ops.deploy(args,state,owner)
    assert record['status'] == 'ready'
    assert containers[record['previous_container']]['Image'] == 'old-id'
    assert all(e[2] == 'sha256:tested-id' for e in events if e[0] == 'start' and len(e)==4)
    restored = ops.rollback(args,state,owner)
    assert restored['status'] == 'rolled_back_ready'
    assert containers['service']['Image'] == 'old-id'


@pytest.mark.parametrize('labels', [None, {}, {ops.LABEL: 'someone-else'}])
def test_foreign_container_is_never_changed(monkeypatch,tmp_path,labels):
    args,state,owner,events,containers = harness(monkeypatch,tmp_path)
    containers['service']['Config']['Labels'] = labels
    with pytest.raises(RuntimeError,match='not owned'):
        ops.deploy(args,state,owner)
    assert events == []


def test_cpu_release_cannot_replace_gpu_deployment(monkeypatch,tmp_path):
    args,state,owner,events,containers = harness(monkeypatch,tmp_path)
    (state/'current.json').write_text(json.dumps({'mode':'gpu','port':8093,'profile':'profile'}))
    with pytest.raises(RuntimeError,match='mode change'):
        ops.deploy(args,state,owner)
    assert events == []


@pytest.mark.parametrize('health,restart', [('healthy',False),('starting',False),('unhealthy',True)])
def test_watch_only_restarts_confirmed_unhealthy(monkeypatch,tmp_path,health,restart):
    args,state,owner,events,containers = harness(monkeypatch,tmp_path)
    containers['service']['State']['Health'] = {'Status':health}
    (state/'current.json').write_text(json.dumps({'port':8093,'profile':'profile'}))
    monkeypatch.setattr(ops,'ready',lambda *a,**kw:{'ready':True})
    result = ops.watch(args,state,owner)
    assert result['restarted_unhealthy'] is restart
    assert any(e[0]=='restart' for e in events) is restart


def test_ready_wrong_profile_fails_immediately(monkeypatch):
    import http_smoke
    monkeypatch.setattr(http_smoke,'request',lambda *a,**kw:b'{"ready":true,"workflow_profile_id":"wrong"}')
    monkeypatch.setattr(http_smoke.time,'sleep',lambda *a:pytest.fail('wrong ready profile must not retry'))
    with pytest.raises(RuntimeError,match='wrong profile'):
        http_smoke.ready('http://127.0.0.1:8000','expected',300)
