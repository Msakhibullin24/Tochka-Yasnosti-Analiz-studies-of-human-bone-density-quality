import json
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'run.sh'


@pytest.mark.parametrize('mode', ['submission', 'complete', 'report'])
def test_batch_shell_forwards_acceptance_and_keeps_offline_execution(tmp_path, mode):
    executable = tmp_path / 'docker'
    executable.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                          'with open(os.environ["CAPTURE"], "a") as stream:\n'
                          '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                          'sys.exit(3 if sys.argv[1] == "run" and "complete" in sys.argv else 0)\n')
    executable.chmod(0o755)
    capture = tmp_path / 'commands.jsonl'
    source = tmp_path / 'source.zip'
    source.touch()
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'CAPTURE': str(capture)}
    result = subprocess.run(['sh', str(SCRIPT), 'batch', str(source), str(tmp_path / 'out'), mode],
                            env=env, capture_output=True, text=True)
    assert result.returncode == (3 if mode == 'complete' else 0)
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert commands[-1][-2:] == ['--acceptance', mode]
    assert commands[-1][commands[-1].index('--network') + 1] == 'none'
    assert 'osseo-dxaqc:1.12.0' in commands[-1]


def test_invalid_acceptance_is_rejected_before_creating_output(tmp_path):
    output = tmp_path / 'out'
    result = subprocess.run(['sh', str(SCRIPT), 'batch', str(tmp_path / 'absent.zip'),
                             str(output), 'invalid'], capture_output=True, text=True)
    assert result.returncode == 64 and 'invalid acceptance mode' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('rootless', [False, True])
def test_bind_mount_user_matches_rootless_namespace_mapping(tmp_path, rootless):
    executable = tmp_path / 'docker'
    executable.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                          'if sys.argv[1] == "info":\n'
                          '    print(json.dumps(["name=rootless"] if os.environ["ROOTLESS_TEST"] == "1" else []))\n'
                          'if sys.argv[1] == "run":\n'
                          '    open(os.environ["CAPTURE"], "w").write(json.dumps(sys.argv[1:]))\n')
    executable.chmod(0o755)
    capture = tmp_path / 'commands.json'
    source = tmp_path / 'source.zip'
    source.touch()
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'CAPTURE': str(capture), 'ROOTLESS_TEST': '1' if rootless else '0'}
    env.pop('DXAQC_CONTAINER_USER', None)
    result = subprocess.run(['sh', str(SCRIPT), 'batch', str(source), str(tmp_path / 'out')], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[args.index('--user') + 1] == ('0:0' if rootless else f'{os.getuid()}:{os.getgid()}')
