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
    assert 'osseo-dxaqc:1.11.6' in commands[-1]


def test_invalid_acceptance_is_rejected_before_creating_output(tmp_path):
    output = tmp_path / 'out'
    result = subprocess.run(['sh', str(SCRIPT), 'batch', str(tmp_path / 'absent.zip'),
                             str(output), 'invalid'], capture_output=True, text=True)
    assert result.returncode == 64 and 'invalid acceptance mode' in result.stderr
    assert not output.exists()
