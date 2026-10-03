#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
gpu_env="${DXA_GPU_ENV:-$repo_root/.venv-gpu}"
if [[ -d "$gpu_env" ]]; then
  echo "Environment already exists: $gpu_env. Choose a new DXA_GPU_ENV to avoid altering it." >&2
  exit 2
fi
command -v uv >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi
uv venv "$gpu_env" --python 3.11
uv pip install --python "$gpu_env/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$gpu_env/bin/python" -r competition/gpu_research/requirements.txt
mkdir -p data/gpu-environment
uv pip freeze --python "$gpu_env/bin/python" > data/gpu-environment/requirements-resolved.txt
"$gpu_env/bin/python" competition/gpu_research/run.py doctor --output data/gpu-environment/doctor.json
