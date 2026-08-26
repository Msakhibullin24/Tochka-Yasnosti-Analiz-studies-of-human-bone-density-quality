#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend_dir="$(cd "$script_dir/.." && pwd)"
venv_python="$backend_dir/.venv/bin/python"
uv_bin="${UV_BIN:-uv}"

if [[ ! -x "$venv_python" ]]; then
  "$uv_bin" venv "$backend_dir/.venv" --python 3.11
fi

"$uv_bin" pip install --python "$venv_python" -e "$backend_dir[test]"
"$uv_bin" pip install --python "$venv_python" pip setuptools wheel
"$uv_bin" pip install --python "$venv_python" \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.1.2 torchvision==0.16.2
"$uv_bin" pip install --python "$venv_python" --no-build-isolation chumpy==0.70
"$uv_bin" pip install --python "$venv_python" -r "$backend_dir/requirements-ml.txt"
"$backend_dir/.venv/bin/mim" install 'mmcv==2.1.0'

"$venv_python" - <<'PY'
import cv2
import mmcv
import mmengine
import mmdet
import mmpose
import numpy
import torch

print({
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "opencv": cv2.__version__,
    "mmcv": mmcv.__version__,
    "mmengine": mmengine.__version__,
    "mmdet": mmdet.__version__,
    "mmpose": mmpose.__version__,
})
PY
