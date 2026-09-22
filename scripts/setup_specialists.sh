#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
command -v uv >/dev/null || { echo 'Install uv before setting up the research environment.' >&2; exit 1; }
python_bin="${OSSEO_PYTHON:-$project_dir/backend/.venv/bin/python}"
"$python_bin" backend/scripts/fetch_specialists.py
if [[ ! -f data/specialists/venv/pyvenv.cfg ]]; then
  uv venv --python "$python_bin" data/specialists/venv
fi
uv pip sync --python data/specialists/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  competition/requirements-specialists-lock.txt
HF_HUB_OFFLINE=1 data/specialists/venv/bin/python competition/check_specialists.py
data/specialists/venv/bin/python competition/audit_nhanes.py
data/specialists/venv/bin/python -m pytest -q competition/tests/test_specialist_qc.py competition/tests/test_training_reports.py competition/tests/test_specialist_upgrade.py
