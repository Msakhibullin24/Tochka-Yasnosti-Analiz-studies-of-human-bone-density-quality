#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
api_port="${DXA_API_PORT:-8001}"
web_port="${DXA_WEB_PORT:-5173}"
python_bin="$project_dir/backend/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Backend environment is missing. Run: uv venv backend/.venv --python 3.11 && uv pip install --python backend/.venv/bin/python -e 'backend[test]'" >&2
  exit 2
fi

export DXAQC_DATA_DIR="${DXAQC_DATA_DIR:-$project_dir/data/runtime}"
export DXA_ALLOWED_ORIGINS="${DXA_ALLOWED_ORIGINS:-http://localhost:$web_port,http://127.0.0.1:$web_port}"
export ANALYSIS_API_TARGET="http://127.0.0.1:$api_port"

"$python_bin" -m uvicorn dxaqc.api:app --app-dir "$project_dir/competition" --host 127.0.0.1 --port "$api_port" &
api_pid=$!

cleanup() {
  kill "$api_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$project_dir"
npm run dev -- --host 127.0.0.1 --port "$web_port"
