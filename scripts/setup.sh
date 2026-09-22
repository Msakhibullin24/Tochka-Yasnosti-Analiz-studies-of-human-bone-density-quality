#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
command -v pnpm >/dev/null || { echo 'Install pnpm first: https://pnpm.io/installation' >&2; exit 1; }

echo 'Installing frontend dependencies with pnpm...'
pnpm install --frozen-lockfile

echo 'Syncing the research/backend environment with uv...'
uv sync --project "$project_dir/backend" --extra test --locked

competition_python="$project_dir/competition/.venv/bin/python"
if [[ ! -x "$competition_python" ]]; then
  echo 'Creating the competition environment with uv...'
  uv venv "$project_dir/competition/.venv" --python 3.11
fi

echo 'Syncing the competition environment with uv...'
uv pip sync --python "$competition_python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "$project_dir/competition/requirements.txt" \
  "$project_dir/competition/requirements-lock.txt" \
  "$project_dir/competition/requirements-test.txt"

echo 'Setup completed. Run: make verify'
