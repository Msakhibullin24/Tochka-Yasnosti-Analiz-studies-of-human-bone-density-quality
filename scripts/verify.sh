#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

command -v uv >/dev/null || { echo 'uv is required; run make setup after installing uv.' >&2; exit 2; }
command -v pnpm >/dev/null || { echo 'pnpm is required; run make setup after installing pnpm.' >&2; exit 2; }
competition_python="$project_dir/competition/.venv/bin/python"
if [[ ! -x "$competition_python" ]]; then
  echo "Competition environment is missing. Run: make setup" >&2
  exit 2
fi

pnpm run lint
pnpm test
pnpm run build
uv run --project "$project_dir/backend" --locked --extra test python -m pytest -q backend/tests
"$competition_python" -m pytest -q competition/tests

echo "All static checks, frontend tests, backend tests, and production build passed."
