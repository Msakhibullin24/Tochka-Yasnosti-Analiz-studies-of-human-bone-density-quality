#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

npm run lint
npm test
npm run build
backend/.venv/bin/python -m pytest -q backend/tests

echo "All static checks, frontend tests, backend tests, and production build passed."
