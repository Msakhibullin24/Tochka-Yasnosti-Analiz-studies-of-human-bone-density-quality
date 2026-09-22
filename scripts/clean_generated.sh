#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

apply=0
if [[ "${1:-}" == "--apply" ]]; then
  apply=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--apply]" >&2
  exit 2
fi

paths=(
  ".pytest_cache"
  "backend/.pytest_cache"
  "backend/osseo_dxa_inference.egg-info"
  "competition/.cache"
  "dist"
)

while IFS= read -r cache_path; do
  paths+=("$cache_path")
done < <(
  find backend competition \
    -path 'backend/.venv' -prune -o \
    -type d -name '__pycache__' -print | sort
)

printf '%s\n' "Generated paths selected for cleanup:"
for relative_path in "${paths[@]}"; do
  if [[ -e "$relative_path" || -L "$relative_path" ]]; then
    printf '  %s\n' "$relative_path"
  fi
done

if [[ "$apply" -eq 0 ]]; then
  echo "Dry run. Use 'make clean-generated APPLY=1' to remove these paths."
  exit 0
fi

for relative_path in "${paths[@]}"; do
  if [[ -e "$relative_path" || -L "$relative_path" ]]; then
    rm -rf -- "$relative_path"
  fi
done

# runs/ может содержать полезные результаты экспериментов, поэтому удаляем его
# только когда в нём нет файлов пользователя.
if [[ -d runs ]]; then
  if find runs -type f -print -quit | grep -q .; then
    echo "Skipped runs/: it contains files; inspect it manually."
  else
    rm -rf -- runs
    echo "Removed empty runs/."
  fi
fi

echo "Generated cleanup completed."
