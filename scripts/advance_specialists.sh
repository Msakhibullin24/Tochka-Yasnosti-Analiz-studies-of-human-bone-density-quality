#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
[[ $# -eq 2 ]] || { echo 'Usage: bash scripts/advance_specialists.sh DATASET NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
dataset_dir="$(realpath "$1")"
experiment_dir="$(realpath -m "$2")"
[[ -d "$dataset_dir" && ! -e "$experiment_dir" ]] || { echo 'Dataset must exist; output must be new.' >&2; exit 2; }
python_bin="$project_dir/data/specialists/venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Run make specialists first.' >&2; exit 2; }
backend/.venv/bin/python backend/scripts/fetch_specialists.py --verify
mkdir -p "$experiment_dir"
export HF_HUB_OFFLINE=1
unset DXAQC_SPECIALIST_PATH
"$python_bin" competition/prepare_annotations.py --dataset "$dataset_dir" --output "$experiment_dir/annotations"
"$python_bin" competition/train_specialist.py --dataset "$dataset_dir" \
  --weights data/specialists/weights/convnextv2_tiny.safetensors --output "$experiment_dir/joint" \
  2>&1 | tee "$experiment_dir/joint.log"
for region in spine hip; do
  "$python_bin" competition/train_specialist.py --dataset "$dataset_dir" \
    --weights data/specialists/weights/convnextv2_tiny.safetensors --region "$region" \
    --train-mode last-stage --epochs 12 --output "$experiment_dir/$region" \
    2>&1 | tee "$experiment_dir/$region.log"
done
"$python_bin" competition/pack_specialists.py --spine "$experiment_dir/spine" --hip "$experiment_dir/hip" --output "$experiment_dir/suite"
"$python_bin" competition/benchmark_specialists.py --dataset "$dataset_dir" --joint "$experiment_dir/joint" \
  --spine "$experiment_dir/spine" --hip "$experiment_dir/hip" --output "$experiment_dir/benchmark" \
  2>&1 | tee "$experiment_dir/benchmark.log"
"$python_bin" competition/report_specialists.py --runs "$experiment_dir/joint" "$experiment_dir/spine" "$experiment_dir/hip" --output "$experiment_dir/report"
printf 'Completed QC training. Expert annotation pending: %s/annotations\nSuite: %s/suite\nMetrics: %s/benchmark/index.html\n' "$experiment_dir" "$experiment_dir" "$experiment_dir"
