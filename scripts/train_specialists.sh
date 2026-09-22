#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/train_specialists.sh DATASET_DIRECTORY NEW_EXPERIMENT_DIRECTORY' >&2
  exit 2
fi
dataset_dir="$(realpath "$1")"
experiment_dir="$(realpath -m "$2")"
[[ -d "$dataset_dir" && ! -e "$experiment_dir" ]] || { echo 'Dataset must exist; experiment directory must be new.' >&2; exit 2; }
python_bin="$project_dir/data/specialists/venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Run make specialists first.' >&2; exit 2; }
mkdir -p "$experiment_dir"
export HF_HUB_OFFLINE=1
for variant in convnextv2-bce convnextv2-asl efficientnet-b4-bce; do
  backbone=convnextv2_tiny
  checkpoint=convnextv2_tiny
  loss=bce
  if [[ "$variant" == convnextv2-asl ]]; then loss=asl; fi
  if [[ "$variant" == efficientnet-b4-bce ]]; then backbone=tf_efficientnet_b4; checkpoint=efficientnet_b4; fi
  "$python_bin" competition/train_specialist.py --dataset "$dataset_dir" \
    --weights "data/specialists/weights/$checkpoint.safetensors" \
    --backbone "$backbone" --loss "$loss" --output "$experiment_dir/$variant" \
    2>&1 | tee "$experiment_dir/$variant.log"
done
"$python_bin" competition/report_specialists.py \
  --runs "$experiment_dir/convnextv2-bce" "$experiment_dir/convnextv2-asl" "$experiment_dir/efficientnet-b4-bce" \
  --output "$experiment_dir/report"
printf 'Report: %s/report/index.html\n' "$experiment_dir"
