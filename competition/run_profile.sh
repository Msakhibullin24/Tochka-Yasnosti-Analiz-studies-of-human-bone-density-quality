#!/usr/bin/env bash
# Build and run a local workflow whose code and weights match its manifest.
set -euo pipefail
if [[ $# != 4 || ( "$1" != batch && "$1" != api ) ]]; then
  echo "Usage: $0 batch|api PROFILE_DIRECTORY INPUT_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi
mode=$1
profile_dir=$(realpath -- "$2")
input_dir=$(realpath -- "$3")
mkdir -p -- "$4"
output_dir=$(realpath -- "$4")
[[ -f "$profile_dir/profile.json" && -d "$input_dir" ]] || { echo 'Profile manifest and DICOM input directory are required.' >&2; exit 2; }
if [[ "$mode" == batch && -e "$output_dir/results" ]]; then
  echo 'Choose a new output directory: results already exist.' >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=osseo-workflow:local
docker build -f "$repo_root/competition/Dockerfile" -t "$image" "$repo_root"
common=(--rm --cpus 4 --memory 4g --user "$(id -u):$(id -g)"
  -e DXAQC_WORKFLOW_PROFILE=/profile/profile.json -e DXAQC_DATA_DIR=/data/out
  -v "$profile_dir:/profile:ro" -v "$input_dir:/data/in:ro" -v "$output_dir:/data/out")
if [[ "$mode" == batch ]]; then
  exec docker run "${common[@]}" --network none --entrypoint python "$image" \
    run_no_new_labels.py --input /data/in --output /data/out/results --profile /profile/profile.json
else
  exec docker run "${common[@]}" -p "127.0.0.1:${PORT:-8000}:8000" "$image"
fi
