#!/usr/bin/env bash
# Use the standard Docker service or this workspace's existing private engine.
set -euo pipefail
task_here=$(cd "$(dirname "$0")" && pwd)
task_root=$(cd "$task_here/.." && pwd)
case "${1:-}" in batch|serve) ;; *) echo 'Usage: local_run.sh batch INPUT OUTPUT | serve [PORT]' >&2; exit 2 ;; esac
if ! docker info >/dev/null 2>&1; then
  if [[ -n ${DOCKER_HOST:-} || -n ${DOCKER_CONTEXT:-} ]]; then
    echo 'The explicitly selected Docker service is unavailable.' >&2; exit 1
  fi
  task_state="$task_root/data/build-rootless/runtime-path.txt"
  [[ -f "$task_state" ]] || { echo 'Start Docker before running this project.' >&2; exit 1; }
  IFS= read -r task_runtime < "$task_state"
  [[ -S "$task_runtime/docker.sock" ]] || { echo 'The private project Docker service is stopped.' >&2; exit 1; }
  export DOCKER_HOST="unix://$task_runtime/docker.sock"
fi
if [[ -z ${DXAQC_GPU_MODE:-} ]]; then
  task_runtimes=$(docker info --format '{{json .Runtimes}}')
  if [[ "$task_runtimes" == *nvidia* ]]; then export DXAQC_GPU_MODE=toolkit
  else export DXAQC_GPU_MODE=devices; fi
fi
if [[ $1 == serve && -z ${DXAQC_DOCKER_NETWORK:-} ]]; then
  if ! docker network inspect bridge >/dev/null 2>&1; then
    task_network=dxaqc-local
    if ! docker network inspect "$task_network" >/dev/null 2>&1; then docker network create "$task_network" >/dev/null; fi
    export DXAQC_DOCKER_NETWORK="$task_network"
  fi
fi
exec "$task_here/run_anatomy.sh" "$@"
