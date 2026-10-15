#!/usr/bin/env bash
# Explicit research GPU image; the released CPU entrypoint is competition/run.sh.
set -euo pipefail

task_image=${DXAQC_ANATOMY_IMAGE:-dxaqc:anatomy-fullfit-1.11.7-20261015}
task_gpu_mode=${DXAQC_GPU_MODE:-toolkit}
task_gpu_args=()
case "$task_gpu_mode" in
  toolkit) task_gpu_args=(--gpus all) ;;
  devices)
    for task_device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
      [[ -c "$task_device" ]] || { echo "Missing GPU device: $task_device" >&2; exit 1; }
      task_gpu_args+=(--device "$task_device")
    done
    for task_soname in libcuda.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
      task_library=$(ldconfig -p | awk -v name="$task_soname" '$1 == name && !found {print $NF; found=1}')
      [[ -n "$task_library" && -f "$task_library" ]] || { echo "Missing driver library: $task_soname" >&2; exit 1; }
      task_gpu_args+=(-v "$task_library:/usr/local/nvidia/lib64/$task_soname:ro")
    done
    task_library=$(ldconfig -p | awk '$1 ~ /^libnvidia-gpucomp\.so\./ && !found {print $NF; found=1}')
    if [[ -n "$task_library" ]]; then
      task_gpu_args+=(-v "$task_library:/usr/local/nvidia/lib64/$(basename "$task_library"):ro")
    fi
    task_gpu_args+=(-e LD_LIBRARY_PATH=/usr/local/nvidia/lib64)
    ;;
  *) echo 'DXAQC_GPU_MODE must be toolkit or devices' >&2; exit 2 ;;
esac

if [[ -n ${DXAQC_CONTAINER_USER:-} ]]; then
  task_user=$DXAQC_CONTAINER_USER
elif [[ $(docker info --format '{{json .SecurityOptions}}') == *rootless* ]]; then
  task_user=0:0
else
  task_user="$(id -u):$(id -g)"
fi

case "${1:-}" in
  batch)
    [[ $# == 3 ]] || { echo 'Usage: run_anatomy.sh batch INPUT OUTPUT' >&2; exit 2; }
    task_input=$(realpath "$2")
    mkdir -p "$3"
    task_output=$(realpath "$3")
    docker run --rm --network none --user "$task_user" "${task_gpu_args[@]}" \
      -v "$task_input:/data/input:ro" -v "$task_output:/data/output" "$task_image" \
      python experiments/live_anatomy_deployment.py --input /data/input --output /data/output --require-network-isolation
    ;;
  serve)
    [[ $# -le 2 ]] || { echo 'Usage: run_anatomy.sh serve [PORT]' >&2; exit 2; }
    task_port=${2:-8091}
    [[ $task_port =~ ^[0-9]+$ && $task_port -ge 1 && $task_port -le 65535 ]] || { echo 'Invalid port' >&2; exit 2; }
    task_data=${DXAQC_DATA:-"$PWD/competition/out-anatomy"}
    mkdir -p "$task_data"
    task_data=$(realpath "$task_data")
    task_server_args=()
    if [[ -n ${DXAQC_CONTAINER_NAME:-} ]]; then task_server_args+=(--name "$DXAQC_CONTAINER_NAME"); fi
    if [[ -n ${DXAQC_DOCKER_NETWORK:-} ]]; then task_server_args+=(--network "$DXAQC_DOCKER_NETWORK"); fi
    docker run --rm --user "$task_user" "${task_gpu_args[@]}" "${task_server_args[@]}" -p "127.0.0.1:$task_port:8000" \
      -v "$task_data:/work" "$task_image"
    ;;
  *) echo 'Usage: run_anatomy.sh batch INPUT OUTPUT | serve [PORT]' >&2; exit 2 ;;
esac
