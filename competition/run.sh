#!/usr/bin/env sh
# Build and run the DXA quality-control container (Linux / UNIX-like systems).
#
#   ./run.sh build                       build the image
#   ./run.sh batch <input> <output_dir>  process a folder or .zip -> results.csv, results.xlsx, additional_series.zip
#   ./run.sh serve [port]                start the HTTP API + web UI (default port 8000)
#   ./run.sh test                        run the unit tests inside the image
#
# The container never needs the network at run time: batch mode runs with --network none.
set -eu
IMAGE="${DXAQC_IMAGE:-osseo-dxaqc:1.0.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cmd="${1:-help}"

case "$cmd" in
  build)
    docker build -t "$IMAGE" "$HERE"
    ;;
  batch)
    [ $# -ge 3 ] || { echo "usage: $0 batch <input folder|zip> <output dir>" >&2; exit 64; }
    in="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    mkdir -p "$3"; out="$(cd "$3" && pwd)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$HERE"
    docker run --rm --network none --user "$(id -u):$(id -g)" \
      -v "$in":/data/in:ro -v "$out":/data/out \
      "$IMAGE" python -m dxaqc.cli --input /data/in --output /data/out
    ;;
  serve)
    port="${2:-8000}"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$HERE"
    docker run --rm -p "127.0.0.1:${port}:8000" -v "${DXAQC_DATA:-$HERE/out}":/data "$IMAGE"
    ;;
  test)
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$HERE"
    docker run --rm --network none --tmpfs /tmp -v "$HERE/tests":/app/tests:ro "$IMAGE" \
      python -m pytest -q -p no:cacheprovider tests
    ;;
  *)
    sed -n '2,9p' "$0"
    ;;
esac
