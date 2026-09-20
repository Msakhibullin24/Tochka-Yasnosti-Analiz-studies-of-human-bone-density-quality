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
IMAGE="${DXAQC_IMAGE:-osseo-dxaqc:1.9.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cmd="${1:-help}"

case "$cmd" in
  build)
    docker build -f "$HERE/Dockerfile" -t "$IMAGE" "$HERE/.."
    ;;
  batch)
    [ $# -ge 3 ] || { echo "usage: $0 batch <input folder|zip> <output dir>" >&2; exit 64; }
    in="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    mkdir -p "$3"; out="$(cd "$3" && pwd)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -f "$HERE/Dockerfile" -t "$IMAGE" "$HERE/.."
    docker run --rm --network none --user "$(id -u):$(id -g)" \
      -v "$in":/data/input:ro -v "$out":/data/out \
      "$IMAGE" python -m dxaqc.cli --input /data/input --output /data/out
    ;;
  serve)
    port="${2:-8000}"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -f "$HERE/Dockerfile" -t "$IMAGE" "$HERE/.."
    work="${DXAQC_DATA:-$HERE/out}"
    mkdir -p "$work"
    work="$(cd "$work" && pwd)"
    docker run --rm --user "$(id -u):$(id -g)" -p "127.0.0.1:${port}:8000" \
      -v "$work":/work -v "$work":/data "$IMAGE"
    ;;
  test)
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -f "$HERE/Dockerfile" -t "$IMAGE" "$HERE/.."
    docker run --rm --network none --tmpfs /tmp -v "$HERE/tests":/app/tests:ro "$IMAGE" \
      python -m pytest -q -p no:cacheprovider tests
    fixture_dir="$(mktemp -d)"
    trap 'rm -rf "$fixture_dir"' EXIT HUP INT TERM
    docker run --rm --network none --user "$(id -u):$(id -g)" \
      -v "$HERE/tests":/app/tests:ro -v "$fixture_dir":/fixtures -e PYTHONPATH=/app/tests \
      "$IMAGE" python -c 'from pathlib import Path; import zipfile; from conftest import synthetic_spine, write_dicom; p=write_dicom(Path("/fixtures/a.dcm"), synthetic_spine()); z=zipfile.ZipFile("/fixtures/input.zip", "w"); z.write(p,"a.dcm"); z.close()'
    "$HERE/run.sh" batch "$fixture_dir/input.zip" "$fixture_dir/out"
    docker run --rm --network none -v "$fixture_dir/out":/results:ro "$IMAGE" \
      python -c 'import json; from pathlib import Path; s=json.loads(Path("/results/summary.json").read_text()); assert s["files"] == s["success"] == 1, s'
    ;;
  *)
    sed -n '2,9p' "$0"
    ;;
esac
