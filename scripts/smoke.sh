#!/usr/bin/env bash
set -euo pipefail

api_url="${DXA_API_URL:-http://127.0.0.1:8001}"
web_url="${DXA_WEB_URL:-http://127.0.0.1:5173}"

curl --fail --silent --show-error "$api_url/health" >/dev/null
curl --fail --silent --show-error "$api_url/api/v1/models" >/dev/null
curl --fail --silent --show-error "$web_url" >/dev/null

echo "Smoke check passed: web, health, and model registry are reachable."
