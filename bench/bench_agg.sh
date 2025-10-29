#!/usr/bin/env bash
set -euo pipefail

# Resolve absolute paths so this works from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (env-overridable)
SRV_IP="${SRV_IP:-127.0.0.1}"
AGG_HTTP_PORT="${AGG_HTTP_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Optional: print where we’re running from
echo "🚀 Running aggregated bench against http://${SRV_IP}:${AGG_HTTP_PORT} (model=${MODEL})"
echo "📄 bench.py = ${SCRIPT_DIR}/bench.py"

# Hand through any extra CLI args to bench.py (e.g., --requests, --concurrency, etc.)
python3 "${SCRIPT_DIR}/bench.py" \
  --host "${SRV_IP}" \
  --port "${AGG_HTTP_PORT}" \
  --model "${MODEL}" \
  "$@"