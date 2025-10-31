#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRV_IP="${SRV_IP:-172.16.40.99}"
PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10001}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "🚀 Running bench_pd.py from $ROOT_DIR"
python3 "${SCRIPT_DIR}/bench_pd.py" \
  --url "http://${SRV_IP}:${PROXY_HTTP_PORT}/v1/chat/completions" \
  --model "$MODEL" "$@"