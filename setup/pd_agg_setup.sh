#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------
# Aggregate (single-server) vLLM
# ----------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/dev/shm/vllm_cache}"   # same as disagg
PORT="${PORT:-9000}"
GPU="${GPU:-0}"
UTIL="${UTIL:-0.8}"                              # match disagg default 0.8
MAX_LEN="${MAX_LEN:-32768}"                      # safe cap for Qwen2.5-7B; same as disagg producer

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# Log exactly under repo logs/, like disagg
export VLLM_LOGGING_DIR="$LOG_DIR"
export PYTHONUNBUFFERED=1

# Align “SAFE” flags with disagg producer
export VLLM_TORCH_COMPILE=0
export VLLM_USE_DYNAMO=0
export VLLM_DISABLE_CUDA_GRAPH=1

die(){ echo "❌ $*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

need vllm
need ss
need curl

# Clean up any stale agg on :$PORT before starting (consistent w/ disagg cleanup)
echo "🧹 Cleaning up old aggregated process and port $PORT…"
if ss -ltn "sport = :$PORT" | tail -n +2 | grep -q . ; then
  # kill the process that owns the port
  PID="$(ss -ltnp | awk -v p=":$PORT" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1 || true)"
  if [[ -n "${PID:-}" ]]; then
    echo "🔪 Killing process using port $PORT (pid=$PID)"
    kill "$PID" || true
    sleep 1
  fi
fi
echo "✅ Cleanup done."

AGG_LOG="$LOG_DIR/agg.log"
: > "$AGG_LOG"

echo "🚀 Launch: Aggregated vLLM on :${PORT} (GPU=${GPU}, UTIL=${UTIL}, MAX_LEN=${MAX_LEN}, MODEL=${MODEL})"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="${GPU}" \
nohup vllm serve "$MODEL" \
  --port "$PORT" \
  --download-dir "$CACHE_DIR" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_LEN" \
  --enforce-eager \
  >> "$AGG_LOG" 2>&1 < /dev/null &

P_AGG=$!
echo "$P_AGG" > "$LOG_DIR/.pid.agg"

# Quick liveness check
sleep 3
kill -0 "$P_AGG" 2>/dev/null || die "Aggregated server exited immediately; see $AGG_LOG"

# Wait for HTTP port (up to 90s)
for i in $(seq 1 90); do
  ss -ltn "sport = :$PORT" | tail -n +2 | grep -q . && break || true
  sleep 1
done
ss -ltn "sport = :$PORT" | tail -n +2 | grep -q . || die "Timeout: aggregated HTTP :$PORT did not open"

echo "✅ Aggregated server pid=$P_AGG ($AGG_LOG)"
echo "🔎 Sanity: /v1/models"
curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 200; echo
echo "ℹ To stop: kill \$(cat logs/.pid.agg)"