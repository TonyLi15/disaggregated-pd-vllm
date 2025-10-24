#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------
# User-configurable parameters
# ----------------------------
SRV_IP="${SRV_IP:-$(hostname -I | awk '{print $1}')}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/vllm}"

# Proxy (HTTP must match proxy script; ZMQ must match producer/consumer extra_config)
PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10001}"
PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-30002}"

# Consumer (decode)
CONS_HTTP_PORT="${CONS_HTTP_PORT:-8200}"
CONS_ZMQ_PORT="${CONS_ZMQ_PORT:-14579}"
CONS_GPU="${CONS_GPU:-2}"
CONS_UTIL="${CONS_UTIL:-0.8}"
CONS_WAIT="${CONS_WAIT:-120}"

# Producer (prefill; runs SAFE mode by default)
PROD_HTTP_PORT="${PROD_HTTP_PORT:-8100}"
PROD_ZMQ_PORT="${PROD_ZMQ_PORT:-14580}"
PROD_GPU="${PROD_GPU:-1}"
PROD_UTIL="${PROD_UTIL:-0.8}"
PROD_WAIT="${PROD_WAIT:-120}"

OPENAI_API_KEY="${OPENAI_API_KEY:-sk-noop}"

# ----------------------------
# Helpers
# ----------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

die(){ echo "❌ $*" >&2; exit 1; }
note(){ echo "▶ $*"; }

need(){ command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

port_free(){
  local port="$1"
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q . ; then
    die "Port $port is already in use. Free it or change the port."
  fi
}

wait_tcp(){
  local port="$1" timeout="$2"
  local t=0
  while ! ss -ltn "sport = :$port" | tail -n +2 | grep -q . ; do
    sleep 1; t=$((t+1))
    if [[ "$t" -ge "$timeout" ]]; then
      return 1
    fi
  done
  return 0
}

kv_json(){
  # $1 role, $2 http_port, $3 zmq_port
  local role="$1" http_p="$2" zmq_p="$3"
  cat <<JSON
{
  "kv_connector":"P2pNcclConnector",
  "kv_role":"${role}",
  "kv_rank":0,
  "kv_parallel_size":1,
  "kv_ip":"${SRV_IP}",
  "kv_port":${zmq_p},
  "kv_connector_extra_config":{
    "http_port":"${http_p}",
    "proxy_ip":"${SRV_IP}",
    "proxy_port":"${PROXY_ZMQ_PORT}"
  }
}
JSON
}

# ----------------------------
# Validation
# ----------------------------
need vllm
need python
[[ -n "$OPENAI_API_KEY" ]] || die "Set OPENAI_API_KEY."

note "SRV_IP=${SRV_IP}"
note "MODEL=${MODEL}"
note "Proxy:   HTTP=${PROXY_HTTP_PORT}, ZMQ=${PROXY_ZMQ_PORT}"
note "Consumer:HTTP=${CONS_HTTP_PORT}, ZMQ=${CONS_ZMQ_PORT}, GPU=${CONS_GPU}, UTIL=${CONS_UTIL}"
note "Producer:HTTP=${PROD_HTTP_PORT}, ZMQ=${PROD_ZMQ_PORT}, GPU=${PROD_GPU}, UTIL=${PROD_UTIL}"

for p in "$PROXY_HTTP_PORT" "$PROXY_ZMQ_PORT" "$CONS_HTTP_PORT" "$CONS_ZMQ_PORT" "$PROD_HTTP_PORT" "$PROD_ZMQ_PORT"; do
  port_free "$p"
done

# ----------------------------
# Cleanup on any failure/exit
# ----------------------------
P_PROXY=""
P_CONS=""
P_PROD=""

cleanup() {
  echo "❗ failure detected. cleaning up…"
  if [[ -n "$P_PROXY" ]] && kill -0 "$P_PROXY" 2>/dev/null; then
    echo "🧹 stopping proxy (pid=$P_PROXY)"; kill "$P_PROXY" || true
  fi
  if [[ -n "$P_CONS" ]] && kill -0 "$P_CONS" 2>/dev/null; then
    echo "🧹 stopping consumer (pid=$P_CONS)"; kill "$P_CONS" || true
  fi
  if [[ -n "$P_PROD" ]] && kill -0 "$P_PROD" 2>/dev/null; then
    echo "🧹 stopping producer (pid=$P_PROD)"; kill "$P_PROD" || true
  fi
}
trap cleanup ERR INT

# ----------------------------
# 1) Proxy
# ----------------------------
PROXY_LOG="$LOG_DIR/proxy.log"
echo "" > "$PROXY_LOG"

echo "🚀 Launch: Proxy (Quart) on HTTP:${PROXY_HTTP_PORT} ZMQ:${PROXY_ZMQ_PORT}"
(
  cd "$ROOT_DIR/proxy"
  # Run with python directly; logs go to file.
  nohup python disagg_proxy_p2p_nccl_xpyd.py > "$PROXY_LOG" 2>&1 &
  P_PROXY=$!
  echo "$P_PROXY" > "$LOG_DIR/.pid.proxy"
)
# Wait for port
wait_tcp "$PROXY_HTTP_PORT" 20 || die "Proxy did not open HTTP :$PROXY_HTTP_PORT"
echo "  ↳ proxy pid=$(cat "$LOG_DIR/.pid.proxy") ($PROXY_LOG)"
echo "✅ Proxy ready"

# ----------------------------
# 2) Consumer (Decode)
# ----------------------------
CONS_LOG="$LOG_DIR/consumer.log"
echo "" > "$CONS_LOG"

echo "🚀 Launch: Consumer (decode) @ GPU${CONS_GPU}"
env CUDA_VISIBLE_DEVICES="${CONS_GPU}" \
nohup vllm serve "$MODEL" \
  --port "$CONS_HTTP_PORT" \
  --download-dir "$CACHE_DIR" \
  --gpu-memory-utilization "$CONS_UTIL" \
  --max-model-len 2048 \
  --kv-transfer-config "$(kv_json "kv_consumer" "$CONS_HTTP_PORT" "$CONS_ZMQ_PORT")" \
  >> "$CONS_LOG" 2>&1 &
P_CONS=$!
echo "$P_CONS" > "$LOG_DIR/.pid.consumer"
echo "  ↳ consumer pid=$P_CONS ($CONS_LOG)"

# Wait short first; if the process dies early, bail fast
sleep 3
if ! kill -0 "$P_CONS" 2>/dev/null; then
  die "Consumer exited early; check $CONS_LOG"
fi
if ! wait_tcp "$CONS_HTTP_PORT" "$CONS_WAIT"; then
  die "Timeout: consumer HTTP :$CONS_HTTP_PORT did not open"
fi
echo "✅ Consumer ready"

# ----------------------------
# 3) Producer (Prefill, SAFE mode)
# ----------------------------
PROD_LOG="$LOG_DIR/producer.log"
echo "" > "$PROD_LOG"

echo "🚀 Launch: Producer (prefill, SAFE: eager/No CUDA-graph) @ GPU${PROD_GPU}"
(
  export VLLM_TORCH_COMPILE=0
  export VLLM_USE_DYNAMO=0
  export VLLM_DISABLE_CUDA_GRAPH=1
  env CUDA_VISIBLE_DEVICES="${PROD_GPU}" \
  nohup vllm serve "$MODEL" \
    --port "$PROD_HTTP_PORT" \
    --download-dir "$CACHE_DIR" \
    --gpu-memory-utilization "$PROD_UTIL" \
    --max-model-len 2048 \
    --kv-transfer-config "$(kv_json "kv_producer" "$PROD_HTTP_PORT" "$PROD_ZMQ_PORT")" \
    >> "$PROD_LOG" 2>&1 &
)
# Discover PID (nohup subshell)
P_PROD=$(pgrep -n -f "vllm serve $MODEL.*--port $PROD_HTTP_PORT" || true)
[[ -n "$P_PROD" ]] || die "Producer PID not found; check $PROD_LOG"
echo "$P_PROD" > "$LOG_DIR/.pid.producer"
echo "  ↳ producer pid=$P_PROD ($PROD_LOG)"

sleep 3
if ! kill -0 "$P_PROD" 2>/dev/null; then
  die "Producer exited early; check $PROD_LOG"
fi
if ! wait_tcp "$PROD_HTTP_PORT" "$PROD_WAIT"; then
  die "Timeout: producer HTTP :$PROD_HTTP_PORT did not open"
fi
echo "✅ Producer ready"

# ----------------------------
# 4) Smoke test through proxy
# ----------------------------
echo "🔎 Sanity check: /v1/chat/completions via proxy"
REQ='{
  "model":"'"$MODEL"'",
  "messages":[{"role":"user","content":"hello"}],
  "max_tokens":16,
  "stream":false
}'
HTTP_CODE=$(curl -sS -o "$LOG_DIR/.probe.out" -w "%{http_code}" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  --data "$REQ" \
  "http://${SRV_IP}:${PROXY_HTTP_PORT}/v1/chat/completions" || true)

if [[ "$HTTP_CODE" != "200" ]]; then
  cat "$LOG_DIR/.probe.out"
  echo
  echo "❌ end-to-end failed (HTTP $HTTP_CODE)"
  echo "  - $PROXY_LOG"
  echo "  - $CONS_LOG"
  echo "  - $PROD_LOG"
  exit 2
fi

echo "✅ end-to-end OK"
cat "$LOG_DIR/.probe.out"
echo

# Success: remove traps so processes keep running
trap - ERR INT
echo "ℹ To stop: kill \$(cat logs/.pid.*) or run pkill lines in README."