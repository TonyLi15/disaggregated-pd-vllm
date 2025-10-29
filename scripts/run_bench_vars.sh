#!/usr/bin/env bash
set -euo pipefail

# ==============================
# vLLM Benchmark Runner (Disaggregated + Aggregated)
# ==============================

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="results/bench_runs"
mkdir -p "${OUTPUT_DIR}"

# Benchmark parameters
CONCURRENCIES=(1 2 4 8)
PROMPT_TOKENS=(256 512)
MAX_TOKENS=(512 1024)
REQUESTS=10

# Define bench scripts
PROXY_BENCH="./bench/bench_proxy.sh"
AGG_BENCH="./bench/bench_agg.sh"

# Function to run and collect results
run_one() {
  local mode="$1"  # disagg or agg
  local conc="$2"
  local p_t="$3"
  local m_t="$4"

  local MODEL_TAG
  MODEL_TAG=$(echo "${MODEL}" | tr '/:' '_')
  local RUN_ID="${mode}_model_${MODEL_TAG}_conc${conc}_pt${p_t}_mt${m_t}"

  local LOG_FILE="${OUTPUT_DIR}/run_${RUN_ID}.log"
  local CSV_FILE="${OUTPUT_DIR}/run_${RUN_ID}.csv"

  echo "=== Running ${RUN_ID} (N=${REQUESTS}) ==="

  if [[ "$mode" == "disagg" ]]; then
    "${PROXY_BENCH}" \
      --model "${MODEL}" \
      --requests "${REQUESTS}" \
      --concurrency "${conc}" \
      --prompt-tokens "${p_t}" \
      --max-tokens "${m_t}" \
      > "${LOG_FILE}" 2>&1
  else
    "${AGG_BENCH}" \
      --model "${MODEL}" \
      --requests "${REQUESTS}" \
      --concurrency "${conc}" \
      --prompt-tokens "${p_t}" \
      --max-tokens "${m_t}" \
      > "${LOG_FILE}" 2>&1
  fi

  python3 scripts/collect_from_log.py --input "${LOG_FILE}" --output "${CSV_FILE}" || true
  echo "✅ Saved CSV to ${CSV_FILE}"
}

# ==============================
# Main Loop
# ==============================
for conc in "${CONCURRENCIES[@]}"; do
  for p_t in "${PROMPT_TOKENS[@]}"; do
    for m_t in "${MAX_TOKENS[@]}"; do
      # Run Disaggregated setup
      run_one "disagg" "${conc}" "${p_t}" "${m_t}"

      # Run Aggregated setup
      run_one "agg" "${conc}" "${p_t}" "${m_t}"
    done
  done
done

echo "🎯 All benchmark runs complete."