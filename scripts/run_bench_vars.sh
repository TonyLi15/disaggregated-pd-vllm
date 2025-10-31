# scripts/run_bench_vars.sh
#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# End-to-end Benchmark Orchestrator
# (runs disaggregated &/or aggregated benches,
#  collects CSVs, then generates all figures)
# ==========================================

# -------- Parameters --------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bench_runs}"
REQUESTS="${REQUESTS:-10}"

# Sweeps
CONCURRENCIES=(${CONCURRENCIES:-1 2 4 8 16 32})
PROMPT_TOKENS=(${PROMPT_TOKENS:-256 512 1024 2048 4096 8192})
MAX_TOKENS=(${MAX_TOKENS:-512 1024 2048 4096 8192})

# Which modes to run: disagg / agg / both
MODE_SET="${MODE_SET:-both}"

# Behavior toggles
RESUME="${RESUME:-1}"              # if 1, skip when CSV exists
FORCE_RECOLLECT="${FORCE_RECOLLECT:-0}"  # if 1, always regenerate CSV from log
RUN_PLOTS="${RUN_PLOTS:-1}"        # if 1, call plot_bench_results.py at the end
POST_COLLECT_ALL="${POST_COLLECT_ALL:-0}" # if 1, sweep all .log and recollect CSVs

# Paths
PROXY_BENCH="${PROXY_BENCH:-./bench/bench_proxy.sh}"
AGG_BENCH="${AGG_BENCH:-./bench/bench_agg.sh}"
COLLECT_PY="${COLLECT_PY:-scripts/collect_from_log.py}"
PLOT_PY="${PLOT_PY:-scripts/plot_bench_results.py}"

mkdir -p "${OUTPUT_DIR}"

# -------- Helpers --------
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

  if [[ "${RESUME}" == "1" && -s "${CSV_FILE}" ]]; then
    echo "⏭  Skip (CSV exists): ${CSV_FILE}"
    return 0
  fi

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

  if [[ "${FORCE_RECOLLECT}" == "1" || ! -s "${CSV_FILE}" ]]; then
    python3 "${COLLECT_PY}" --input "${LOG_FILE}" --output "${CSV_FILE}" || true
  fi
  echo "✅ Saved CSV to ${CSV_FILE}"
}

# -------- Main Sweep --------
for conc in "${CONCURRENCIES[@]}"; do
  for p_t in "${PROMPT_TOKENS[@]}"; do
    for m_t in "${MAX_TOKENS[@]}"; do
      case "${MODE_SET}" in
        disagg)
          run_one "disagg" "${conc}" "${p_t}" "${m_t}"
          ;;
        agg)
          run_one "agg" "${conc}" "${p_t}" "${m_t}"
          ;;
        both)
          run_one "disagg" "${conc}" "${p_t}" "${m_t}"
          run_one "agg"    "${conc}" "${p_t}" "${m_t}"
          ;;
        *)
          echo "Unknown MODE_SET='${MODE_SET}' (use disagg|agg|both)"; exit 2;;
      esac
    done
  done
done

# -------- Optional post-collection over all logs --------
if [[ "${POST_COLLECT_ALL}" == "1" ]]; then
  echo "🔁 Re-collecting CSVs for all logs in ${OUTPUT_DIR}..."
  shopt -s nullglob
  for lf in "${OUTPUT_DIR}"/run_*.log; do
    cf="${lf%.log}.csv"
    python3 "${COLLECT_PY}" --input "${lf}" --output "${cf}" || true
  done
fi

# -------- Plotting --------
if [[ "${RUN_PLOTS}" == "1" ]]; then
  echo "📈 Generating figures..."
  python3 "${PLOT_PY}"
  echo "🎉 Figures written under results/figures/"
fi

echo "🎯 All benchmark runs complete."