#!/usr/bin/env bash
set -euo pipefail

# Default model (can be overridden by environment variable)
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"


# Fixed parameters
# PRODUCER=1
# CONSUMER=1
OUTPUT_DIR="results/bench_runs"
mkdir -p "${OUTPUT_DIR}"


# Variable arrays
CONCURRENCIES=(1 2 4 8)
PROMPT_TOKENS=(32 64 128)
MAX_TOKENS=(128 256)


for conc in "${CONCURRENCIES[@]}"; do
  for p_t in "${PROMPT_TOKENS[@]}"; do
    for m_t in "${MAX_TOKENS[@]}"; do
      MODEL_TAG=$(echo "${MODEL}" | tr '/:' '_')
      RUN_ID="model_${MODEL_TAG}_conc${conc}_pt${p_t}_mt${m_t}"
      LOG_FILE="${OUTPUT_DIR}/run_${RUN_ID}.log"
      CSV_FILE="${OUTPUT_DIR}/run_${RUN_ID}.csv"

      echo "=== Running ${RUN_ID} ==="
      python3 bench/bench_pd.py \
        --model "${MODEL}" \
        --concurrency ${conc} \
        --prompt-tokens ${p_t} \
        --max-tokens ${m_t} \
        > "${LOG_FILE}" 2>&1

      python3 scripts/collect_from_log.py --input "${LOG_FILE}" --output "${CSV_FILE}"
      echo "Saved CSV to ${CSV_FILE}"
    done
  done
done