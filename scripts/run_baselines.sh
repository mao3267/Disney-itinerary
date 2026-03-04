#!/usr/bin/env bash
set -euo pipefail

# -----------------
# Config
# -----------------
BUDGET_MIN="${1:-600}"   # default 600 minutes (10 hours)
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/baselines.py"
PROCESSED_DIR="${REPO_ROOT}/data/processed"
OUTDIR="${REPO_ROOT}/outputs/baselines"

mkdir -p "${OUTDIR}"

SUMMARY="${OUTDIR}/summary.csv"
echo "baseline,wait_type,budget_min,total_time_used_min,total_rating,rides_count,rides_list" > "${SUMMARY}"

# -----------------
# Experiment grid
# -----------------
WAIT_TYPES=("value" "regular" "peak" "all")
BASELINES=("random" "greedy_rating" "max_rating" "max_count")

# -----------------
# Function
# -----------------
run_one () {
  local baseline="$1"
  local wait_type="$2"
  local wait_file="${PROCESSED_DIR}/wait_stats_${wait_type}.json"
  local log_file="${OUTDIR}/${baseline}__${wait_type}__budget${BUDGET_MIN}.txt"

  if [[ ! -f "${wait_file}" ]]; then
    echo "ERROR: missing ${wait_file}" >&2
    exit 1
  fi

  # Always pass --seed (harmless for non-random baselines; useful if you later add seeded behavior)
  python3 "${PY_SCRIPT}" \
    --baseline "${baseline}" \
    --budget_min "${BUDGET_MIN}" \
    --seed "${SEED}" \
    --wait_stats "${wait_file}" \
    | tee "${log_file}" > /dev/null

  # -----------------
  # Parse output
  # -----------------
  local total_time total_rating rides_count rides_list

  total_time="$(grep -E "Total time used \(min\):" "${log_file}" | awk -F': ' '{print $2}' | head -n1 | tr -d '\r')"
  total_rating="$(grep -E "^Total rating:" "${log_file}" | awk -F': ' '{print $2}' | head -n1 | tr -d '\r')"

  # Extract ride keys from lines like:
  #   - seven_dwarfs_train      | Seven Dwarfs ... | rating=...
  rides_list="$(grep -E "^  - " "${log_file}" | sed -E 's/^  - ([^[:space:]]+).*/\1/' | paste -sd '|' -)"
  rides_count="$(grep -E "^  - " "${log_file}" | wc -l | tr -d ' ')"

  rides_list="${rides_list:-}"

  echo "${baseline},${wait_type},${BUDGET_MIN},${total_time},${total_rating},${rides_count},\"${rides_list}\"" >> "${SUMMARY}"
}

# -----------------
# Main loop
# -----------------
echo "Running baselines (budget=${BUDGET_MIN})..."
echo

for wt in "${WAIT_TYPES[@]}"; do
  for bl in "${BASELINES[@]}"; do
    echo "==> baseline=${bl}, wait_type=${wt}"
    run_one "${bl}" "${wt}"
  done
done

echo
echo "Done."
echo "Summary: ${SUMMARY}"
echo "Logs: ${OUTDIR}/"