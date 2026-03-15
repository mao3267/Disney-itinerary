#!/usr/bin/env bash
set -euo pipefail

# -----------------
# Config
# -----------------
BUDGET_MIN="${1:-600}"      # default 600 minutes (10 hours)
N_SIM="${2:-1000}"          # default Monte Carlo runs
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/baselines.py"
PROCESSED_DIR="${REPO_ROOT}/data/processed"
OUTDIR="${REPO_ROOT}/outputs/baselines"

mkdir -p "${OUTDIR}"

SUMMARY="${OUTDIR}/summary.csv"
echo "baseline,wait_type,budget_min,total_time_used_min,total_rating,rides_count,rides_list,prob_overrun,mean_sim_time,std_sim_time,p95_sim_time,p99_sim_time,avg_overrun_if_over,min_sim_time,max_sim_time" > "${SUMMARY}"

# -----------------
# Experiment grid
# -----------------
WAIT_TYPES=("value" "regular" "peak" "all")
BASELINES=("random" "greedy_rating" "max_rating" "max_count")

# -----------------
# Helpers
# -----------------
extract_value_after_colon () {
  local pattern="$1"
  local file="$2"
  grep -E "${pattern}" "${file}" | awk -F': ' '{print $2}' | head -n1 | tr -d '\r'
}

# -----------------
# Function
# -----------------
run_one () {
  local baseline="$1"
  local wait_type="$2"

  local wait_json="${PROCESSED_DIR}/wait_stats_${wait_type}.json"
  local wait_npz="${PROCESSED_DIR}/wait_stats_${wait_type}.npz"
  local log_file="${OUTDIR}/${baseline}__${wait_type}__budget${BUDGET_MIN}__sim${N_SIM}.txt"

  if [[ ! -f "${wait_json}" ]]; then
    echo "ERROR: missing ${wait_json}" >&2
    exit 1
  fi

  if [[ ! -f "${wait_npz}" ]]; then
    echo "ERROR: missing ${wait_npz}" >&2
    exit 1
  fi

  python3 "${PY_SCRIPT}" \
    --baseline "${baseline}" \
    --budget_min "${BUDGET_MIN}" \
    --seed "${SEED}" \
    --wait_stats "${wait_json}" \
    --wait_stats_npz "${wait_npz}" \
    --simulate_overrun \
    --n_sim "${N_SIM}" \
    | tee "${log_file}" > /dev/null

  # -----------------
  # Parse deterministic output
  # -----------------
  local total_time total_rating rides_count rides_list
  total_time="$(extract_value_after_colon '^Total time used \(min\):' "${log_file}")"
  total_rating="$(extract_value_after_colon '^Total rating:' "${log_file}")"

  rides_list="$(grep -E '^  - ' "${log_file}" | sed -E 's/^  - ([^[:space:]]+).*/\1/' | paste -sd '|' -)"
  rides_count="$(grep -E '^  - ' "${log_file}" | wc -l | tr -d ' ')"
  rides_list="${rides_list:-}"

  # -----------------
  # Parse simulation output
  # Expected lines:
  #   P(total_time > budget): ...
  #   Mean simulated total time: ...
  #   Std simulated total time: ...
  #   95th percentile total time: ...
  #   99th percentile total time: ...
  #   Avg overrun | overrun happened: ...
  #   Min / Max simulated total time: A / B min
  # -----------------
  local prob_overrun mean_sim_time std_sim_time p95_sim_time p99_sim_time
  local avg_overrun_if_over min_sim_time max_sim_time minmax_line

  prob_overrun="$(extract_value_after_colon '^  P\(total_time > budget\):' "${log_file}")"
  mean_sim_time="$(extract_value_after_colon '^  Mean simulated total time:' "${log_file}" | awk '{print $1}')"
  std_sim_time="$(extract_value_after_colon '^  Std simulated total time:' "${log_file}" | awk '{print $1}')"
  p95_sim_time="$(extract_value_after_colon '^  95th percentile total time:' "${log_file}" | awk '{print $1}')"
  p99_sim_time="$(extract_value_after_colon '^  99th percentile total time:' "${log_file}" | awk '{print $1}')"
  avg_overrun_if_over="$(extract_value_after_colon '^  Avg overrun \| overrun happened:' "${log_file}" | awk '{print $1}')"

  minmax_line="$(extract_value_after_colon '^  Min / Max simulated total time:' "${log_file}")"
  min_sim_time="$(echo "${minmax_line}" | awk -F' / ' '{print $1}')"
  max_sim_time="$(echo "${minmax_line}" | awk -F' / ' '{print $2}' | awk '{print $1}')"

  echo "${baseline},${wait_type},${BUDGET_MIN},${total_time},${total_rating},${rides_count},\"${rides_list}\",${prob_overrun},${mean_sim_time},${std_sim_time},${p95_sim_time},${p99_sim_time},${avg_overrun_if_over},${min_sim_time},${max_sim_time}" >> "${SUMMARY}"
}

# -----------------
# Main loop
# -----------------
echo "Running baselines (budget=${BUDGET_MIN}, n_sim=${N_SIM})..."
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