#!/usr/bin/env bash

# Comprehensive H100 GPU training + evaluation sweep for AMLNet and Elliptic.
# Runs the scaled configurations (and key risk variants) sequentially, writing
# logs and artifacts into timestamped folders so results are easy to compare.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_h100_full_suite.sh [--skip-setup] [--dry-run] [-h|--help]

Options:
  --skip-setup  Skip GPU health check (scripts/check_h100_setup.py)
  --dry-run     Print commands without executing them
  -h, --help    Show this help message
USAGE
}

SKIP_SETUP=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] The 'uv' CLI is required. Install it before running this script." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

DATA_AML="data/amlnet/AMLNet_August_2025.csv"
DATA_ELLIPTIC_DIR="data/elliptic"

if [[ ! -f "${DATA_AML}" ]]; then
  echo "[ERROR] Unable to find AMLNet dataset at ${DATA_AML}. Sync data before running." >&2
  exit 1
fi

if [[ ! -d "${DATA_ELLIPTIC_DIR}" ]]; then
  echo "[ERROR] Unable to find Elliptic dataset directory at ${DATA_ELLIPTIC_DIR}." >&2
  exit 1
fi

LOG_DIR="logs/h100_full_suite"
RUN_ROOT="outputs/h100_full_suite"
RUN_TAG="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
SUMMARY_FILE="${RUN_DIR}/summary.csv"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
printf "job,start_utc,end_utc,status,output_dir,log_file,duration_minutes\n" > "${SUMMARY_FILE}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

print_header() {
  cat <<EOF
================================================================================
H100 Full Evaluation Suite
Timestamp (UTC): $(timestamp)
Project Root:    ${ROOT_DIR}
Outputs:         ${RUN_DIR}
Logs:            ${LOG_DIR}
Skip Setup:      ${SKIP_SETUP}
Dry Run:         ${DRY_RUN}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
================================================================================
EOF
}

print_header

if [[ "${SKIP_SETUP}" -eq 0 ]]; then
  echo "[INFO] Running H100 setup verification..."
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] uv run python scripts/check_h100_setup.py"
  else
    uv run python scripts/check_h100_setup.py
  fi
fi

COMPLETED=()
FAILED=()

run_job() {
  local job_name="$1"
  shift

  local output_dir="${RUN_DIR}/${job_name}"
  local log_file="${LOG_DIR}/${RUN_TAG}_${job_name}.log"
  local start_ts
  local end_ts
  local duration_minutes=""
  local status=0

  mkdir -p "${output_dir}"

  start_ts="$(timestamp)"
  echo
  echo "[$(timestamp)] >>> Starting ${job_name}"
  echo "[$(timestamp)] >>> Output: ${output_dir}"
  echo "[$(timestamp)] >>> Log:    ${log_file}"

  local cmd=(uv run python scripts/train_agent.py --output_dir "${output_dir}")
  cmd+=("$@")

  printf "[$(timestamp)] >>> Command: " 
  printf "%q " "${cmd[@]}"
  printf "\n"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    end_ts="$(timestamp)"
    status=0
    duration_minutes="0.0"
  else
    set +e
    "${cmd[@]}" > >(tee "${log_file}") 2> >(tee -a "${log_file}" >&2)
    status=$?
    set -e
    end_ts="$(timestamp)"
    if [[ -f "${output_dir}/training_stats.npz" ]]; then
      local start_epoch end_epoch
      start_epoch=$(date -d "${start_ts}" +%s 2>/dev/null || date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "${start_ts}" +%s)
      end_epoch=$(date -d "${end_ts}" +%s 2>/dev/null || date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "${end_ts}" +%s)
      if [[ -n "${start_epoch}" && -n "${end_epoch}" ]]; then
        duration_minutes=$(python - <<'PY' "${start_epoch}" "${end_epoch}"
import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
print(f"{(end - start) / 60:.2f}")
PY
)
      fi
    fi
  fi

  if [[ -z "${duration_minutes}" ]]; then
    duration_minutes="N/A"
  fi

  if [[ "${status}" -eq 0 ]]; then
    echo "[$(timestamp)] >>> Completed ${job_name}"
    COMPLETED+=("${job_name}")
    printf "%s,%s,%s,success,%s,%s,%s\n" \
      "${job_name}" "${start_ts}" "${end_ts}" "${output_dir}" "${log_file}" "${duration_minutes}" \
      >> "${SUMMARY_FILE}"
  else
    echo "[$(timestamp)] >>> FAILED ${job_name} (see ${log_file})" >&2
    FAILED+=("${job_name}")
    printf "%s,%s,%s,failure,%s,%s,%s\n" \
      "${job_name}" "${start_ts}" "${end_ts}" "${output_dir}" "${log_file}" "${duration_minutes}" \
      >> "${SUMMARY_FILE}"
  fi
}

# ------------------------------------------------------------------------------
# AMLNet (production dataset) - scaled H100 runs
# ------------------------------------------------------------------------------
run_job \
  amlnet_h100_main \
  --config_file configs/h100_scaled_training.json \
  --device cuda

run_job \
  amlnet_h100_cvar95 \
  --config_file configs/h100_scaled_training.json \
  --device cuda \
  --risk-measure cvar_95 \
  --num-quantiles 192 \
  --episodes 12000 \
  --batch-size 256 \
  --epsilon-end 0.10 \
  --epsilon-decay 0.99985

run_job \
  amlnet_h100_mean \
  --config_file configs/h100_scaled_training.json \
  --device cuda \
  --risk-measure mean \
  --num-quantiles 128 \
  --episodes 8000 \
  --batch-size 256 \
  --epsilon-end 0.12 \
  --epsilon-decay 0.9998 \
  --no-guided-exploration

# ------------------------------------------------------------------------------
# Elliptic dataset - scaled H100 runs (with unlabeled nodes included)
# ------------------------------------------------------------------------------
ELLIPTIC_COMMON_ARGS=(
  --config_file configs/elliptic_rmganets.json
  --device cuda
  --include-unlabeled
  --agent-type qrdqn
  --n-step 5
  --batch-size 256
  --hidden-dims 512,512,256,128
  --prioritized-alpha 0.8
  --prioritized-beta 0.6
  --prioritized-beta-annealing 0.0005
  --epsilon-end 0.10
  --epsilon-decay 0.99985
  --max-steps 200
  --episodes 10000
  --guided-exploration
  --exploration-temperature 1.0
)

run_job \
  elliptic_h100_mean \
  "${ELLIPTIC_COMMON_ARGS[@]}" \
  --risk-measure mean \
  --num-quantiles 128 \
  --positive-fraction 0.30

run_job \
  elliptic_h100_cvar90 \
  "${ELLIPTIC_COMMON_ARGS[@]}" \
  --risk-measure cvar_90 \
  --num-quantiles 192 \
  --positive-fraction 0.35

run_job \
  elliptic_h100_cvar95 \
  "${ELLIPTIC_COMMON_ARGS[@]}" \
  --risk-measure cvar_95 \
  --num-quantiles 224 \
  --positive-fraction 0.40

echo
echo "================================================================================"
echo "H100 full suite finished - Summary"
echo "================================================================================"
echo "Outputs directory: ${RUN_DIR}"
echo "Summary CSV:       ${SUMMARY_FILE}"
echo
echo "Successful runs (${#COMPLETED[@]}): ${COMPLETED[*]:-none}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed runs (${#FAILED[@]}): ${FAILED[*]}" >&2
  exit 1
fi
echo "All runs completed successfully."
