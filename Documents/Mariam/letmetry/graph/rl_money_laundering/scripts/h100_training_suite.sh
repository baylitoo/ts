#!/usr/bin/env bash

# End-to-end H100 training sweep across all architectures and agents.
# Runs DQN, QR-DQN, and RMGANETS multi-branch variants on the full AMLNet and
# Elliptic datasets, validating loss functions by executing actual training
# loops. Each run writes its own output and log directory so results are easy
# to inspect post-execution.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

DATA_AML="data/amlnet/AMLNet_August_2025.csv"
DATA_ELLIPTIC="data/elliptic"

if [[ ! -f "${DATA_AML}" ]]; then
  echo "[ERROR] Expected AMLNet dataset at ${DATA_AML}. Update the path or sync the dataset before running." >&2
  exit 1
fi

if [[ ! -d "${DATA_ELLIPTIC}" ]]; then
  echo "[ERROR] Expected Elliptic dataset directory at ${DATA_ELLIPTIC}. Update the path or sync the dataset before running." >&2
  exit 1
fi

LOG_DIR="logs/h100_suite"
OUT_ROOT="outputs/h100"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_job() {
  local job_name="$1"; shift
  local output_dir="${OUT_ROOT}/${job_name}"
  local log_file="${LOG_DIR}/${job_name}.log"

  mkdir -p "${output_dir}"

  echo "[$(timestamp)] >>> Starting ${job_name}"
  echo "[$(timestamp)] >>> Logging to ${log_file}"

  uv run python scripts/train_agent.py \
    --output_dir "${output_dir}" \
    "$@" \
    > "${log_file}" 2>&1

  echo "[$(timestamp)] >>> Completed ${job_name}"
  echo "[$(timestamp)] >>> Stats stored in ${output_dir}/training_stats.npz"
}

# --------------------------------------------------------------------------- #
# 1. AMLNet baseline DQN (wide hidden dims, large replay)                     #
# --------------------------------------------------------------------------- #
run_job \
  amlnet_dqn_wide \
  --config amlnet_full \
  --device cuda \
  --episodes 400 \
  --hidden-dims 512,512,256 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# --------------------------------------------------------------------------- #
# 2. AMLNet QR-DQN distributional agent (risk-aware, prioritized N-step)      #
# --------------------------------------------------------------------------- #
run_job \
  amlnet_qrdqn_riskaware \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 400 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure cvar_90 \
  --prioritized-alpha 0.7 \
  --prioritized-beta 0.5 \
  --prioritized-beta-annealing 0.002 \
  --positive-fraction 0.35 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# --------------------------------------------------------------------------- #
# 3. AMLNet RMGANETS + multi-branch (improved loss)                           #
#    Validates fusion of Att-GCM / To-GCM / Diff-GCM reasoning modules        #
# --------------------------------------------------------------------------- #
run_job \
  amlnet_rmganets_improved \
  --config_file configs/amlnet_rmganets_multibranch_improved.json \
  --device cuda \
  --episodes 300 \
  --agent-type qrdqn \
  --num-quantiles 96 \
  --n-step 5 \
  --risk-measure mean \
  --hidden-dims 512,512,256

# --------------------------------------------------------------------------- #
# 4. AMLNet RMGANETS + multi-branch (paper loss variant)                      #
# --------------------------------------------------------------------------- #
run_job \
  amlnet_rmganets_paper \
  --config_file configs/amlnet_rmganets_multibranch_paper.json \
  --device cuda \
  --episodes 300 \
  --agent-type qrdqn \
  --num-quantiles 96 \
  --n-step 5 \
  --risk-measure mean \
  --hidden-dims 512,512,256

# --------------------------------------------------------------------------- #
# 5. Elliptic dataset DQN baseline (Graph Attention, CPU fallback for data)   #
# --------------------------------------------------------------------------- #
run_job \
  elliptic_dqn_baseline \
  --config elliptic \
  --device cuda \
  --episodes 300 \
  --hidden-dims 512,256,128 \
  --disable-double-dqn

# --------------------------------------------------------------------------- #
# 6. Elliptic QR-DQN (risk-aware tail emphasis)                               #
# --------------------------------------------------------------------------- #
run_job \
  elliptic_qrdqn_tailrisk \
  --config elliptic \
  --device cuda \
  --agent-type qrdqn \
  --episodes 300 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure cvar_95 \
  --prioritized-alpha 0.7 \
  --prioritized-beta 0.5 \
  --prioritized-beta-annealing 0.002 \
  --positive-fraction 0.4

echo "[$(timestamp)] === H100 training sweep finished successfully ==="
echo "Logs:     ${LOG_DIR}"
echo "Outputs:  ${OUT_ROOT}"
