#!/usr/bin/env bash

################################################################################
# H100 FINAL EXPERIMENT SUITE
#
# Comprehensive training suite for AML detection with:
# - Critical bug fixes (Dropout, N-step buffer flush, gradient clipping)
# - Enhanced exploration (hint-guided, longer episodes, adaptive epsilon)
# - Full architecture sweep (DQN, QR-DQN, RMGANets + multi-branch loss)
# - AMLNet dataset focus (primary production dataset)
#
# Expected runtime on H100: ~8-12 hours for all experiments
# Date: 2025-10-23
################################################################################

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# Dataset paths
DATA_AML="data/amlnet/AMLNet_August_2025.csv"

if [[ ! -f "${DATA_AML}" ]]; then
  echo "[ERROR] Expected AMLNet dataset at ${DATA_AML}. Update the path or sync the dataset before running." >&2
  exit 1
fi

# Output directories
LOG_DIR="logs/final_experiments"
OUT_ROOT="outputs/final"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUT_ROOT}/${TIMESTAMP}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

# Helper functions
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

run_experiment() {
  local job_name="$1"; shift
  local output_dir="${RUN_DIR}/${job_name}"
  local log_file="${LOG_DIR}/${job_name}_${TIMESTAMP}.log"

  mkdir -p "${output_dir}"

  log ">>> Starting ${job_name}"
  log ">>> Logging to ${log_file}"
  log ">>> Output to ${output_dir}"

  # Run with proper error handling
  if uv run python scripts/train_agent.py \
    --output_dir "${output_dir}" \
    "$@" \
    > "${log_file}" 2>&1; then
    log "✓ COMPLETED ${job_name}"
    log "  Stats: ${output_dir}/training_stats.npz"
  else
    log "✗ FAILED ${job_name} (see ${log_file})"
    return 1
  fi
}

# Print experiment plan
cat <<EOF
================================================================================
H100 FINAL EXPERIMENT SUITE - $(timestamp)
================================================================================

Run directory: ${RUN_DIR}
Log directory: ${LOG_DIR}

Experiments planned:
  1. Baseline DQN (old params for comparison)
  2. Enhanced DQN (new exploration params)
  3. Enhanced DQN + Hint-Guided Exploration
  4. QR-DQN Risk-Neutral (mean)
  5. QR-DQN Risk-Averse (CVaR₉₀)
  6. QR-DQN Ultra-Conservative (CVaR₉₅)
  7. RMGANets + Multi-Branch (Improved Loss)
  8. RMGANets + Multi-Branch (Paper Loss)
  9. Ablation: No Hints
 10. Ablation: Temperature=0.5 (Greedy)
 11. Ablation: Temperature=2.0 (Exploratory)

================================================================================
EOF

# ============================================================================ #
# EXPERIMENT 1: Baseline DQN (OLD PARAMS - for comparison)
# ============================================================================ #
log "EXPERIMENT 1/11: DQN Baseline"
run_experiment \
  exp01_baseline_dqn_old \
  --config amlnet_full \
  --device cuda \
  --episodes 500 \
  --hidden-dims 512,512,256 \
  --epsilon-end 0.01 \
  --epsilon-decay 0.995 \
  --max-steps 20 \
  --no-guided-exploration \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# EXPERIMENT 2: Enhanced DQN (NEW PARAMS - longer epsilon, longer episodes)
# ============================================================================ #
log "EXPERIMENT 2/11: DQN"
run_experiment \
  exp02_enhanced_dqn \
  --config amlnet_full \
  --device cuda \
  --episodes 2000 \
  --hidden-dims 512,512,256 \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 100 \
  --no-guided-exploration \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# EXPERIMENT 3: Enhanced DQN + HINT-GUIDED EXPLORATION (FULL SYSTEM)
# ============================================================================ #
log "EXPERIMENT 3/11: DQN + Hints"
run_experiment \
  exp03_enhanced_dqn_hints \
  --config amlnet_full \
  --device cuda \
  --episodes 2000 \
  --hidden-dims 512,512,256 \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 100 \
  --guided-exploration \
  --exploration-temperature 1.0 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# EXPERIMENT 4: QR-DQN Risk-Neutral (mean aggregation)
# ============================================================================ #
log "EXPERIMENT 4/11: QR-DQN Mean"
run_experiment \
  exp04_qrdqn_mean \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 2500 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure mean \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 100 \
  --guided-exploration \
  --exploration-temperature 1.0 \
  --prioritized-alpha 0.7 \
  --prioritized-beta 0.5 \
  --prioritized-beta-annealing 0.001 \
  --positive-fraction 0.35 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# EXPERIMENT 5: QR-DQN Risk-Averse (CVaR₉₀ - conservative)
# ============================================================================ #
log "EXPERIMENT 5/11: QR-DQN CVaR90"
run_experiment \
  exp05_qrdqn_cvar90 \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 2500 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure cvar_90 \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 100 \
  --guided-exploration \
  --exploration-temperature 1.0 \
  --prioritized-alpha 0.7 \
  --prioritized-beta 0.5 \
  --prioritized-beta-annealing 0.001 \
  --positive-fraction 0.35 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# EXPERIMENT 6: QR-DQN Ultra-Conservative (CVaR₉₅ - worst-case optimization)
# ============================================================================ #
log "EXPERIMENT 6/11: QR-DQN CVaR95"
run_experiment \
  exp06_qrdqn_cvar95 \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 2500 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure cvar_95 \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 100 \
  --guided-exploration \
  --exploration-temperature 1.0 \
  --prioritized-alpha 0.7 \
  --prioritized-beta 0.5 \
  --prioritized-beta-annealing 0.001 \
  --positive-fraction 0.4 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# EXPERIMENT 7: RMGANets + Multi-Branch (IMPROVED LOSS)
# ============================================================================ #
log "EXPERIMENT 7/11: RMGANets Improved"
run_experiment \
  exp07_rmganets_improved \
  --config-file configs/amlnet_rmganets_multibranch_improved.json \
  --device cuda \
  --episodes 3000 \
  --agent-type qrdqn \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure mean \
  --hidden-dims 512,512,256 \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 150 \
  --guided-exploration \
  --exploration-temperature 1.0

# ============================================================================ #
# EXPERIMENT 8: RMGANets + Multi-Branch (PAPER LOSS)
# ============================================================================ #
log "EXPERIMENT 8/11: RMGANets Paper"
run_experiment \
  exp08_rmganets_paper \
  --config-file configs/amlnet_rmganets_multibranch_paper.json \
  --device cuda \
  --episodes 3000 \
  --agent-type qrdqn \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure mean \
  --hidden-dims 512,512,256 \
  --epsilon-end 0.15 \
  --epsilon-decay 0.999 \
  --max-steps 150 \
  --guided-exploration \
  --exploration-temperature 1.0

# ============================================================================ #
# ABLATION STUDIES
# ============================================================================ #

# ABLATION 1: Verify hints disabled works correctly
log "EXPERIMENT 9/11: QR-DQN No Hints"
run_experiment \
  exp09_ablation_no_hints \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 800 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure mean \
  --epsilon-end 0.1 \
  --epsilon-decay 0.9985 \
  --max-steps 50 \
  --no-guided-exploration \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ABLATION 2: Temperature = 0.5 (more greedy, focuses on highest risk)
log "EXPERIMENT 10/11: QR-DQN Temp0.5"
run_experiment \
  exp10_ablation_temp05 \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 800 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure mean \
  --epsilon-end 0.1 \
  --epsilon-decay 0.9985 \
  --max-steps 50 \
  --guided-exploration \
  --exploration-temperature 0.5 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ABLATION 3: Temperature = 2.0 (more exploratory, diverse coverage)
log "EXPERIMENT 11/11: QR-DQN Temp2.0"
run_experiment \
  exp11_ablation_temp20 \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --episodes 800 \
  --hidden-dims 768,512,256 \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure mean \
  --epsilon-end 0.1 \
  --epsilon-decay 0.9985 \
  --max-steps 50 \
  --guided-exploration \
  --exploration-temperature 2.0 \
  --max_edges 500000 \
  --fraud_ratio 0.02

# ============================================================================ #
# COMPLETION SUMMARY
# ============================================================================ #

log "================================================================================"
log "H100 FINAL EXPERIMENT SUITE COMPLETED"
log "================================================================================"
log ""
log "Results directory: ${RUN_DIR}"
log "Logs directory:    ${LOG_DIR}"
log ""
log "Experiment summary:"
log "  Baseline (old params):       exp01_baseline_dqn_old"
log "  Enhanced (no hints):         exp02_enhanced_dqn"
log "  Enhanced + Hints (BEST):     exp03_enhanced_dqn_hints"
log "  QR-DQN Risk-Neutral:         exp04_qrdqn_mean"
log "  QR-DQN Risk-Averse:          exp05_qrdqn_cvar90"
log "  QR-DQN Ultra-Conservative:   exp06_qrdqn_cvar95"
log "  RMGANets (Improved):         exp07_rmganets_improved"
log "  RMGANets (Paper):            exp08_rmganets_paper"
log "  Ablation (No Hints):         exp09_ablation_no_hints"
log "  Ablation (Temp 0.5):         exp10_ablation_temp05"
log "  Ablation (Temp 2.0):         exp11_ablation_temp20"
log ""
log "Next steps:"
log "  1. Analyze results:  python scripts/analyze_experiments.py ${RUN_DIR}"
log "  2. Compare metrics:  tensorboard --logdir ${RUN_DIR}"
log "  3. Best model:       Check exp03_enhanced_dqn_hints for production deployment"
log ""
log "================================================================================"
