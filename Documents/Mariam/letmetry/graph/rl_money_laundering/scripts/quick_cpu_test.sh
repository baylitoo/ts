#!/usr/bin/env bash

################################################################################
# QUICK CPU TEST SUITE
#
# Fast smoke tests on CPU with reduced data to verify all components work
# before launching expensive H100 runs.
#
# Runtime: ~5-10 minutes on modern CPU
# Purpose: Integration testing, bug detection, sanity checks
#
# Usage:
#   ./scripts/quick_cpu_test.sh
#
# Or test specific component:
#   ./scripts/quick_cpu_test.sh dqn          # Test DQN only
#   ./scripts/quick_cpu_test.sh qrdqn        # Test QR-DQN only
#   ./scripts/quick_cpu_test.sh rmganets     # Test RMGANets only
#   ./scripts/quick_cpu_test.sh hints        # Test hint-guided exploration
################################################################################

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Test configuration (reduced for speed)
TEST_EPISODES=20
TEST_MAX_EDGES=5000
TEST_FRAUD_RATIO=0.05
TEST_BATCH_SIZE=16
TEST_MAX_STEPS=10

LOG_DIR="logs/cpu_tests"
OUT_DIR="outputs/cpu_tests"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

run_test() {
  local test_name="$1"; shift
  local log_file="${LOG_DIR}/${test_name}_${TIMESTAMP}.log"
  local output_dir="${OUT_DIR}/${test_name}"

  mkdir -p "${output_dir}"

  log ">>> Testing ${test_name}"
  log "    Log: ${log_file}"
  log "    Output: ${output_dir}"

  if python scripts/train_agent.py \
    --output_dir "${output_dir}" \
    --device cpu \
    --episodes ${TEST_EPISODES} \
    --max-edges ${TEST_MAX_EDGES} \
    --fraud-ratio ${TEST_FRAUD_RATIO} \
    --batch-size ${TEST_BATCH_SIZE} \
    --max-steps ${TEST_MAX_STEPS} \
    "$@" \
    > "${log_file}" 2>&1; then
    log "✓ PASSED ${test_name}"
    return 0
  else
    log "✗ FAILED ${test_name} (see ${log_file})"
    return 1
  fi
}

# Determine which tests to run
COMPONENT="${1:-all}"

run_all_tests() {
  local failed_tests=()

  # Test 1: Baseline DQN (old params)
  log "TEST 1/7: Baseline DQN (old exploration params)"
  if ! run_test test1_baseline_dqn \
    --config amlnet_full \
    --agent-type dqn \
    --epsilon-end 0.01 \
    --epsilon-decay 0.995 \
    --hidden-dims 128,128,64 \
    --no-guided-exploration; then
    failed_tests+=("test1_baseline_dqn")
  fi

  # Test 2: Enhanced DQN (new params, no hints)
  log "TEST 2/7: Enhanced DQN (new exploration, no hints)"
  if ! run_test test2_enhanced_dqn \
    --config amlnet_full \
    --agent-type dqn \
    --epsilon-end 0.1 \
    --epsilon-decay 0.998 \
    --hidden-dims 128,128,64 \
    --no-guided-exploration; then
    failed_tests+=("test2_enhanced_dqn")
  fi

  # Test 3: Enhanced DQN + Hint-Guided Exploration
  log "TEST 3/7: Enhanced DQN + Hint-Guided Exploration"
  if ! run_test test3_hints_guided \
    --config amlnet_full \
    --agent-type dqn \
    --epsilon-end 0.1 \
    --epsilon-decay 0.998 \
    --hidden-dims 128,128,64 \
    --guided-exploration \
    --exploration-temperature 1.0; then
    failed_tests+=("test3_hints_guided")
  fi

  # Test 4: QR-DQN Mean
  log "TEST 4/7: QR-DQN Risk-Neutral"
  if ! run_test test4_qrdqn_mean \
    --config amlnet_full \
    --agent-type qrdqn \
    --epsilon-end 0.1 \
    --epsilon-decay 0.998 \
    --hidden-dims 256,128,64 \
    --num-quantiles 32 \
    --n-step 3 \
    --risk-measure mean \
    --guided-exploration; then
    failed_tests+=("test4_qrdqn_mean")
  fi

  # Test 5: QR-DQN CVaR₉₀
  log "TEST 5/7: QR-DQN Risk-Averse (CVaR₉₀)"
  if ! run_test test5_qrdqn_cvar90 \
    --config amlnet_full \
    --agent-type qrdqn \
    --epsilon-end 0.1 \
    --epsilon-decay 0.998 \
    --hidden-dims 256,128,64 \
    --num-quantiles 32 \
    --n-step 3 \
    --risk-measure cvar_90 \
    --guided-exploration; then
    failed_tests+=("test5_qrdqn_cvar90")
  fi

  # Test 6: Dropout fix verification (should be deterministic)
  log "TEST 6/7: Dropout Fix Verification"
  if ! run_test test6_dropout_fix \
    --config amlnet_full \
    --agent-type dqn \
    --epsilon-end 0.0 \
    --hidden-dims 128,64 \
    --no-guided-exploration; then
    failed_tests+=("test6_dropout_fix")
  fi

  # Test 7: N-step buffer flush (verify no data loss)
  log "TEST 7/7: N-Step Buffer Flush"
  if ! run_test test7_nstep_flush \
    --config amlnet_full \
    --agent-type qrdqn \
    --epsilon-end 0.1 \
    --epsilon-decay 0.998 \
    --hidden-dims 128,64 \
    --num-quantiles 16 \
    --n-step 5 \
    --risk-measure mean; then
    failed_tests+=("test7_nstep_flush")
  fi

  # Print summary
  log "=========================================================================="
  log "CPU TEST SUITE COMPLETED"
  log "=========================================================================="
  log ""
  if [ ${#failed_tests[@]} -eq 0 ]; then
    log "✓ ALL TESTS PASSED (7/7)"
    log ""
    log "Results directory: ${OUT_DIR}"
    log "Logs directory:    ${LOG_DIR}"
    log ""
    log "System is ready for H100 production runs!"
    return 0
  else
    log "✗ SOME TESTS FAILED (${#failed_tests[@]}/7)"
    log ""
    log "Failed tests:"
    for test in "${failed_tests[@]}"; do
      log "  - ${test}"
    done
    log ""
    log "Check logs in: ${LOG_DIR}"
    log ""
    log "Fix issues before running H100 suite."
    return 1
  fi
}

# Run specific component or all tests
case "${COMPONENT}" in
  dqn)
    log "Running DQN-only tests..."
    run_test test_dqn_quick \
      --config amlnet_full \
      --agent-type dqn \
      --epsilon-end 0.1 \
      --epsilon-decay 0.998 \
      --hidden-dims 128,64 \
      --guided-exploration
    ;;

  qrdqn)
    log "Running QR-DQN-only tests..."
    run_test test_qrdqn_quick \
      --config amlnet_full \
      --agent-type qrdqn \
      --epsilon-end 0.1 \
      --epsilon-decay 0.998 \
      --hidden-dims 128,64 \
      --num-quantiles 16 \
      --n-step 3 \
      --risk-measure mean \
      --guided-exploration
    ;;

  rmganets)
    log "Running RMGANets-only tests..."
    if [ ! -f "configs/amlnet_rmganets_multibranch_improved.json" ]; then
      log "ERROR: RMGANets config not found at configs/amlnet_rmganets_multibranch_improved.json"
      exit 1
    fi
    run_test test_rmganets_quick \
      --config-file configs/amlnet_rmganets_multibranch_improved.json \
      --agent-type qrdqn \
      --epsilon-end 0.1 \
      --epsilon-decay 0.998 \
      --hidden-dims 128,64 \
      --num-quantiles 16 \
      --n-step 3
    ;;

  hints)
    log "Running hint-guided exploration test..."
    run_test test_hints_only \
      --config amlnet_full \
      --agent-type dqn \
      --epsilon-end 0.5 \
      --epsilon-decay 1.0 \
      --hidden-dims 128,64 \
      --guided-exploration \
      --exploration-temperature 1.0
    ;;

  all)
    run_all_tests
    ;;

  *)
    log "ERROR: Unknown component '${COMPONENT}'"
    log "Usage: $0 [all|dqn|qrdqn|rmganets|hints]"
    exit 1
    ;;
esac
