# Quick CPU Test Suite (PowerShell)
$ErrorActionPreference = "Stop"

# Navigate to project root
$ROOT_DIR = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT_DIR

# Test configuration
$TEST_EPISODES = 20
$TEST_MAX_EDGES = 5000
$TEST_FRAUD_RATIO = 0.05
$TEST_BATCH_SIZE = 16
$TEST_MAX_STEPS = 10

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = "logs/cpu_tests"
$OUT_DIR = "outputs/cpu_tests"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $Message"
}

function Test-Component {
    param(
        [string]$Name,
        [string[]]$Args
    )

    $log = "$LOG_DIR/${Name}_${TIMESTAMP}.log"
    $out = "$OUT_DIR/$Name"
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    Write-Log ">>> Testing $Name"
    Write-Log "    Log: $log"

    $cmd = @("scripts/train_agent.py", "--output_dir", $out, "--device", "cpu",
             "--episodes", $TEST_EPISODES, "--max_edges", $TEST_MAX_EDGES,
             "--fraud_ratio", $TEST_FRAUD_RATIO, "--batch-size", $TEST_BATCH_SIZE,
             "--max-steps", $TEST_MAX_STEPS) + $Args

    python @cmd > $log 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Log "✓ PASSED $Name"
        return $true
    }
    Write-Log "✗ FAILED $Name (see $log)"
    return $false
}

$failed = @()

Write-Log "TEST 1/7: Baseline DQN"
if (-not (Test-Component "test1_baseline_dqn" @("--config", "amlnet_full", "--agent-type", "dqn", "--epsilon-end", "0.01", "--epsilon-decay", "0.995", "--hidden-dims", "128,128,64", "--no-guided-exploration"))) { $failed += "test1" }

Write-Log "TEST 2/7: Enhanced DQN"
if (-not (Test-Component "test2_enhanced_dqn" @("--config", "amlnet_full", "--agent-type", "dqn", "--epsilon-end", "0.1", "--epsilon-decay", "0.998", "--hidden-dims", "128,128,64", "--no-guided-exploration"))) { $failed += "test2" }

Write-Log "TEST 3/7: DQN + Hints"
if (-not (Test-Component "test3_hints_guided" @("--config", "amlnet_full", "--agent-type", "dqn", "--epsilon-end", "0.1", "--epsilon-decay", "0.998", "--hidden-dims", "128,128,64", "--guided-exploration", "--exploration-temperature", "1.0"))) { $failed += "test3" }

Write-Log "TEST 4/7: QR-DQN Mean"
if (-not (Test-Component "test4_qrdqn_mean" @("--config", "amlnet_full", "--agent-type", "qrdqn", "--epsilon-end", "0.1", "--epsilon-decay", "0.998", "--hidden-dims", "256,128,64", "--num-quantiles", "32", "--n-step", "3", "--risk-measure", "mean", "--guided-exploration"))) { $failed += "test4" }

Write-Log "TEST 5/7: QR-DQN CVaR90"
if (-not (Test-Component "test5_qrdqn_cvar90" @("--config", "amlnet_full", "--agent-type", "qrdqn", "--epsilon-end", "0.1", "--epsilon-decay", "0.998", "--hidden-dims", "256,128,64", "--num-quantiles", "32", "--n-step", "3", "--risk-measure", "cvar_90", "--guided-exploration"))) { $failed += "test5" }

Write-Log "TEST 6/7: Dropout Fix"
if (-not (Test-Component "test6_dropout_fix" @("--config", "amlnet_full", "--agent-type", "dqn", "--epsilon-end", "0.0", "--hidden-dims", "128,64", "--no-guided-exploration"))) { $failed += "test6" }

Write-Log "TEST 7/7: N-Step Flush"
if (-not (Test-Component "test7_nstep_flush" @("--config", "amlnet_full", "--agent-type", "qrdqn", "--epsilon-end", "0.1", "--epsilon-decay", "0.998", "--hidden-dims", "128,64", "--num-quantiles", "16", "--n-step", "5", "--risk-measure", "mean"))) { $failed += "test7" }

Write-Log "=" * 70
Write-Log "CPU TEST SUITE COMPLETED"
Write-Log "=" * 70

if ($failed.Count -eq 0) {
    Write-Log "✓ ALL TESTS PASSED (7/7)"
    Write-Log "System is ready for H100!"
    exit 0
}

Write-Log "✗ FAILED: $($failed.Count)/7 tests"
Write-Log "Check logs in: $LOG_DIR"
exit 1
