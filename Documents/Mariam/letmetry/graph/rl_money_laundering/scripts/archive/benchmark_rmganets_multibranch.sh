#!/usr/bin/env bash
# Benchmark RMGANets variants with and without multi-branch training

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_DIR="$ROOT_DIR/configs"
CONFIG_BASE="$CONFIG_DIR/amlnet_rmganets.json"
CONFIG_IMPROVED="$CONFIG_DIR/amlnet_rmganets_multibranch_improved.json"
CONFIG_PAPER="$CONFIG_DIR/amlnet_rmganets_multibranch_paper.json"

for cfg in "$CONFIG_BASE" "$CONFIG_IMPROVED" "$CONFIG_PAPER"; do
  if [ ! -f "$cfg" ]; then
    echo "Missing config file: $cfg" >&2
    exit 1
  fi
done

mkdir -p logs outputs

LOG_BASE="logs/rmganets_base.log"
LOG_IMPROVED="logs/rmganets_multibranch_improved.log"
LOG_PAPER="logs/rmganets_multibranch_paper.log"

OUT_BASE="outputs/amlnet_rmganets"
OUT_IMPROVED="outputs/amlnet_rmganets_multibranch_improved"
OUT_PAPER="outputs/amlnet_rmganets_multibranch_paper"

mkdir -p "$OUT_BASE" "$OUT_IMPROVED" "$OUT_PAPER"

echo "========================================="
echo "RMGANets Multi-Branch Benchmark"
echo "========================================="
echo "Running three configurations sequentially:"
echo "  1. Baseline RMGANets"
echo "  2. Multi-branch loss (improved variant)"
echo "  3. Multi-branch loss (paper variant)"
echo "========================================="

echo "[1/3] Baseline RMGANets"
python scripts/train_agent.py --config_file "$CONFIG_BASE" --device cpu | tee "$LOG_BASE"

echo "[2/3] Multi-branch (improved)"
python scripts/train_agent.py --config_file "$CONFIG_IMPROVED" --device cpu | tee "$LOG_IMPROVED"

echo "[3/3] Multi-branch (paper)"
python scripts/train_agent.py --config_file "$CONFIG_PAPER" --device cpu | tee "$LOG_PAPER"

echo "========================================="
echo "Benchmark complete!"
echo "Results stored in:"
echo "  $OUT_BASE"
echo "  $OUT_IMPROVED"
echo "  $OUT_PAPER"
echo "Logs:" 
echo "  $LOG_BASE"
echo "  $LOG_IMPROVED"
echo "  $LOG_PAPER"
echo "========================================="
