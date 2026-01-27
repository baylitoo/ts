#!/usr/bin/env bash
# Compare GraphSAGE vs GAT vs RMGANets architectures
# Trains all 3 in parallel and compares results

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_DIR="$ROOT_DIR/configs"
CONFIG_SAGE="$CONFIG_DIR/amlnet_sage.json"
CONFIG_GAT="$CONFIG_DIR/amlnet_gat.json"
CONFIG_RMGANETS="$CONFIG_DIR/amlnet_rmganets.json"

for cfg in "$CONFIG_SAGE" "$CONFIG_GAT" "$CONFIG_RMGANETS"; do
  if [ ! -f "$cfg" ]; then
    echo "Missing config file: $cfg" >&2
    exit 1
  fi
done

mkdir -p logs outputs

echo "========================================="
echo "GNN Architecture Comparison"
echo "========================================="
echo ""
echo "Training 3 architectures in parallel:"
echo "  1. GraphSAGE (baseline)"
echo "  2. GAT (baseline)"
echo "  3. RMGANets (new!)"
echo ""
echo "This will take ~30-60 minutes depending on hardware"
echo ""

LOG_SAGE="logs/sage_run.log"
LOG_GAT="logs/gat_run.log"
LOG_RMGANETS="logs/rmganets_run.log"

OUT_SAGE="outputs/amlnet_sage"
OUT_GAT="outputs/amlnet_gat"
OUT_RMGANETS="outputs/amlnet_rmganets"

mkdir -p "$OUT_SAGE" "$OUT_GAT" "$OUT_RMGANETS"

# Launch all 3 trainings in parallel
echo "[1/3] Launching GraphSAGE training..."
python scripts/train_agent.py --config_file "$CONFIG_SAGE" --device cpu > "$LOG_SAGE" 2>&1 &
SAGE_PID=$!
echo "  PID $SAGE_PID, logging to $LOG_SAGE"

echo "[2/3] Launching GAT training..."
python scripts/train_agent.py --config_file "$CONFIG_GAT" --device cpu > "$LOG_GAT" 2>&1 &
GAT_PID=$!
echo "  PID $GAT_PID, logging to $LOG_GAT"

echo "[3/3] Launching RMGANets training..."
python scripts/train_agent.py --config_file "$CONFIG_RMGANETS" --device cpu > "$LOG_RMGANETS" 2>&1 &
RMGANETS_PID=$!
echo "  PID $RMGANETS_PID, logging to $LOG_RMGANETS"

echo ""
echo "All trainings launched! Waiting for completion..."
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_SAGE"
echo "  tail -f $LOG_GAT"
echo "  tail -f $LOG_RMGANETS"
echo ""

# Wait for all to complete
wait $SAGE_PID
echo "[DONE] GraphSAGE training completed"

wait $GAT_PID
echo "[DONE] GAT training completed"

wait $RMGANETS_PID
echo "[DONE] RMGANets training completed"

echo ""
echo "========================================="
echo "All trainings completed!"
echo "========================================="
echo ""
echo "Results:"
echo "  GraphSAGE: $OUT_SAGE"
echo "  GAT:       $OUT_GAT"
echo "  RMGANets:  $OUT_RMGANETS"
echo ""
echo "Compare metrics with:"
echo "  python scripts/compare_results.py $OUT_SAGE $OUT_GAT $OUT_RMGANETS"
echo ""
