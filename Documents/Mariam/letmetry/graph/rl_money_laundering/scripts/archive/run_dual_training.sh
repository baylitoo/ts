#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

python scripts/train_agent.py --config amlnet_full --device cuda --agent-type dqn --output_dir outputs/dqn_run > logs/dqn_run.log 2>&1 &
DQN_PID=$!
echo "Launched DQN training (PID $DQN_PID), logging to logs/dqn_run.log"

python scripts/train_agent.py --config amlnet_full --device cuda --agent-type qrdqn --output_dir outputs/qrdqn_run --num-quantiles 128 --n_step 5 --positive_fraction 0.25 > logs/qrdqn_run.log 2>&1 &
QR_PID=$!
echo "Launched QR-DQN training (PID $QR_PID), logging to logs/qrdqn_run.log"

wait $DQN_PID
wait $QR_PID

echo "Both training jobs completed."
