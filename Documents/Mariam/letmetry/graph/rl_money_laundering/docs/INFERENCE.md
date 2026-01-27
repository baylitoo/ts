# Inference Guide: Running Trained RL-AML Models

This guide explains how to use trained RL-based AML detection models for inference on new data, including support for both AMLNet and Elliptic Bitcoin datasets.

## Quick Start

```bash
# Simple inference with trained model
python scripts/simple_inference.py outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt 20

# Advanced inference with detailed results (JSON output)
python scripts/run_inference.py \
  --checkpoint outputs/checkpoint_ep2300.pt \
  --config configs/amlnet_rmganets_fixed.json \
  --dataset amlnet \
  --output results/inference.json \
  --num-episodes 100 \
  --max-steps 200
```

## Scripts Overview

### 1. `simple_inference.py` - Quick Evaluation

**Purpose**: Fast inference with minimal setup
**Use when**: You want to quickly evaluate a trained checkpoint

**Features**:
- Automatically loads config from checkpoint directory
- Runs both fraud-seeded and random-start episodes
- Prints summary statistics (confusion matrix, F1, precision, recall)
- No dependencies on external configs

**Usage**:
```bash
python scripts/simple_inference.py <checkpoint_path> [num_episodes]
```

**Example**:
```bash
python scripts/simple_inference.py outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt 50
```

**Output**:
```
================================================================================
INFERENCE SUMMARY
================================================================================

Fraud-Seeded (25 episodes):
  TP=23 FP=1 FN=1 TN=0
  Precision: 95.8%
  Recall:    95.8%
  F1 Score:  95.8%
  Avg Reward: 15.23
  Avg Steps:  47.2

Random-Start (25 episodes):
  TP=8 FP=2 FN=3 TN=12
  Precision: 80.0%
  Recall:    72.7%
  F1 Score:  76.2%
  Avg Reward: 8.45
  Avg Steps:  52.8
================================================================================
```

---

### 2. `run_inference.py` - Comprehensive Analysis

**Purpose**: Detailed inference with full episode tracking and JSON export
**Use when**: You need per-episode details, custom datasets, or automated analysis

**Features**:
- Per-episode results (visited nodes, actions, rewards)
- Supports both AMLNet and Elliptic datasets
- Exports detailed JSON with full episode traces
- Configurable episode count and step limits
- Temporal grouping for Elliptic dataset

**Usage**:
```bash
python scripts/run_inference.py \
  --checkpoint <path_to_checkpoint.pt> \
  --config <path_to_config.json> \
  --dataset <amlnet|elliptic> \
  [--elliptic-dir <elliptic_data_dir>] \
  [--output <output.json>] \
  [--num-episodes 100] \
  [--max-steps 200]
```

**AMLNet Example**:
```bash
python scripts/run_inference.py \
  --checkpoint outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt \
  --config configs/amlnet_rmganets_fixed.json \
  --dataset amlnet \
  --output results/cvar_inference.json \
  --num-episodes 100
```

**Elliptic Example**:
```bash
python scripts/run_inference.py \
  --checkpoint outputs/elliptic_model.pt \
  --config configs/elliptic_config.json \
  --dataset elliptic \
  --elliptic-dir data/elliptic \
  --output results/elliptic_inference.json \
  --num-episodes 200
```

**JSON Output Structure**:
```json
{
  "checkpoint": "outputs/.../checkpoint_ep2300.pt",
  "config": "configs/amlnet_rmganets_fixed.json",
  "dataset": "amlnet",
  "num_episodes": 100,
  "max_steps": 200,
  "summary": {
    "fraud_seeded": {
      "total_tp": 45,
      "total_fp": 3,
      "total_fn": 2,
      "total_tn": 0,
      "precision": 0.9375,
      "recall": 0.9565,
      "f1": 0.9469
    },
    "random_start": { ... }
  },
  "episodes": [
    {
      "episode_id": 0,
      "episode_type": "fraud_seeded",
      "visited_nodes": ["node_123", "node_456", ...],
      "actions_taken": ["MOVE_FORWARD", "MOVE_FORWARD", "FLAG"],
      "rewards": [0.05, 0.1, 10.0],
      "total_reward": 15.45,
      "steps": 48,
      "fraud_detected": true,
      "flag_node": "node_789",
      "true_fraud_nodes": ["node_789", "node_012"],
      "confusion": {"TP": 1, "FP": 0, "TN": 0, "FN": 0},
      "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    },
    ...
  ]
}
```

---

## Dataset Support

### AMLNet Dataset
- **Features**: 20-dimensional node features (temporal, network statistics)
- **Graph size**: Varies (10k-100k nodes depending on configuration)
- **Fraud rate**: ~3-5%
- **Use case**: Synthetic but realistic money laundering patterns

### Elliptic Bitcoin Dataset
- **Features**: 166-dimensional node features (94 local + 72 aggregated)
  - Feature 1: Time step (1-49, 2-week intervals)
  - Features 2-94: Local transaction features (inputs/outputs, fees, volumes)
  - Features 95-166: Aggregated neighbor features (max, min, std, correlation)
- **Graph size**: 203,769 nodes, 234,355 edges
- **Fraud rate**: 2% (4,545 illicit nodes)
- **Time steps**: 49 (evenly spaced, ~2 weeks apart)
- **Use case**: Real Bitcoin blockchain data, industry-standard benchmark

**Elliptic Integration**:
The system automatically:
1. Loads all 166 features from CSV files
2. Applies feature engineering (temporal context, standardization)
3. Groups fraud subgraphs by time step for temporal coherence
4. Attaches processed features to graph nodes
5. Uses existing `EllipticLoader` with full feature pipeline

**Advantage**: More features = better fraud detection signal. The GNN encoder compresses 166 features → 64-dimensional embeddings, preserving the most informative patterns.

---

## Understanding Evaluation Metrics

### Confusion Matrix
- **TP (True Positives)**: Fraud present & flagged ✓
- **FP (False Positives)**: No fraud & flagged ✗
- **FN (False Negatives)**: Fraud present & not flagged ✗
- **TN (True Negatives)**: No fraud & not flagged ✓

### Performance Metrics
- **Precision** = TP / (TP + FP) - How accurate are flags?
- **Recall** = TP / (TP + FN) - How many frauds are caught?
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean
- **Specificity** = TN / (TN + FP) - How well do we avoid false alarms?

### Episode Types
1. **Fraud-Seeded**: Agent starts in fraud-containing subgraphs
   - Tests detection ability when fraud is present
   - Higher expected TP rate
   - Measures recall primarily

2. **Random-Start**: Agent starts anywhere in full graph
   - Tests real-world scenario (natural fraud distribution)
   - Tests precision (avoiding false alarms)
   - More realistic performance metric

---

## Risk Measures (QR-DQN)

When using QR-DQN agents, the risk measure affects inference behavior:

### `risk_measure: "mean"` (Default)
- Uses average over all 64 quantiles
- **Balanced** precision/recall
- Best for general-purpose detection

### `risk_measure: "cvar_90"` (Risk-Averse)
- Uses worst 10% quantiles (most pessimistic scenarios)
- **Conservative flagging** → higher precision, lower recall
- Minimizes false positives (fewer false alarms)
- **Best for production** where false alarms are costly

### `risk_measure: "cvar_95"` (Very Risk-Averse)
- Uses worst 5% quantiles
- **Extremely conservative**
- Only flags when very confident
- Ultra-low false positive rate

**Example**: CVaR Risk-Averse (Config #5) showed:
- Multiple 100% F1 evaluations
- Zero false positives in many episodes
- Slightly lower recall but compensated by perfect precision

---

## Model Checkpoints

Checkpoints are saved every 100 episodes by default:
```
outputs/
└── cpu_tests/
    └── 05_CVAR_RISK_AVERSE/
        ├── config.json                    # Experiment configuration
        ├── checkpoints/
        │   ├── checkpoint_ep100.pt
        │   ├── checkpoint_ep200.pt
        │   ├── ...
        │   └── checkpoint_ep2300.pt       # Latest checkpoint
        └── training_stats.npz             # Training metrics
```

**Checkpoint contents**:
- `q_network_state_dict`: Q-network weights
- `target_network_state_dict`: Target network weights
- `optimizer_state_dict`: Optimizer state
- `epsilon`: Current exploration rate
- `episode`: Episode number
- `metrics`: Optional training metrics

---

## Common Use Cases

### 1. Compare Multiple Checkpoints
```bash
for ep in 1000 1500 2000 2300; do
  python scripts/simple_inference.py \
    outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep${ep}.pt 50 \
    > results/inference_ep${ep}.txt
done
```

### 2. Benchmark on Elliptic
```bash
# Download Elliptic dataset first
# https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

python scripts/run_inference.py \
  --checkpoint outputs/best_model.pt \
  --config configs/elliptic_config.json \
  --dataset elliptic \
  --elliptic-dir data/elliptic \
  --num-episodes 200 \
  --output results/elliptic_benchmark.json
```

### 3. Production Deployment Simulation
```bash
# Use CVaR risk-averse model for low false positive rate
python scripts/simple_inference.py \
  outputs/cvar_model/checkpoints/checkpoint_final.pt 1000
```

### 4. Analyze Per-Episode Behavior
```bash
python scripts/run_inference.py \
  --checkpoint outputs/model.pt \
  --config configs/config.json \
  --dataset amlnet \
  --num-episodes 50 \
  --output results/detailed.json

# Then analyze JSON:
python -c "
import json
with open('results/detailed.json') as f:
    data = json.load(f)

for ep in data['episodes']:
    if ep['fraud_detected']:
        print(f\"Episode {ep['episode_id']}: Flagged at node {ep['flag_node']} after {ep['steps']} steps\")
"
```

---

## Troubleshooting

### Memory Issues
If you encounter segfaults or out-of-memory errors:
```bash
# Use simple_inference.py (lighter memory footprint)
python scripts/simple_inference.py <checkpoint> 20

# Or limit episodes
python scripts/run_inference.py ... --num-episodes 20
```

### Checkpoint Not Found
Ensure you provide the full path:
```bash
# Correct
python scripts/simple_inference.py outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt

# Wrong
python scripts/simple_inference.py checkpoint_ep2300.pt
```

### Feature Dimension Mismatch
This occurs when the checkpoint was trained with different GNN config:
- Check `config.json` in checkpoint directory
- Ensure `gnn.hidden_channels`, `gnn.num_layers`, `gnn.heads` match

### Device Mismatch
```bash
# Force CPU inference
export CUDA_VISIBLE_DEVICES=""
python scripts/simple_inference.py <checkpoint> 20
```

---

## Next Steps

1. **Download Elliptic Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
2. **Create Elliptic Config**: See `configs/elliptic_config.json` template
3. **Train on Elliptic**: `python scripts/train_agent.py --config elliptic --episodes 3000`
4. **Benchmark Results**: Compare AMLNet vs Elliptic performance

---

## Performance Expectations

### AMLNet (Synthetic)
- **Training F1**: 85-95% (fraud-seeded)
- **Random F1**: 70-85%
- **Training time**: ~2-3 hours (2400 episodes, CPU)

### Elliptic (Real Bitcoin Data)
- **Expected F1**: 75-90% (benchmark: 85% with XGBoost)
- **Challenge**: Lower fraud rate (2% vs 3-5%), larger scale
- **Advantage**: 166 features provide rich signal
- **Training time**: ~4-6 hours (3000 episodes, CPU)

### CVaR Risk-Averse
- **Precision**: 90-100% (minimal false positives)
- **Recall**: 80-95% (trades coverage for accuracy)
- **Best for**: Production deployment, low false alarm tolerance
