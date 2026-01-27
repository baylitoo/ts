# Archived Scripts

This directory contains deprecated scripts that have been superseded by the unified training system.

## Why These Scripts Were Archived

### Superseded by `train_agent.py`
All agent-specific training scripts have been unified into the main `train_agent.py` entrypoint with `--agent-type` flag.

### Superseded by `final/h100_final_experiments.sh`
Ad-hoc experiment scripts have been replaced by the systematic comprehensive suite.

---

## Archived Scripts

### `train_agent_qrdqn.py`
**Archived**: 2025-10-23
**Reason**: Deprecated wrapper that forced `--agent-type qrdqn`
**Replacement**: `python scripts/train_agent.py --agent-type qrdqn`

**Original Purpose**: Convenience wrapper for QR-DQN training

**Why Deprecated**:
- Duplicates logic from `train_agent.py`
- Adds unnecessary complexity
- Not type-safe (uses dynamic import)

---

### `run_dual_training.sh`
**Archived**: 2025-10-23
**Reason**: Ad-hoc parallel training without systematic comparison
**Replacement**: `scripts/final/h100_final_experiments.sh`

**Original Purpose**: Run DQN and QR-DQN in parallel

**Why Deprecated**:
- No systematic hyperparameter sweep
- Missing enhanced exploration params
- No ablation studies
- Replaced by comprehensive experiment suite

---

### `benchmark_rmganets_multibranch.sh`
**Archived**: 2025-10-23
**Reason**: Single-purpose benchmark without comparison baselines
**Replacement**: `scripts/final/h100_final_experiments.sh` (Experiments 7-8)

**Original Purpose**: Benchmark RMGANets architecture

**Why Deprecated**:
- Missing comparison to DQN baselines
- No multi-branch loss variants
- Replaced by systematic architecture comparison

---

### `compare_gnn_architectures.sh`
**Archived**: 2025-10-23
**Reason**: Limited GNN comparison without RL integration
**Replacement**: Integrated into `final/h100_final_experiments.sh`

**Original Purpose**: Compare different GNN encoders

**Why Deprecated**:
- Only compared GNN architectures, not full RL systems
- Missing RMGANets variants
- No multi-branch loss comparison
- Replaced by comprehensive suite with GNN + RL + loss ablations

---

## Policy

**Retention**: Archived scripts are kept for historical reference but are not maintained.

**Usage**: Not recommended for production. Use replacement scripts instead.

**Removal**: May be permanently deleted in future cleanup if no longer referenced.

---

## Migration Guide

### From `train_agent_qrdqn.py`
```bash
# Old
python scripts/train_agent_qrdqn.py --config amlnet_full

# New
python scripts/train_agent.py --config amlnet_full --agent-type qrdqn
```

### From `run_dual_training.sh`
```bash
# Old
./scripts/run_dual_training.sh

# New - Comprehensive suite with 11 experiments
cd scripts/final
./h100_final_experiments.sh

# Or single experiment
python scripts/train_agent.py \
  --config amlnet_full \
  --agent-type qrdqn \
  --guided-exploration
```

### From `benchmark_rmganets_multibranch.sh`
```bash
# Old
./scripts/benchmark_rmganets_multibranch.sh

# New - RMGANets experiments
cd scripts/final
./h100_final_experiments.sh  # Runs experiments 7-8 with RMGANets
```

### From `compare_gnn_architectures.sh`
```bash
# Old
./scripts/compare_gnn_architectures.sh

# New - Comprehensive comparison
cd scripts/final
./h100_final_experiments.sh  # Compares all architectures systematically

# Then analyze results
python scripts/compare_models.py \
  --experiment-dir outputs/final/YYYYMMDD_HHMMSS/
```

---

**Archived**: 2025-10-23
**Reason**: Code consolidation and unification
