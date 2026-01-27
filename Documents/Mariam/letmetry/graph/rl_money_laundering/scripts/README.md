# Scripts Directory

## Active Scripts (Production-Ready)

### Training & Experiments

#### `train_agent.py` ⭐
**Main training entrypoint** - Unified CLI for all agents and configurations.

```bash
# Basic DQN training
python scripts/train_agent.py --config amlnet_full --device cuda

# QR-DQN with risk-averse policy
python scripts/train_agent.py \
  --config amlnet_full \
  --device cuda \
  --agent-type qrdqn \
  --num-quantiles 128 \
  --n-step 5 \
  --risk-measure cvar_90

# Enhanced exploration + hints
python scripts/train_agent.py \
  --config amlnet_full \
  --device cuda \
  --epsilon-end 0.1 \
  --epsilon-decay 0.998 \
  --max-steps 50 \
  --guided-exploration \
  --exploration-temperature 1.0
```

**Key Features**:
- ✅ Supports DQN and QR-DQN agents
- ✅ Hint-guided exploration
- ✅ All bug fixes included
- ✅ Full checkpoint management
- ✅ MLflow logging ready

---

### Testing & Validation

#### `test_components_standalone.py` ⭐
**RLlib component tests (No Ray required)** - Test GraphSpace and batch utilities.

```bash
python scripts/test_components_standalone.py
```

**What it tests**:
- ✅ GraphSpace initialization and sampling
- ✅ GraphSpace observation validation
- ✅ PyG batch processing utilities
- ✅ First node extraction from batched graphs
- ✅ GraphBatchCollator workflow
- ✅ Edge cases (empty graphs, single graph)

**Requirements**: PyTorch, PyTorch Geometric, NumPy, Gymnasium (no Ray needed)

**Runtime**: ~5 seconds

---

#### `test_rllib_training.py` ⭐
**Full RLlib training pipeline test** - Comprehensive workflow validation.

```bash
python scripts/test_rllib_training.py
```

**What it tests**:
- ✅ Environment registration and creation
- ✅ GNN RLModule initialization
- ✅ Complete DQN training loop (3 iterations)
- ✅ Fraud-aware replay buffer integration
- ✅ Custom metrics and callbacks
- ✅ Checkpoint save/load
- ✅ Synthetic graph data generation

**Requirements**: Ray RLlib >= 2.40.0, Python 3.9-3.12 (not 3.13 on Windows)

**Runtime**: ~2-3 minutes

**Note**: Ray doesn't support Python 3.13 on Windows yet. Use Python 3.11 or run on Linux/Mac.

---

#### `quick_cpu_test.sh` ⭐
**Fast smoke tests on CPU** - Verify all components before H100 runs.

```bash
# Run all tests (~5-10 minutes)
./scripts/quick_cpu_test.sh

# Test specific component
./scripts/quick_cpu_test.sh dqn          # DQN only
./scripts/quick_cpu_test.sh qrdqn        # QR-DQN only
./scripts/quick_cpu_test.sh rmganets     # RMGANets only
./scripts/quick_cpu_test.sh hints        # Hint-guided exploration
```

**What it tests**:
- ✅ Baseline DQN (old params)
- ✅ Enhanced DQN (new params)
- ✅ Hint-guided exploration
- ✅ QR-DQN (mean & CVaR₉₀)
- ✅ Dropout fix verification
- ✅ N-step buffer flush

**Configuration**: Uses reduced data (5K edges, 20 episodes) for speed.

---

### Production Suites

#### `final/h100_final_experiments.sh` ⭐
**Comprehensive H100 experiment suite** - Production-ready full training.

```bash
cd scripts/final
./h100_final_experiments.sh
```

**11 Experiments**:
1. Baseline DQN (old params)
2. Enhanced DQN (no hints)
3. Enhanced DQN + Hints (PRODUCTION) ⭐
4. QR-DQN Mean
5. QR-DQN CVaR₉₀
6. QR-DQN CVaR₉₅
7. RMGANets + Improved Loss (RESEARCH) ⭐
8. RMGANets + Paper Loss
9. Ablation: No Hints
10. Ablation: Temperature 0.5
11. Ablation: Temperature 2.0

**Runtime**: 8-12 hours on H100

See [final/README.md](final/README.md) for details.

---

#### `h100_training_suite.sh`
**Original H100 suite** - Legacy version, superseded by `final/h100_final_experiments.sh`.

**Status**: Still functional but missing:
- ❌ Enhanced exploration params
- ❌ Hint-guided exploration
- ❌ Recent bug fixes
- ❌ Ablation studies

**Recommendation**: Use `final/h100_final_experiments.sh` instead.

---

### Analysis & Utilities

#### `compare_models.py`
**Model comparison and visualization** - Analyze experiment results.

```bash
# Compare two models
python scripts/compare_models.py \
  --model1 outputs/final/exp03_enhanced_dqn_hints/best_model.pth \
  --model2 outputs/final/exp07_rmganets_improved/best_model.pth \
  --output comparison_report.pdf

# Analyze full experiment directory
python scripts/compare_models.py \
  --experiment-dir outputs/final/20251023_120000/ \
  --output full_analysis.pdf
```

#### `precompute_graph_features.py`
**Feature caching utility** - Pre-compute graph embeddings for faster training.

```bash
# Precompute for AMLNet
python scripts/precompute_graph_features.py \
  --dataset amlnet \
  --output data/cached_features/

# Precompute for Elliptic
python scripts/precompute_graph_features.py \
  --dataset elliptic \
  --output data/cached_features/
```

**Note**: Currently optional, may improve training speed by 10-20%.

---

## Archived Scripts

Legacy scripts moved to `archive/` subdirectory:

### `archive/train_agent_qrdqn.py`
**Status**: Deprecated wrapper
**Reason**: Unified into `train_agent.py --agent-type qrdqn`

### `archive/run_dual_training.sh`
**Status**: Deprecated parallel runner
**Reason**: Use `final/h100_final_experiments.sh` for systematic experiments

### `archive/benchmark_rmganets_multibranch.sh`
**Status**: Deprecated benchmark
**Reason**: Replaced by comprehensive suite with ablations

### `archive/compare_gnn_architectures.sh`
**Status**: Deprecated comparison
**Reason**: Integrated into `final/h100_final_experiments.sh`

**Policy**: Archived scripts are kept for reference but not maintained.

---

## Recommended Workflow

### 1. Development & Testing (CPU)
```bash
# Quick sanity check
./scripts/quick_cpu_test.sh

# Test specific change
./scripts/quick_cpu_test.sh hints
```

### 2. Full Validation (Single GPU)
```bash
# Run reduced version of production suite
python scripts/train_agent.py \
  --config amlnet_full \
  --device cuda \
  --episodes 100 \
  --max-edges 50000 \
  --guided-exploration
```

### 3. Production Training (H100)
```bash
cd scripts/final
./h100_final_experiments.sh
```

### 4. Analysis
```bash
# Compare results
python scripts/compare_models.py \
  --experiment-dir outputs/final/YYYYMMDD_HHMMSS/ \
  --output report.pdf

# Launch TensorBoard
tensorboard --logdir outputs/final/YYYYMMDD_HHMMSS/
```

---

## Directory Structure

```
scripts/
├── README.md                          # This file
├── train_agent.py                     # Main training entrypoint ⭐
├── quick_cpu_test.sh                  # Fast CPU smoke tests ⭐
├── h100_training_suite.sh             # Legacy H100 suite
├── compare_models.py                  # Model comparison
├── precompute_graph_features.py       # Feature caching
├── final/                             # Production experiment suite
│   ├── README.md                      # Detailed documentation
│   └── h100_final_experiments.sh      # 11 comprehensive experiments ⭐
└── archive/                           # Deprecated scripts
    ├── train_agent_qrdqn.py
    ├── run_dual_training.sh
    ├── benchmark_rmganets_multibranch.sh
    └── compare_gnn_architectures.sh
```

---

## Environment Setup

```bash
# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(f'PyTorch {torch.__version__}')"
uv run python -c "import torch_geometric; print('PyG OK')"

# Check GPU availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Common Issues

### "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 32  # Instead of 64

# Reduce hidden dimensions
--hidden-dims 256,256,128  # Instead of 512,512,256
```

### "Dataset not found"
```bash
# Check dataset path
ls data/amlnet/AMLNet_August_2025.csv

# Update path in config if needed
--config amlnet_full  # Uses default path
```

### "Import errors"
```bash
# Reinstall environment
uv sync --reinstall

# Verify Python version
python --version  # Should be 3.11+
```

---

## Performance Benchmarks

| Script | Hardware | Episodes | Runtime | Purpose |
|--------|----------|----------|---------|---------|
| `quick_cpu_test.sh` | CPU | 20 | 5-10 min | Smoke testing |
| `train_agent.py` (reduced) | RTX 3090 | 100 | 30 min | Validation |
| `final/h100_final_experiments.sh` | H100 | 1500 | 8-12 hrs | Production |

---

## Contributing

When adding new scripts:
1. ✅ Add to this README with clear usage examples
2. ✅ Include proper error handling
3. ✅ Use `set -euo pipefail` for bash scripts
4. ✅ Log to `logs/` directory
5. ✅ Output to `outputs/` directory
6. ✅ Add to `.gitignore` if generating large files

---

**Last Updated**: 2025-10-23
**Version**: 2.0 (Post-cleanup)
