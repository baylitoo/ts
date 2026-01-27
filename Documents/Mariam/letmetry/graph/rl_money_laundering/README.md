# Temporal Graph Guardrails with RL-Guard (AML Case Study)

This repository houses the guardrail-aware reinforcement learning stack that powers our HARBOR / RL-Guard deployments on temporal transaction graphs. It combines curriculum-driven exploration, graph neural encoders, distributional DQN variants, and compliance telemetry so policies remain auditable under extreme label sparsity. Anti-money laundering (AML) remains our flagship case study, but the abstractions are built to accept any temporal heterogeneous graph that must satisfy production guardrails.

---

## Highlights

- **Full pipeline integration** – data loading, graph construction, feature engineering, environment simulation, agent training, evaluation, and guardrail telemetry.
- **RL-Guard alignment** – every encoder/agent combo emits immutable audit payloads, directive IDs, and risk metrics that slot directly into the RL-Guard gateway.
- **RLlib distributed training** – Ray RLlib integration enables parallel environment rollouts (2-8x speedup), multi-GPU training, hyperparameter tuning via Ray Tune, and production deployment.
- **Modular GNN architecture** – RMGANets with separate Att-GCM, To-GCM, and Hy-GCM modules for interpretable graph reasoning, plus GraphSAGE and GAT alternatives.
- **Multi-branch loss** – implements the RMGANets paper loss plus an improved variant with focal loss for extreme class imbalance and temporal-aware DQN loss weighting.
- **Joint GNN+RL training** – GNN encoder is trained simultaneously with the RL agent, enabling end-to-end learning from graph structure to policy.
- **Graph + temporal features** – combines PyTorch Geometric network statistics with temporal velocity/business-hour descriptors.
- **Curriculum & exploration** – staged fraud-rate curriculum, adjustable intrinsic rewards, and slow ε decay tailored for sparse rewards.
- **Fraud-aware replay buffer** – guarantees each training batch contains rare positive examples so the agent keeps seeing successful trajectories.
- **Distributional & dueling DQN options** – supports standard DQN, dueling DQN, and QR-DQN with n-step prioritized replay.
- **Budgeted compliance metrics** – Recall@K, Precision@K, and AUPR metrics that reflect real-world alert budget constraints.
- **Pre-computation tooling** – cache fully featured graphs once, then reuse to speed up repeated training runs or online deployment.
- **H100-ready** – designed for large-GPU setups but also runnable on CPU with reduced configs.

---

## Repository Layout

```
rl_money_laundering/
├─ src/rl_money_laundering/
│  ├─ datasets/            # AMLNet & Elliptic loaders, feature pipeline
│  ├─ features/            # Temporal + PyG network features, extractors
│  ├─ gnn_modules/         # RMGANets components (Att-GCM, To-GCM, Hy-GCM, multi-branch loss)
│  ├─ rllib_integration/   # Ray RLlib wrappers, GraphSpace, GNNDQNModule, fraud callbacks
│  ├─ evaluation/          # Budgeted metrics (Recall@K, Precision@K, AUPR)
│  ├─ environment.py       # Gymnasium env simulating graph navigation
│  ├─ gnn_encoder.py       # State encoder supporting GraphSAGE/GAT/RMGANets
│  ├─ agent.py             # DQN, QR-DQN, buffers, replay logic
│  ├─ trainer.py           # Curriculum loop, joint GNN+RL training, multi-branch loss
│  ├─ pipeline.py          # Unified pipeline builder from config
│  └─ utils/               # Feature factories, graph splits, metrics
├─ configs/                # JSON experiment configurations
├─ scripts/
│  ├─ train_agent.py       # Config-driven training CLI (supports DQN/QR-DQN)
│  ├─ train_rllib.py       # RLlib distributed training CLI
│  ├─ precompute_graph_features.py  # Feature caching utility
│  ├─ compare_models.py    # Model comparison utilities
│  └─ tune_hyperparameters.py  # Ray Tune hyperparameter optimization
├─ tests/                  # Unit / integration suites
├─ docs/                   # Architecture notes, API references, RLlib integration guide
└─ README.md               # You are here
```

---

## Getting Started

### 1. Prerequisites

- Python ≥ 3.10  
- [uv](https://github.com/astral-sh/uv) for environment management  
- CUDA-capable GPU (recommended) and the matching PyTorch/PyG wheels  
- AMLNet dataset (`AMLNet_August_2025.csv`) or your own transactional graph dump

### 2. Environment Setup

```bash
# Create & sync the virtual environment (installs prod + dev deps)
uv sync --dev

# Activate the environment (PowerShell example)
.\.venv\Scripts\Activate.ps1
```

### 3. Dataset Placement

By default the configs expect:

```
rl_money_laundering/
└─ data/
   └─ amlnet/
      └─ AMLNet_August_2025.csv
```

For custom datasets, edit `src/rl_money_laundering/config.py` or supply `--config_file` to the training script.

### 4. Pre-compute Graph Features (optional but recommended)

```bash
python scripts/precompute_graph_features.py \
  --dataset data/amlnet/AMLNet_August_2025.csv \
  --output data/amlnet/cache/graph.gpickle \
  --device cuda \
  --nrows 200000   # optional row cap for prototyping
```

This extracts temporal + network features once, stores them as a pickled NetworkX graph, and saves metadata for reproducibility. Future training runs can load this cache instead of recomputing expensive features.

---

## Training Recipes

### Quick Smoke Test (CPU-friendly)

```bash
python scripts/train_agent.py --config quick_test --device cpu
```

Key characteristics:
- Loads 10 k transactions
- Short curriculum `[1.0, 0.5, 0.1]`
- Elevated intrinsic reward (0.5) and slower ε decay
- Completes in a few minutes on laptop hardware

### Full AMLNet Training (GPU, e.g. H100)

```bash
python scripts/train_agent.py --config amlnet_full --device cuda
```

Default behaviour:
- Curriculum from 100% fraud starts down to 2%
- Intrinsic reward decays with curriculum stage so exploration focus shifts naturally
- Replay buffer capacity 100 k (consider raising to 500 k+ on H100)
- Fraud-aware sampler enforces ~25% positive samples per batch when available

Tweakable config knobs live in `src/rl_money_laundering/config.py`. For production runs you might:
- Increase `agent.hidden_dims` via `--hidden-dims 512,512,256` (or edit the config) and raise `gnn.embedding_dim`
- Expand `trainer.batch_size` to 512 and `buffer_capacity` to several hundred thousand
- Reduce `trainer.curriculum_schedule` granularity for online streaming updates

### QR-DQN (Distributional) Training

```bash
python scripts/train_agent.py --config amlnet_full --agent-type qrdqn --device cuda \
  --num-quantiles 128 --n-step 5 --positive-fraction 0.25
```

This variant enables distributional value estimation with the N-step prioritized replay buffer. Combine it with `--risk-measure cvar_90`, additional PER knobs (e.g. `--prioritized-beta 0.6`), or wider heads via `--hidden-dims 512,512,256` for more risk-aware exploration. The legacy `scripts/train_agent_qrdqn.py` remains as a thin compatibility wrapper around this entrypoint.

### Launch Both Agents Concurrently

```bash
bash scripts/run_dual_training.sh
```

This helper script runs the default DQN and the QR-DQN trainers side-by-side, streaming logs to `logs/dqn_run.log` and `logs/qrdqn_run.log` while writing checkpoints to distinct output directories.

### Distributed Training with RLlib

For production-scale training with parallel environment rollouts and multi-GPU support:

```bash
python scripts/train_rllib.py \
  --dataset data/amlnet/AMLNet_August_2025.csv \
  --num-workers 4 \
  --num-gpus 1 \
  --training-iterations 1000
```

Key benefits:
- **2-8x speedup** from parallel environment rollouts
- **Multi-GPU training** for large graphs
- **Hyperparameter tuning** via Ray Tune integration
- **Production deployment** ready with Ray Serve
- **Advanced replay buffers** with fraud-aware sampling

See `docs/RLLIB_INTEGRATION.md` for detailed setup and usage.

### Custom Configuration

Create a JSON file with `ExperimentConfig` payload (see `config.py`) and pass `--config_file path/to/custom.json`. CLI overrides such as `--nrows`, `--episodes`, or `--device` still work.

---

## Evaluation & Monitoring

Evaluations run automatically every `trainer.eval_frequency` episodes and at the end of training. They report:
- Mean reward over evaluation episodes
- Detection rate (share of episodes that flagged at least one fraud edge)
- False-positive rate

To inspect detailed trajectories, enable rendering or dump `trajectory` objects from the trainer.

For ad-hoc diagnostics, use the integration tests in `tests/` (e.g. `tests/test_full_pipeline.py`) which print sectioned status updates.

---

## Key Components

- **Environment (`environment.py`)**
  Gymnasium-compatible `AMLDetectionEnv` enforces temporal ordering, manages the action space (neighbors + flag), computes intrinsic rewards, and recognises fraud edges through all known labels (`is_fraud`, `isFraud`, `is_money_laundering`).

- **Modular GNN Architecture (`gnn_modules/`)**
  - **RMGANets encoder** with separate Att-GCM (attention), To-GCM (topology), and Hy-GCM (hybrid) modules
  - **Multi-branch loss** – paper-faithful implementation plus improved variant with focal loss and temporal weighting
  - **Fusion layer** – adaptively combines the three correlation modules
  - Also supports GraphSAGE and GAT backbones for comparison

- **State Encoder (`gnn_encoder.py`)**
  Extracts k-hop subgraphs, converts to PyG data, applies the selected GNN backbone (RMGANets/GraphSAGE/GAT), and concatenates history features. **Trained end-to-end with the RL agent** for joint optimization.

- **Fraud-aware Replay (`agent.py`)**
  Both standard and quantile DQN variants label transitions with a fraud flag. The prioritized buffer enforces a configurable positive sample fraction per minibatch, ensuring rare successes shape the value function.

- **Trainer (`trainer.py`)**
  Implements curriculum scheduling, epsilon reporting, intrinsic reward scaling, **joint GNN+RL optimization**, multi-branch loss integration, and fraud-aware episode sampling. Supports checkpointing via `CheckpointManager`.

- **RLlib Integration (`rllib_integration/`)**
  Production-ready distributed training with Ray RLlib, including custom GraphSpace, environment wrappers, GNNDQNModule, fraud-aware replay buffers, and performance callbacks.

- **Evaluation Metrics (`evaluation/`)**
  Budgeted detection metrics (Recall@K, Precision@K, AUPR) that reflect real-world compliance constraints where teams can only investigate K transactions per day.

---

## Scaling Tips (H100 / Large Graphs)

1. **Increase capacity** – raise `buffer_capacity`, `batch_size`, and network widths dramatically; VRAM is abundant.
2. **Parallel collectors** – spin up multiple environment instances feeding a shared buffer for faster data gathering.
3. **Feature caches** – use `precompute_graph_features.py` and keep caches in NVMe/ramdisk to avoid CPU bottlenecks.
4. **Mixed precision** – enable AMP if desired (not yet default) to accelerate DQN training further.
5. **Component sweeps** – use connected-component metadata to schedule resets across the entire graph, maintaining coverage in a live deployment.

---

## Testing & Quality

```bash
# Lint & format
uv run ruff format
uv run ruff check --fix

# Type-check (selected modules; extend as needed)
uv run mypy src/rl_money_laundering/config.py src/rl_money_laundering/environment.py

# Unit / integration tests
uv run pytest
```

Many test scripts (e.g. `tests/test_full_pipeline.py`) are designed to print human-readable checkpoints rather than assert-heavy outputs—use them as integration sanity checks after major refactors.

---

## Troubleshooting

- **All episodes end with −1 reward:** confirm dataset edges carry at least one of `isFraud`, `is_fraud`, or `is_money_laundering`. The environment now checks all aliases.
- **Training plateau after curriculum stage 1:** try widening the epsilon decay (e.g., `epsilon_decay=0.999`), increasing `positive_fraction`, or boosting intrinsic rewards.
- **PyG CUDA import errors:** ensure the installed PyG wheels match your CUDA/PyTorch version; reinstall with `pip install torch-geometric -f https://data.pyg.org/whl/torch-<torch_version>+cu<cuda_version>.html`.
- **Slow startup due to feature extraction:** run the precompute script and point configs to the cached graph.

---

## Recent Achievements

### ✅ Completed (Latest Release)
- **RLlib distributed training** – Full Ray RLlib integration with parallel rollouts, multi-GPU support, and production deployment capabilities
- **Modular GNN architecture** – Separated RMGANets into Att-GCM, To-GCM, Hy-GCM modules with interpretable fusion
- **Multi-branch loss variants** – Paper implementation + improved version with focal loss and temporal-aware DQN loss
- **Joint GNN+RL training** – GNN encoder trained end-to-end with the RL agent for unified optimization
- **Budgeted compliance metrics** – Recall@K, Precision@K, and AUPR for real-world evaluation
- **Elliptic dataset integration** – Full support with specialized feature extractor
- **Config-driven pipeline** – `build_pipeline()` enables experiment configuration via JSON files
- **Progressive episode length** – Episodes grow from 20→200 steps during training to match evaluation horizons

### Roadmap

#### Near-term Improvements
- **Adaptive fraud-seeding curriculum** – Dynamically adjust fraud-seeded starts based on performance metrics (Found%, Detection Rate, avg reward). Candidates:
  - Performance-triggered boosting (smart, complex)
  - Declining with floor (e.g., never below 30% fraud starts)
- **Hyperparameter optimization at scale** – Leverage Ray Tune for systematic sweeps on H100 clusters

#### Research Extensions
- Random Network Distillation (RND) intrinsic bonuses
- Multi-agent exploration with shared replay
- Streaming graph updates & online evaluation dashboards
- Edge-level precision/recall metrics (complement episode-level confusion matrix)
- Integration with additional graph benchmarks (IBM AMLSim, synthetic laundering scenarios)

---

## Contributing

We welcome issues and pull requests. Please run the lint/type/test suite before submitting and follow the PR template below:

```
### Summary
<What changed and why?>

### Changes
- ...

### Testing
- ...

### Risks / Mitigations
- ...
```

---

Happy hunting! Feel free to file issues, discuss research ideas, or contribute additional graph benchmarks. With the fraud-aware replay sampler and curriculum in place, we’re ready to tackle truly large, sparse transactional graphs.
