(experiments)=
# Training & Experiments

The experiment workflow is now driven by configuration files and a reusable pipeline builder. Whether you are benchmarking encoders, toggling the multi-branch loss, or spinning up distributional agents, the same entrypoints apply.

```{mermaid}
flowchart LR
    CFG["ExperimentConfig (JSON/preset)"] -->|build_pipeline| PIPE["PipelineArtifacts"]
    PIPE --> DATASET["DatasetArtifacts<br/>transactions + graph"]
    PIPE --> ENV["AMLDetectionEnv"]
    PIPE --> ENC["StateEncoder<br/>(SAGE | GAT | RMGANets)"]
    PIPE --> AGENT["DQNAgent | QRDQNAgent"]
    PIPE --> TRAIN["AMLTrainer<br/>(curriculum + eval)"]
    TRAIN --> OUTPUT["outputs/<run>/"]
    OUTPUT -->|checkpoints| CKPT["Checkpoint Files"]
    OUTPUT -->|training_stats.npz| METRICS["Metrics"]
    OUTPUT -->|config.json| SNAPSHOT["Config Snapshot"]
```

## Dataset Checklist

::::{grid} 1 1 2 2
:gutter: 2
:class-container: sd-gap-3 sd-p-2

:::{grid-item-card} AMLNet
:class-card: sd-shadow-sm
- Recommended path: `data/amlnet/AMLNet_August_2025.csv`
- Low-base-rate fraud (≈0.4 % in the curated subset)
- Supports on-the-fly sampling via `--max_edges` & `--fraud_ratio`
:::

:::{grid-item-card} Elliptic
:class-card: sd-shadow-sm
- Root folder: `data/elliptic/`
- Precomputed temporal buckets; ideal for cross-market validation
- Use `--config elliptic` for the bundled preset
:::

:::{grid-item-card} Synthetic
:class-card: sd-shadow-sm
- Generate with `python generate_data.py --output data/simulated_edges.csv`
- Adjust fraud prevalence & graph size for regression testing
- Lightweight CI smoke tests
:::

::::

All loaders converge on the same schema (NetworkX DiGraph + raw transactions) consumed by the pipeline.

## Core Commands

```{code-block} bash
# Quick AMLNet curriculum (GraphSAGE)
uv run python scripts/train_agent.py --config quick_test

# Full AMLNet run with RMGANets multi-branch improvements
uv run python scripts/train_agent.py \
    --config_file configs/amlnet_rmganets_multibranch_improved.json \
    --dataset_path AMLNet_August\ 2025.csv \
    --max_edges 6000 \
    --fraud_ratio 0.25 \
    --episodes 1500

# Elliptic benchmark, CPU only
uv run python scripts/train_agent.py \
    --config elliptic \
    --device cpu \
    --output_dir outputs/elliptic_cpu
```

### CLI Overrides

`train_agent.py` exposes the following switches (applied after loading the preset/JSON config):

| Flag | Purpose |
| --- | --- |
| `--episodes` | Overrides `config.trainer.num_episodes` |
| `--nrows` | Restricts the number of dataset rows ingested |
| `--max_edges` | Limits the transactions/edges kept for AMLNet (sampling) |
| `--fraud_ratio` | Target fraud proportion when sampling AMLNet subsets |
| `--device` | Forces `cpu` / `cuda` / `auto` |
| `--output_dir` | Redirects checkpoints & metrics |
| `--dataset_path` | Points to a custom CSV or dataset root |

This makes experiments highly reproducible—drop a config in `configs/`, version it, and override the few knobs you need at runtime.

## Curriculum & Evaluation

```{mermaid}
flowchart LR
    Stage1["Stage 1<br/>fraud_rate = 1.00"] --> Stage2["Stage 2<br/>0.80"]
    Stage2 --> Stage3["Stage 3<br/>0.60"]
    Stage3 --> Stage4["Stage 4<br/>0.40"]
    Stage4 --> Stage5["Stage 5<br/>0.20"]
    Stage5 --> Stage6["Stage 6<br/>0.10"]
    Stage6 --> Stage7["Stage 7<br/>0.02<br/>production prior"]
```

- **Evaluation** (`eval_frequency`) runs 20 greedy rollouts (`ε=0`) and logs reward, detection rate, and false-positive rate.
- **Checkpointing** (`checkpoint_frequency`) drops `checkpoint_ep{N}.pt` under `outputs/<run>/checkpoints/`.
- **Metrics archive**: `training_stats.npz` stores arrays for rewards, episode lengths, detection rate, and FPR. Load with `np.load(..., allow_pickle=True)` for analysis notebooks.

## Switching Encoders & Agents

`build_pipeline` reads whatever you encode in the config object:

```python
from rl_money_laundering.config import ExperimentConfig
from rl_money_laundering.pipeline import build_pipeline

cfg = ExperimentConfig.load("configs/amlnet_gat.json")
cfg.gnn.gnn_type = "rmganets"
cfg.gnn.multi_branch = True
cfg.trainer.multi_branch.enabled = True

artifacts = build_pipeline(cfg, max_edges=5000, fraud_ratio=0.2, device="cuda")
trainer = artifacts.trainer
history = trainer.train(num_episodes=750)
```

Swap `DQNAgent` with `QRDQNAgent` before constructing the trainer if you need distributional policies:

```python
from rl_money_laundering.agent import QRDQNAgent

agent = QRDQNAgent(
    state_dim=artifacts.state_encoder.get_state_dim(),
    action_dim=artifacts.env.action_space.n,
    num_quantiles=64,
    n_step=3,
    risk_measure="cvar_90",
    device="cuda",
)
artifacts.trainer.agent = agent
```

Everything else (curriculum, logging, checkpointing) continues unchanged.

## Distributed Training with RLlib

For production-scale experiments, use the RLlib integration for parallel rollouts and multi-GPU training:

```bash
# Single-node distributed training
python scripts/train_rllib.py \
    --dataset data/amlnet/AMLNet_August_2025.csv \
    --num-workers 4 \
    --num-gpus 1 \
    --training-iterations 1500 \
    --checkpoint-freq 100 \
    --output-dir outputs/rllib_amlnet

# Multi-node cluster training (requires Ray cluster)
python scripts/train_rllib.py \
    --dataset data/amlnet/AMLNet_August_2025.csv \
    --num-workers 16 \
    --num-gpus 4 \
    --training-iterations 3000 \
    --use-ray-cluster
```

### Hyperparameter Tuning with Ray Tune

Automatically search for optimal hyperparameters:

```bash
python scripts/tune_hyperparameters.py \
    --dataset data/amlnet/AMLNet_August_2025.csv \
    --num-samples 50 \
    --cpus-per-trial 4 \
    --gpus-per-trial 0.25 \
    --scheduler asha
```

This uses ASHA (Asynchronous Successive Halving) to efficiently explore:
- Learning rates (1e-5 to 1e-2)
- Hidden layer dimensions
- GNN architecture (SAGE/GAT/RMGANets)
- Multi-branch loss weights (λ, ε, β)
- Curriculum schedules

Results are logged to TensorBoard and saved to `outputs/tune_results/`.

### Comparison: Standalone vs RLlib

| Metric | Standalone (CPU, 1 worker) | RLlib (4 workers) | RLlib (8 workers, GPU) |
|--------|---------------------------|-------------------|------------------------|
| Training time (1500 episodes) | ~8 hours | ~2 hours | ~1 hour |
| Throughput (episodes/min) | ~3 | ~12 | ~25 |
| Memory usage | ~4 GB | ~8 GB | ~12 GB |
| GPU utilization | 0% | 0% | 75-85% |

```{note}
RLlib training requires `ray[rllib]>=2.40.0`. See {doc}`../RLLIB_INTEGRATION` for detailed setup.
```

## Multi-Branch Loss & Monitoring

The improved RMGANets pipeline adds a dedicated multi-branch loss (`MultiBranchLoss`) that fuses:

```{mermaid}
flowchart LR
    Main["Main logits"] --> Sum["Adaptive Loss Blend"]
    ToBranch["To-GCM logits"] --> Sum
    HyBranch["Hy-GCM logits"] --> Sum
    DQNPreds["DQN predictions"] --> Sum
    Sum --> Metrics["Metrics<br/>(total, main_ce, to_gcm_focal,<br/>hy_gcm_bce, dqn_mae)"]
```

Enable it by setting:

```json
"gnn": {
  "gnn_type": "rmganets",
  "multi_branch": true
},
"trainer": {
  "multi_branch": {
    "enabled": true,
    "variant": "improved",
    "lambda_branch": 0.30,
    "epsilon_dqn": 0.40,
    "beta_reg": 0.05,
    "temporal_decay": 0.10
  }
}
```

During training, `trainer.multi_branch_metrics` accumulates loss components so you can monitor auxiliary branches:

```python
metrics = artifacts.trainer.multi_branch_metrics[-1]
print(metrics["total"], metrics["hy_gcm_bce"], metrics["dqn_mae"])
```

Use the bundled integration script for quick verification:

```bash
uv run python test_multibranch_comprehensive.py \
    --config configs/amlnet_rmganets_multibranch_improved.json \
    --dataset_path AMLNet_August\ 2025.csv \
    --max_edges 2500 \
    --fraud_ratio 0.2
```

## Artefacts & Reporting

* `outputs/<run>/checkpoints/` – PyTorch weights (online & target networks).
* `outputs/<run>/training_stats.npz` – NumPy metrics archive.
* `outputs/<run>/config.json` – frozen configuration snapshot.
* `outputs/<run>/logs/` – optional tracker logs if you enable {class}`rl_money_laundering.utils.ExperimentTracker`.

Export compliance-friendly summaries using the evaluator:

```python
from rl_money_laundering.evaluator import EvaluationReport

report = EvaluationReport.from_run_path("outputs/amlnet_rmganets/latest")
report.export_markdown("reports/amlnet_rmganets.md")
```

Pair the generated Markdown with the Mermaid diagrams from this documentation to deliver explainable AI packages for auditors.

## Budgeted Evaluation Metrics

The `evaluation/` module provides compliance-oriented metrics for real-world assessment:

```python
from rl_money_laundering.evaluation import BudgetedMetrics
import numpy as np

# Load model predictions and ground truth
predictions = model.predict(X_test)  # Risk scores (higher = more suspicious)
labels = y_test  # 0 = normal, 1 = fraud

metrics = BudgetedMetrics()

# Scenario: compliance team can investigate 100 transactions/day
k = 100
recall_k = metrics.recall_at_k(predictions, labels, k=k)
precision_k = metrics.precision_at_k(predictions, labels, k=k)
aupr = metrics.aupr(predictions, labels)

print(f"Alert budget: {k} transactions/day")
print(f"Recall@{k}: {recall_k*100:.1f}% of fraud caught")
print(f"Precision@{k}: {precision_k*100:.1f}% of alerts are fraud")
print(f"AUPR: {aupr:.3f}")

# Generate precision-recall curve
metrics.plot_precision_recall_curve(
    predictions, labels,
    save_path="outputs/pr_curve.png"
)
```

### Why Budgeted Metrics Matter

**Traditional metrics are misleading for extreme imbalance:**

With 0.16% fraud rate (16 frauds per 10,000 transactions):
- **Accuracy = 99.84%** achieved by always predicting "not fraud" ❌
- **AUROC** insensitive to positive class prevalence ❌

**Budgeted metrics answer operational questions:**

- **Recall@K**: "If I can only investigate K cases/day, what % of fraud will I catch?"
- **Precision@K**: "What % of my K daily alerts will be true positives?"
- **AUPR**: Area under precision-recall curve (better than AUROC for imbalanced data)

### Practical Example

Compliance team scenario:
- Daily transaction volume: 50,000
- Alert capacity: 200 investigations/day (0.4% of volume)
- Fraud prevalence: 0.16% (80 frauds/day)

```python
# Evaluate at operational capacity
recall_200 = metrics.recall_at_k(predictions, labels, k=200)
precision_200 = metrics.precision_at_k(predictions, labels, k=200)

# How many frauds caught?
frauds_caught = recall_200 * 80
false_positives = 200 - frauds_caught

print(f"With 200 alerts/day:")
print(f"  Frauds caught: {frauds_caught:.0f} / 80 ({recall_200*100:.1f}%)")
print(f"  False positives: {false_positives:.0f}")
print(f"  Precision: {precision_200*100:.1f}%")
```

These metrics directly inform:
- **Resource allocation** – How many investigators needed?
- **Alert thresholds** – Where to set risk score cutoff?
- **Model comparison** – Which approach maximizes fraud detection within budget?

## Automation Tips

```{admonition} Recommended Scripts
:class: tip
- `scripts/compare_gnn_architectures.sh` – run SAGE, GAT, and RMGANets back-to-back.
- `scripts/benchmark_rmganets_multibranch.sh` – sweep multi-branch hyperparameters using configs in `configs/`.
- `scripts/run_dual_training.sh` – launch DQN and QRDQN variants simultaneously for A/B comparisons.
```

Trigger these from CI or a scheduler; each wraps `train_agent.py` and therefore inherits the new pipeline functionality.
