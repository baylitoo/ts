(getting_started)=
# Getting Started

Welcome to the refreshed quickstart. The project now ships with a declarative pipeline (`rl_money_laundering.pipeline.build_pipeline`) that assembles loaders, feature augmentation, encoders, agents, and trainers straight from a config file. This page walks you through setup, running your first experiment, and validating the new multi-branch stack.

> **Setup time:** ~15 minutes  
> **Prerequisites:** Python ≥ 3.10, `uv` (or pip), access to AMLNet / Elliptic datasets.

```{mermaid}
flowchart TD
    A["Clone Repository"] --> B["Install Dependencies"]
    B --> C["Fetch Datasets"]
    C --> D["Choose Config"]
    D --> E["Run Training via train_agent.py"]
    E --> F["Inspect outputs/ + reports"]
```

## 1. Environment & Dependencies

1. Clone the repository and enter the workspace.
2. Install Python packages (this pulls in docs extras, linting, and Sphinx):

   ```bash
   uv sync --all-extras --group docs
   # or
   python -m pip install -e .[dev,docs]
   ```

3. Build the docs if you want offline HTML:

   ```bash
   uv run sphinx-build docs docs/_build/html
   ```

## 2. Datasets

* Place `AMLNet_August 2025.csv` under `data/amlnet/` **or** keep a copy at repository root and point the CLI to it with `--dataset_path`.
* Extract the Elliptic benchmark into `data/elliptic/` if you plan to run those experiments.
* For smoke tests without regulated data, run `python generate_data.py --output data/synthetic_edges.csv`.

## 3. First Training Run

```bash
# Quick curriculum on AMLNet with RMGANets multi-branch config
uv run python scripts/train_agent.py \
    --config_file configs/amlnet_rmganets_multibranch_improved.json \
    --dataset_path AMLNet_August\ 2025.csv \
    --max_edges 4000 \
    --fraud_ratio 0.25 \
    --episodes 200
```

What happens:

1. `scripts/train_agent.py` loads the JSON config and hands it to `build_pipeline`.
2. The pipeline samples ~4 000 transactions (25 % fraud target), annotates nodes with temporal + network features, and constructs:
   - `StateEncoder` with the RMGANets multi-branch backbone,
   - `DQNAgent` (dueling + Double + PER),
   - `AMLTrainer` with curriculum, evaluation cadence, and multi-branch loss wiring.
3. Artefacts land in `outputs/<experiment-name>/`:
   - `config.json` (fully resolved config),
   - `checkpoints/*.pt`,
   - `training_stats.npz`.

```{note}
Use `--config quick_test` for an even faster smoke test, or `--config elliptic` to train on the Elliptic graph. All overrides (`--episodes`, `--device`, `--max_edges`, `--fraud_ratio`) are applied after loading the preset or JSON file, so you never have to duplicate configs.
```

## 4. Modular “Plug & Play” via Configs

Every preset and JSON configuration uses the same schema defined in {mod}`rl_money_laundering.config`. Key sections you can edit:

```json
{
  "gnn": {
    "gnn_type": "rmganets",
    "multi_branch": true,
    "use_dqn_enhancement": true,
    "embedding_dim": 64,
    "history_dim": 16
  },
  "trainer": {
    "num_episodes": 1500,
    "multi_branch": {
      "enabled": true,
      "variant": "improved",
      "lambda_branch": 0.30,
      "epsilon_dqn": 0.40,
      "beta_reg": 0.05,
      "temporal_decay": 0.10
    }
  }
}
```

Swap encoders (`sage`, `gat`, `rmganets`), toggle the multi-branch loss, or change the RL agent (DQN vs QRDQN) without touching code—just edit and rerun `train_agent.py`.

## 5. Verifying the Multi-Branch Stack

The project includes a CPU-only integration test that exercises the full pipeline on a small AMLNet slice, ensuring the multi-branch heads emit logits and the improved loss receives real DQN signals.

```bash
uv run python test_multibranch_comprehensive.py \
    --dataset_path AMLNet_August\ 2025.csv \
    --max_edges 2500 \
    --fraud_ratio 0.2
```

Expect output similar to:

```
[2/5] Verifying encoder auxiliary heads...
  Focus node: C8310
  Embedding shape: (64,)
  To-branch logits shape: (2,)
  Hy-branch logits shape: (1,)
...
[4/5] Simulating multi-branch optimisation steps...
  Node C9241: total=0.31, main_ce=0.07, to_gcm_focal=0.14, hy_gcm_bce=0.56, dqn_mae=0.06
```

The script relies solely on `build_pipeline`, making it an excellent template for custom smoke tests or CI jobs.

## 6. Distributed Training with RLlib (Optional)

For production-scale training with parallel environment rollouts:

```bash
# Install RLlib dependencies (if not already installed)
pip install "ray[rllib]>=2.40.0"

# Run distributed training
uv run python scripts/train_rllib.py \
    --dataset AMLNet_August\ 2025.csv \
    --num-workers 4 \
    --num-gpus 1 \
    --training-iterations 1000 \
    --checkpoint-freq 100
```

**What this enables:**

- **2-8x speedup** – Parallel workers collect experience simultaneously
- **Multi-GPU training** – Distribute batches across GPUs
- **Hyperparameter tuning** – Use Ray Tune for automated search:
  ```bash
  uv run python scripts/tune_hyperparameters.py \
      --dataset AMLNet_August\ 2025.csv \
      --num-samples 20 \
      --cpus-per-trial 4
  ```
- **Production deployment** – Export trained models for Ray Serve

**Key differences from standalone training:**

| Feature | Standalone (`train_agent.py`) | RLlib (`train_rllib.py`) |
|---------|------------------------------|--------------------------|
| Environment rollouts | Sequential (single process) | Parallel (multi-process) |
| GPU utilization | Single GPU | Multi-GPU support |
| Hyperparameter search | Manual | Ray Tune integration |
| Deployment | Manual export | Ray Serve ready |
| Scalability | Single machine | Cluster-ready |

```{note}
RLlib training requires additional setup. See {doc}`../RLLIB_INTEGRATION` for detailed installation and configuration.
```

## 7. Next Steps

* Read {doc}`architecture` for deep dives into the pipeline, multi-branch loss, and RMGANets internals.
* Follow {doc}`experiments` for benchmarking, curriculum tuning, and evaluation exports.
* Use `sphinx-autobuild docs docs/_build/html` during authoring to preview the Mermaid diagrams and cross-references live.

```{admonition} Rapid Iteration Tips
:class: tip
- Keep alternative configs under `configs/` and share them across the team.
- Use `--max_edges`/`--fraud_ratio` to stabilise experiment durations when iterating.
- Combine `test_multibranch_comprehensive.py` with `pytest -k multibranch` to catch regressions before long runs.
- Drop artefacts into `outputs/<run>/` and run `python rl_money_laundering/evaluator.py` for compliance-ready summaries.
```
