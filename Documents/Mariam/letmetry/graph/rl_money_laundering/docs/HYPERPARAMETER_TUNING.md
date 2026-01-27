# Hyperparameter Tuning with Ray Tune

Automated hyperparameter search for GNN-based anti-money laundering detection using Ray Tune.

## Overview

This system provides comprehensive hyperparameter optimization infrastructure with:

- **Multiple search spaces**: Quick (grid), medium (random), large (Bayesian)
- **Advanced search algorithms**: Optuna (TPE), HyperOpt, random/grid search
- **Early stopping**: ASHA scheduler for efficient resource usage
- **Population-based training**: PBT for dynamic hyperparameter adaptation
- **Comprehensive analysis**: Visualization, comparison tables, best config export

## Quick Start

### 1. Basic Hyperparameter Search

```bash
python scripts/tune_hyperparameters.py \
    --node-features data/elliptic/elliptic_txs_features.csv \
    --edges data/elliptic/elliptic_txs_edgelist.csv \
    --classes data/elliptic/elliptic_txs_classes.csv \
    --search-type medium \
    --search-algo optuna \
    --scheduler asha \
    --num-samples 20 \
    --max-concurrent 4 \
    --max-iterations 50
```

This will:
- Run 20 trials with different hyperparameter combinations
- Use Optuna (Tree-structured Parzen Estimator) for intelligent search
- Employ ASHA scheduler for early stopping of poorly performing trials
- Train each trial for up to 50 iterations
- Run 4 trials concurrently

### 2. Analyze Results

```bash
python scripts/analyze_tune_results.py \
    --results-dir ./ray_results \
    --experiment-name fraud_detection_tune \
    --metric fraud_f1_score \
    --output-dir ./tune_analysis \
    --top-n 10
```

This generates:
- `best_config.json`: Best hyperparameters found
- `top_trials_comparison.csv`: Top 10 trials comparison
- `parameter_importance.png`: Which parameters matter most
- `parallel_coordinates.png`: Visual comparison of top trials
- `learning_curves.png`: Training progression
- `metric_distribution.png`: Performance distribution across trials

## Search Spaces

### Quick Search (Grid Search)

Fast exploration of key parameters. Best for initial experiments.

**Parameters tuned:**
- `lr`: [1e-4, 5e-4, 1e-3]
- `gamma`: [0.95, 0.99]
- `embedding_dim`: [32, 64]
- `gnn_type`: ["rmganets", "gat"]
- `batch_size`: [64, 128]
- `n_step`: [1, 3]
- `fraud_boost_factor`: [2.0, 3.0]

**Usage:**
```bash
python scripts/tune_hyperparameters.py \
    --search-type quick \
    --search-algo random \
    --scheduler fifo \
    --num-samples 10 \
    [data arguments...]
```

**Expected runtime:** ~2-4 hours (depends on hardware)

### Medium Search (Random Search)

Balanced exploration with continuous parameter ranges. Recommended for most use cases.

**Parameters tuned:**
- `lr`: log-uniform(1e-5, 1e-2)
- `gamma`: uniform(0.95, 0.999)
- `embedding_dim`: choice([32, 64, 96, 128])
- `history_dim`: choice([8, 16, 24, 32])
- `hidden_dims`: choice([[128, 128, 64], [256, 128, 64], ...])
- `gnn_type`: choice(["rmganets", "gat", "sage"])
- `batch_size`: choice([32, 64, 128, 256])
- `buffer_capacity`: choice([50000, 100000, 200000])
- `per_alpha`: uniform(0.4, 0.8)
- `per_beta`: uniform(0.3, 0.6)
- `n_step`: choice([1, 2, 3, 5])
- `target_update_freq`: choice([250, 500, 1000])
- `fraud_boost_factor`: uniform(1.5, 5.0)

**Usage:**
```bash
python scripts/tune_hyperparameters.py \
    --search-type medium \
    --search-algo optuna \
    --scheduler asha \
    --num-samples 30 \
    [data arguments...]
```

**Expected runtime:** ~6-12 hours

### Large Search (Bayesian Optimization)

Comprehensive search with advanced features. Best for production optimization.

**Additional parameters:**
- `lr_schedule`: choice([None, "linear", "exponential"])
- `dropout`: uniform(0.1, 0.5)
- `multi_branch`: choice([True, False])
- `epsilon_timesteps`: log-uniform(1000, 50000)
- `final_epsilon`: log-uniform(0.01, 0.2)
- `use_fraud_buffer`: choice([True, False])

**Usage:**
```bash
python scripts/tune_hyperparameters.py \
    --search-type large \
    --search-algo optuna \
    --scheduler pbt \
    --num-samples 50 \
    --max-iterations 100 \
    [data arguments...]
```

**Expected runtime:** ~1-3 days

## Search Algorithms

### Optuna (Recommended)

Tree-structured Parzen Estimator (TPE) - learns from previous trials to suggest better hyperparameters.

**Pros:**
- Intelligent sampling (learns from history)
- Works well with categorical and continuous parameters
- Fast convergence to good regions

**Cons:**
- Requires more trials to be effective (minimum ~20)

**When to use:** Medium to large search spaces, when you can afford 20+ trials

```bash
--search-algo optuna
```

### HyperOpt

Alternative TPE implementation with similar properties.

**Pros:**
- Similar to Optuna
- Well-tested and mature

**Cons:**
- Slightly slower than Optuna

**When to use:** Alternative to Optuna if you prefer its interface

```bash
--search-algo hyperopt
```

### Random/Grid Search

No intelligence - samples uniformly or exhaustively.

**Pros:**
- Simple and reliable
- Works with any search space
- No minimum trial requirement

**Cons:**
- Inefficient for large spaces
- Doesn't learn from previous trials

**When to use:** Quick search space, or as baseline

```bash
--search-algo random
```

## Schedulers

### ASHA (Recommended)

Asynchronous Successive Halving Algorithm - early stops poorly performing trials.

**How it works:**
1. Starts all trials with small budget
2. Periodically evaluates and stops bottom 2/3
3. Gives more resources to promising trials

**Configuration:**
```python
ASHAScheduler(
    metric="fraud_f1_score",
    mode="max",
    max_t=50,              # Max iterations per trial
    grace_period=10,       # Minimum iterations before stopping
    reduction_factor=3     # Stop bottom 2/3 at each rung
)
```

**Pros:**
- Saves resources by stopping bad trials early
- Can run more trials with same budget
- Works well with any search algorithm

**Cons:**
- Might stop promising slow-starters

**When to use:** Default choice for most cases

```bash
--scheduler asha
```

### PBT (Population Based Training)

Dynamically mutates hyperparameters during training.

**How it works:**
1. Runs population of trials in parallel
2. Periodically evaluates performance
3. Poor trials copy weights from good trials
4. Mutates hyperparameters (±20%)

**Configuration:**
```python
PopulationBasedTraining(
    metric="fraud_f1_score",
    mode="max",
    perturbation_interval=10,  # How often to mutate
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.95, 0.999),
    }
)
```

**Pros:**
- Can find better hyperparameters than static search
- Adapts learning rate schedule automatically
- Good for long training runs

**Cons:**
- Requires more concurrent workers
- More complex setup

**When to use:** Large search, long training, many resources

```bash
--scheduler pbt --max-concurrent 8
```

### FIFO

No scheduler - runs all trials to completion.

**Pros:**
- Simplest approach
- Every trial gets full budget

**Cons:**
- Wastes resources on bad trials
- Slower overall

**When to use:** Small number of trials, debugging

```bash
--scheduler fifo
```

## Resource Configuration

### CPU/GPU Allocation

```bash
# Single GPU, 4 workers
python scripts/tune_hyperparameters.py \
    --num-gpus 1 \
    --num-cpus 8 \
    --num-workers 2 \
    --max-concurrent 2 \
    [other args...]

# Multi-GPU, 8 workers
python scripts/tune_hyperparameters.py \
    --num-gpus 4 \
    --num-cpus 16 \
    --num-workers 4 \
    --max-concurrent 4 \
    [other args...]
```

**Resource allocation per trial:**
- 1 GPU for learner (shared across concurrent trials via Ray)
- `num_workers` CPUs for environment rollouts
- 1 CPU for driver

**Recommended configurations:**

| Hardware | num_gpus | num_cpus | num_workers | max_concurrent |
|----------|----------|----------|-------------|----------------|
| 1 GPU, 8 CPU | 1 | 8 | 2 | 2 |
| 2 GPU, 16 CPU | 2 | 16 | 2 | 4 |
| 4 GPU, 32 CPU | 4 | 32 | 4 | 8 |

### Memory Considerations

**Replay buffer capacity** is the main memory consumer:
- `buffer_capacity=50000`: ~2GB RAM
- `buffer_capacity=100000`: ~4GB RAM
- `buffer_capacity=200000`: ~8GB RAM

**Recommendation:** Start with `buffer_capacity=100000` and adjust based on available RAM.

## Interpreting Results

### Best Configuration

The `best_config.json` file contains:

```json
{
  "best_hyperparameters": {
    "lr": 0.0003,
    "gamma": 0.99,
    "embedding_dim": 64,
    "gnn_type": "rmganets",
    "batch_size": 128,
    "fraud_boost_factor": 2.5,
    ...
  },
  "best_metrics": {
    "fraud_f1_score": 0.7234,
    "episode_reward_mean": 45.67,
    "fraud_edges_per_episode": 3.2
  },
  "trial_id": "...",
  "checkpoint_path": "..."
}
```

**How to use:**

1. **Direct integration:**
   ```bash
   python scripts/train_rllib.py \
       --lr 0.0003 \
       --gamma 0.99 \
       --embedding-dim 64 \
       --gnn-type rmganets \
       --batch-size 128 \
       --fraud-boost-factor 2.5 \
       [data arguments...]
   ```

2. **Resume from checkpoint:**
   ```python
   from ray.rllib.algorithms.dqn import DQN

   algo = DQN.from_checkpoint(checkpoint_path)
   result = algo.train()
   ```

### Parameter Importance Plot

Shows which hyperparameters have the strongest correlation with performance.

**Interpretation:**
- **High importance (>0.5)**: Critical parameter, needs careful tuning
- **Medium importance (0.2-0.5)**: Meaningful impact, worth exploring
- **Low importance (<0.2)**: Minor effect, can use default values

**Example:**
```
lr                    ████████████████████ 0.847
fraud_boost_factor    ███████████████ 0.623
gamma                 ██████████ 0.412
embedding_dim         ████ 0.156
batch_size            ██ 0.089
```

**Action:** Focus tuning efforts on `lr` and `fraud_boost_factor`.

### Parallel Coordinates Plot

Visual comparison of top trials across all hyperparameters.

**How to read:**
- Each line = one trial
- Color intensity = performance (darker = better)
- Look for patterns: where do top trials cluster?

**Example insights:**
- Top trials all use `lr` in [0.0002, 0.0005] range
- `fraud_boost_factor` varies widely (2.0-5.0) → less critical
- `gnn_type="rmganets"` dominates top trials

### Learning Curves

Shows training progression for top 5 trials.

**What to look for:**
- **Convergence speed**: How fast does F1 score improve?
- **Stability**: Smooth curves or noisy?
- **Final performance**: Asymptotic value
- **Overfitting**: If curves peak then drop

**Red flags:**
- Flat curves → learning rate too low
- Oscillating curves → learning rate too high
- Early divergence → instability (reduce lr or increase gamma)

### Top Trials Comparison Table

Side-by-side comparison of hyperparameters and metrics.

**Example:**
```
Rank  lr      gamma  embedding_dim  gnn_type   fraud_f1_score  reward
1     0.0003  0.99   64            rmganets   0.7234          45.67
2     0.0005  0.99   96            rmganets   0.7189          44.21
3     0.0002  0.99   64            gat        0.7134          43.89
...
```

**What to look for:**
- **Consistent winners**: Parameters appearing in multiple top trials
- **Surprising combinations**: Unexpected parameter interactions
- **Performance plateaus**: Similar scores despite different params → parameter insensitivity

## Advanced Usage

### Custom Search Space

Edit `tune_hyperparameters.py` to add custom parameters:

```python
def create_search_space(search_type: str = "custom"):
    return {
        # Existing parameters...

        # Add new parameter
        "custom_param": tune.choice([1, 2, 3, 4]),

        # Conditional parameter (only for specific GNN)
        "sage_aggregator": tune.choice(["mean", "sum", "max"]) if gnn_type == "sage" else None,
    }
```

### Multi-Objective Optimization

Optimize for multiple metrics simultaneously:

```python
# In tune_hyperparameters.py, modify tune.report():
tune.report(
    fraud_f1_score=result.get("fraud_f1_score", 0),
    episode_reward_mean=result.get("episode_reward_mean", 0),

    # Composite metric (weighted sum)
    composite_score=0.7 * f1_score + 0.3 * (reward / 100.0)
)
```

Then optimize for `composite_score`:
```bash
python scripts/analyze_tune_results.py --metric composite_score
```

### Distributed Tuning

Run across multiple machines:

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address=<head-node-ip>:6379

# Run tuning (connects to cluster)
python scripts/tune_hyperparameters.py \
    --num-gpus 8 \
    --num-cpus 64 \
    --max-concurrent 16 \
    [other args...]
```

### Warm Starting from Previous Run

Resume tuning with previous trials as starting point:

```python
# In tune_hyperparameters.py
tuner = tune.Tuner(
    ...,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        search_alg=search_alg,
        # Add this:
        resume="AUTO",  # or "ERRORED_ONLY"
    )
)
```

## Troubleshooting

### "Out of memory" errors

**Solutions:**
1. Reduce `buffer_capacity`: `--buffer-capacity 50000`
2. Reduce `max_concurrent`: `--max-concurrent 2`
3. Reduce `batch_size` in search space
4. Use smaller graphs: `--max-nodes 20 --max-edges 100`

### Trials failing immediately

**Check:**
1. Data paths are correct
2. GPU is available (if using `--num-gpus > 0`)
3. Dependencies installed: `pip install ray[tune] optuna`
4. Ray version: `ray --version` (should be ≥2.40.0)

### Poor trial performance

**Likely causes:**
1. Search space too restricted → expand ranges
2. Too few iterations → increase `--max-iterations`
3. Bad reward shaping → check environment rewards
4. Data quality issues → verify Elliptic dataset

### Tuning takes too long

**Speed up:**
1. Use ASHA scheduler for early stopping
2. Reduce `--num-samples`
3. Reduce `--max-iterations` (trial budget)
4. Use "quick" search space
5. Increase `--max-concurrent` (if you have resources)

### Analysis script fails

**Common issues:**
1. Wrong experiment name: `--experiment-name fraud_detection_tune`
2. Results not found: check `--results-dir ./ray_results`
3. Missing matplotlib: `pip install matplotlib seaborn`
4. Incomplete trials: wait for tuning to finish

## Best Practices

### 1. Start Small, Then Scale

```bash
# Phase 1: Quick exploration (2 hours)
python scripts/tune_hyperparameters.py \
    --search-type quick \
    --num-samples 10 \
    --max-iterations 20

# Phase 2: Medium search (8 hours)
python scripts/tune_hyperparameters.py \
    --search-type medium \
    --num-samples 30 \
    --max-iterations 50

# Phase 3: Fine-tuning (2 days)
python scripts/tune_hyperparameters.py \
    --search-type large \
    --num-samples 100 \
    --max-iterations 100
```

### 2. Use ASHA for Efficient Search

ASHA typically provides 3-5x speedup compared to FIFO:

```bash
# Without ASHA: 50 trials × 50 iters = 2500 trial-iterations
--scheduler fifo --num-samples 50 --max-iterations 50

# With ASHA: ~50 trials × 15 avg iters = 750 trial-iterations (same quality!)
--scheduler asha --num-samples 50 --max-iterations 50
```

### 3. Monitor Progress

```bash
# In separate terminal, monitor Ray dashboard
ray dashboard  # Opens at http://localhost:8265

# Or check results during tuning
python scripts/analyze_tune_results.py \
    --results-dir ./ray_results \
    --experiment-name fraud_detection_tune
```

### 4. Validate Best Config

Always validate the best configuration with a longer training run:

```bash
# Get best config from tuning
python scripts/analyze_tune_results.py  # exports best_config.json

# Run full training
python scripts/train_rllib.py \
    --num-iterations 500 \
    [use best hyperparameters from best_config.json]
```

### 5. Version Control Results

```bash
# Save tuning results with timestamp
cp -r ray_results/fraud_detection_tune \
      ray_results/fraud_detection_tune_$(date +%Y%m%d_%H%M%S)

# Commit best config
git add tune_analysis/best_config.json
git commit -m "Best hyperparameters from $(date)"
```

## Example Workflow

Complete end-to-end hyperparameter tuning:

```bash
# 1. Run medium search (overnight)
python scripts/tune_hyperparameters.py \
    --node-features data/elliptic/elliptic_txs_features.csv \
    --edges data/elliptic/elliptic_txs_edgelist.csv \
    --classes data/elliptic/elliptic_txs_classes.csv \
    --search-type medium \
    --search-algo optuna \
    --scheduler asha \
    --num-samples 30 \
    --max-concurrent 4 \
    --max-iterations 50 \
    --experiment-name overnight_tune_$(date +%Y%m%d)

# 2. Analyze results (next morning)
python scripts/analyze_tune_results.py \
    --experiment-name overnight_tune_20250116 \
    --output-dir ./analysis_20250116

# 3. Review best config
cat ./analysis_20250116/best_config.json

# 4. Validate with full training
python scripts/train_rllib.py \
    --num-iterations 200 \
    --lr 0.0003 \
    --gamma 0.99 \
    --embedding-dim 64 \
    --gnn-type rmganets \
    --batch-size 128 \
    --fraud-boost-factor 2.5 \
    --use-fraud-buffer \
    --detailed-callbacks \
    [data arguments...]

# 5. Save results
git add analysis_20250116/best_config.json
git commit -m "Hyperparameter tuning results - F1=0.72"
```

## Performance Benchmarks

Expected improvements from hyperparameter tuning:

| Metric | Baseline (default) | After Tuning | Improvement |
|--------|-------------------|--------------|-------------|
| Fraud F1 Score | 0.65 | 0.72 | +11% |
| Episode Reward | 38.4 | 45.7 | +19% |
| Fraud Edges/Episode | 2.1 | 3.2 | +52% |
| Training Stability | High variance | Low variance | More reliable |

**Typical tuning gains:**
- Quick search: +5-8% F1 improvement
- Medium search: +8-12% F1 improvement
- Large search: +10-15% F1 improvement

## References

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [Optuna: A hyperparameter optimization framework](https://optuna.org/)
- [ASHA: Asynchronous Successive Halving Algorithm](https://arxiv.org/abs/1810.05934)
- [Population Based Training](https://arxiv.org/abs/1711.09846)

## Support

For issues or questions:
1. Check Ray Tune logs: `ray_results/<experiment>/trial_*/result.json`
2. Review Ray dashboard: `http://localhost:8265`
3. See [Ray RLlib documentation](https://docs.ray.io/en/latest/rllib/index.html)
