(architecture)=
# System Architecture

The platform couples graph-native feature extraction, configurable GNN encoders, and curriculum-driven reinforcement learning to surface money-laundering patterns. Each stage can be swapped independently—datasets, encoders, or agents—without touching the rest of the stack.

> Mermaid diagrams throughout the page provide compact views of data flow and module boundaries.

::::{grid} 1 1 2 2
:gutter: 2
:class-container: sd-gap-3 sd-mb-4

:::{grid-item-card} Data Sources
:class-card: sd-shadow-sm
`AMLNetLoader`, `EllipticLoader`, and `generate_data.py` normalise disparate transaction graphs into a shared schema (IDs, timestamps, risk labels).
:::

:::{grid-item-card} Feature Augmentation
:class-card: sd-shadow-sm
`TemporalFeatureExtractor`, `add_network_features_pyg`, and `NodeFeatureExtractor` inject temporal velocity, amount statistics, and structural signals directly into the NetworkX graph.
:::

:::{grid-item-card} GNN Modules
:class-card: sd-shadow-sm
Modular GNN components in `gnn_modules/`: RMGANets (Att-GCM, To-GCM, Hy-GCM, fusion), multi-branch loss (paper + improved), trained jointly with RL agent.
:::

:::{grid-item-card} State Encoding
:class-card: sd-shadow-sm
`StateEncoder` supports GraphSAGE, GAT, and RMGANets encoders, combining learned embeddings with visit-history statistics. **Trained end-to-end with the RL policy.**
:::

:::{grid-item-card} Policy & Ops
:class-card: sd-shadow-sm
`AMLDetectionEnv`, `DQNAgent`/`QRDQNAgent`, and `AMLTrainer` manage training, evaluation, and checkpoint export via `CheckpointManager` and `EvaluationReport`.
:::

:::{grid-item-card} Distributed Training
:class-card: sd-shadow-sm
`rllib_integration/` provides Ray RLlib wrappers for parallel rollouts, multi-GPU training, hyperparameter tuning, and production deployment.
:::

:::{grid-item-card} Evaluation Metrics
:class-card: sd-shadow-sm
`evaluation/` implements budgeted metrics (Recall@K, Precision@K, AUPR) reflecting real-world compliance constraints.
:::

::::

## Component Graph

```{mermaid}
flowchart TD
    subgraph Data["Data Layer"]
        AML[AMLNetLoader]
        ELL[EllipticLoader]
        SYN[Synthetic Generator]
    end
    subgraph Feature["Feature Augmentation"]
        TEMP[TemporalFeatureExtractor]
        NET[add_network_features_pyg]
        NODE[NodeFeatureExtractor]
    end
    subgraph Encode["State Encoding"]
        ENC["StateEncoder<br/>(GraphSAGE / GAT / RMGANets)"]
    end
    subgraph Env["Environment"]
        AMLENV["AMLDetectionEnv"]
    end
    subgraph Policy["Agents"]
        DQN["DQNAgent<br/>(Dueling + Double + PER)"]
        QR["QRDQNAgent<br/>(Quantile + N-step + CVaR)"]
    end
    subgraph Train["Training"]
        TRAIN[AMLTrainer]
    end
    subgraph Eval["Evaluation & Ops"]
        EVAL[EvaluationReport]
        CKPT[CheckpointManager]
        STATS[training_stats.npz]
    end
    OUTPUT[Compliance Reports]

    AML --> TEMP
    ELL --> TEMP
    SYN --> TEMP
    TEMP --> NET
    NET --> NODE
    NODE --> ENC
    ENC --> AMLENV
    AMLENV --> DQN
    AMLENV --> QR
    DQN --> TRAIN
    QR --> TRAIN
    TRAIN --> AMLENV
    TRAIN --> CKPT
    TRAIN --> STATS
    TRAIN --> EVAL
    EVAL --> OUTPUT
```

## Data & Feature Pipeline

1. **Transaction ingestion** – dataset loaders in `rl_money_laundering.datasets` return a directed NetworkX graph alongside raw transaction tables.
2. **Temporal enrichment** – `TemporalFeatureExtractor` populates velocity, business-hour ratios, and periodicity metrics on nodes and edges.
3. **Structural descriptors** – `add_network_features_pyg` computes degree, clustering, and centrality metrics via PyTorch Geometric.
4. **Unified node vectors** – `NodeFeatureExtractor` concatenates categorical encodings, amount statistics, temporal signals, and structural features into the fixed-length vector expected by the encoder.

```{mermaid}
flowchart LR
    Transactions -->|"temporal features"| TemporalFX[add_temporal_features_to_graph]
    TemporalFX -->|"network metrics"| NetworkFX[add_network_features_pyg]
    NetworkFX -->|"per-node aggregation"| NodeFX[NodeFeatureExtractor]
    NodeFX -->|"state dim"| StateDim[config.gnn.node_feature_dim]
```

## Bring Your Own Dataset

To plug a new transaction source into the pipeline, mirror the following structure:

1. **Loader**
   - Implement `BaseGraphDataset` (see `datasets/base.py`) and expose a `load_raw()` method that returns a `pandas.DataFrame` with, at minimum:
     - `nameOrig` (sender ID)
     - `nameDest` (receiver ID)
     - `amount` (transaction value)
     - `step` (integer timestep or sequence number)
     - Target label column (e.g. `isMoneyLaundering`, `isFraud`)
   - Optional columns that unlock richer features:
     - `hour` (0–23); otherwise the temporal utilities derive it from `step % 24`.
     - `type`, `category`, or other categorical descriptors to encode later.

2. **Graph construction**
   - Use `build_graph()` to emit a `networkx.DiGraph` with edge attributes:
     ```python
     {
         "amount": float(row.amount),
         "step": int(row.step),
         "is_money_laundering": int(row[target_col]),
         "isFraud": int(row[target_col]),  # compatibility alias
     }
     ```
   - If you maintain additional metadata (scheme IDs, channel codes), attach them here—`StateEncoder` will carry them into the PyG `Data` object.

3. **Temporal specification**
   - Call `add_temporal_features_to_graph(graph, transactions_view)` where `transactions_view` is a DataFrame with the columns `[nameOrig, nameDest, amount, step, hour?]`.
   - Ensure `step` is monotonic per account; the helper computes velocity (`transactions / Δstep`) and first/last timestamps.
   - Provide `hour` to get business-hour ratios; otherwise the function assumes `step % 24`.

4. **Network features**
   - Invoke `add_network_features_pyg(graph)` to populate `degree_centrality`, `clustering_coefficient`, `in_degree`, and `out_degree`. These feed the default feature extractor and multi-branch regularisers.

5. **Node features**
   - If the default `NodeFeatureExtractor` (amount statistics, degrees, risk score) does not fit your schema, subclass it and update `config.gnn.node_feature_dim` accordingly:
     ```python
     from rl_money_laundering.features.extractors import NodeFeatureExtractor

     class MyFeatureExtractor(NodeFeatureExtractor):
         def __init__(self):
             super().__init__(feature_names=[
                 "total_sent", "total_received", "avg_tx_amount",
                 "merchant_category", "region_code", "risk_score",
             ])
     ```
   - Inject your extractor into `StateEncoder.forward_state` by passing it via the trainer or by overriding `PipelineArtifacts.dataset`.

Once the loader returns a DataFrame + graph that satisfy these contracts, `build_pipeline()` handles the rest—temporal augmentation, node annotations, environment assembly, GNN selection, agent instantiation, and trainer wiring—without any additional code changes.

### Example: Elliptic Bitcoin dataset

The Elliptic integration follows the same checklist with a few dataset-specific twists:

- **Loader (`EllipticLoader`)** merges the raw feature, class, and edge CSVs, normalizes `txId`, and exposes `feature_pipeline()` which returns the 166 raw features plus six engineered temporal/statistical signals (`time_tx_density`, `time_illicit_ratio`, `local_mean/std`, `agg_mean/std`). Only labeled nodes are kept by default so the fraud rate mirrors the 2 % benchmark split.
- Pass `--include-unlabeled` (CLI) or set `include_unlabeled=true` in the config to keep the “unknown” nodes when building the graph. That restores the sparse prior (≈2 % positives) that the reward system expects; the flag is intended for Elliptic/Elliptic2 datasets only.
- **Graph construction** calls `build_graph(include_unknown=False, processed_features=processed_df)`. Passing the processed frame lets the loader attach all 172 attributes to each node, prune unlabeled vertices, and drop edges that point outside the labelled subgraph.
- **Feature extractor** uses `EllipticNodeFeatureExtractor(node_feature_dim=172)` so the encoder receives the exact vector emitted by the pipeline. The trainer now picks this extractor automatically whenever `dataset_type="elliptic"`, while AMLNet continues to rely on the 20-D `NodeFeatureExtractor`.
- **Config**: set `gnn.node_feature_dim` to 172 and point `dataset_path` to `data/elliptic`. The new preset `configs/elliptic_rmganets.json` mirrors the AMLNet QR-DQN/RMGANets stack but swaps in the Elliptic dataset, smaller curriculum ratios (`[0.5, 0.35, 0.2, 0.1, 0.05, 0.02]`), and an output bucket `outputs/elliptic_rmganets_fixed`.

With those pieces in place, `build_pipeline()` produces a fully wired stack (dataset → Elliptic feature extractor → state encoder → QR-DQN/RMGANets trainer) without any additional glue code.

## RMGANets Encoder (New)

RMGANets brings three complementary graph reasoning blocks—attention-based, topology-based, and hybrid correlation modules—followed by a fusion stage. The encoder mirrors the original paper while exposing clean hooks for the RL stack.

```{mermaid}
flowchart LR
    subgraph RMGANets["RMGANets Encoder"]
        ATT["Att-GCM<br/>(Attentional Graph Correlation)"]
        TO["To-GCM<br/>(Topological Correlation)"]
        HY["Hy-GCM<br/>(Hybrid Correlation)"]
        FUSE["Fusion Head<br/>(Adaptive Weighting)"]
    end
    SUBG["2-hop Ego Graph + Features"] --> ATT
    SUBG --> TO
    SUBG --> HY
    ATT --> FUSE
    TO --> FUSE
    HY --> FUSE
    FUSE --> EMBED["State Embedding"]
    EMBED --> HISTORY["Concat History Features"]
    HISTORY --> STATE_VEC["Final State Vector"]
```

- **Graph input** – the existing `StateEncoder` extracts a two-hop ego network and converts it to a PyTorch Geometric `Data` object shared across all backbones.
- **Att-GCM** – learns soft importance weights for neighbour interactions (paper Equations 5-7).
- **To-GCM** – models topology-aware similarities (Equations 8-9).
- **Hy-GCM** – fuses heterogeneous signals with feature-level attention (Equations 10-11).
- **Fusion head** – dynamically weights the three branches (Equation 12), producing the embedding passed to the RL agent.

Set `config.gnn.gnn_type = "rmganets"` or start `scripts/compare_gnn_architectures.sh` to benchmark GraphSAGE, GAT, and RMGANets side-by-side.

### Modular Architecture & Multi-Branch Loss

The RMGANets implementation is split into clean, testable modules under `gnn_modules/`:

- **`att_gcm.py`** – Attentional Graph Correlation Module (Equations 5-7)
- **`to_gcm.py`** – Topological Correlation Module (Equations 8-9)
- **`hy_gcm.py`** – Hybrid Correlation Module (Equations 10-11)
- **`fusion.py`** – Adaptive fusion layer (Equation 12)
- **`rmganets_encoder.py`** – Main encoder combining all modules
- **`rmganets_encoder_multibranch.py`** – Multi-branch variant with auxiliary classification heads
- **`multi_branch_loss.py`** – Loss functions for joint training

#### Multi-Branch Loss Variants

The system supports two multi-branch loss implementations:

1. **Paper-faithful** (`variant="paper"`) – Direct implementation of Equation 14:
   ```
   ζ_total = ζ_M + λ·(ζ_a + ζ_b) + ε·ζ_d
   ```
   where ζ_M is main cross-entropy, ζ_a and ζ_b are To-GCM and Hy-GCM auxiliary losses, and ζ_d is DQN alignment.

2. **Improved** (`variant="improved"`) – Enhanced for extreme class imbalance:
   - **Focal Loss** with adaptive α and γ for hard example mining
   - **Temporal-aware DQN loss** – weights recent transactions higher using temporal features
   - **Adaptive loss weighting** – adjusts λ and ε based on training progress
   - **Subgraph-aware regularization** – penalizes predictions that violate graph structure (β_reg term)

Enable in config:
```json
{
  "gnn": {
    "gnn_type": "rmganets",
    "multi_branch": true,
    "use_dqn_enhancement": true
  },
  "trainer": {
    "multi_branch": {
      "enabled": true,
      "variant": "improved",
      "lambda_branch": 0.25,
      "epsilon_dqn": 0.4,
      "beta_reg": 0.1,
      "temporal_decay": 0.1
    }
  }
}
```

### Joint GNN+RL Training

A key architectural improvement is **end-to-end training** of the GNN encoder with the RL agent:

```python
# In trainer.py
self.gnn_optimizer = torch.optim.Adam(
    self.state_encoder.gnn.parameters(),
    lr=1e-4,
)
```

During each training step:
1. **Forward pass** – GNN encodes current graph state
2. **RL update** – DQN/QR-DQN computes TD error and updates Q-network
3. **GNN update** – Multi-branch loss (if enabled) updates GNN parameters
4. **Backprop through encoder** – RL gradients flow back to GNN weights

This enables the GNN to learn representations optimized for the RL task, rather than pre-training on a separate supervised objective. The multi-branch loss provides additional supervision signals that prevent the encoder from overfitting to the RL signal alone.

```{mermaid}
flowchart LR
    subgraph Training["Joint Training Loop"]
        Graph["Graph State"] --> GNN["GNN Encoder<br/>(trainable)"]
        GNN --> Embed["State Embedding"]
        Embed --> RL["RL Agent<br/>(DQN/QRDQN)"]
        RL --> TDError["TD Error"]
        GNN --> AuxHeads["Auxiliary Heads<br/>(To-GCM, Hy-GCM)"]
        AuxHeads --> MBLoss["Multi-Branch Loss"]
        TDError --> GNNGrad["∂L/∂θ_GNN"]
        MBLoss --> GNNGrad
        GNNGrad --> GNNUpdate["Update GNN"]
    end
```

Benefits:
- **Task-aligned representations** – GNN learns features useful for fraud detection navigation
- **Reduced overfitting** – Multi-branch auxiliary signals regularize the encoder
- **Better sample efficiency** – Joint training exploits all available supervision
- **End-to-end optimization** – No separate pre-training phase required

## State Encoding

`StateEncoder` extracts a two-hop ego network around the current account, converts it to PyTorch Geometric `Data`, and applies the selected backbone (`sage`, `gat`, or `rmganets`). The learned embedding is concatenated with a 16-dimensional history vector covering:

- counts of visited nodes and edges,
- fraud edge density encountered so far,
- log-scaled transaction amount statistics,
- cyclical time encodings (`sin`/`cos` of the visit hour),
- aggregated node risk scores.

The result is a deterministic `state_dim` that feeds both agents.

## Learning Stack

- **Environment** – `AMLDetectionEnv` exposes discrete actions (neighbour moves plus `FLAG`) with curiosity bonuses, sparse terminal rewards, and rich debugging info.
- **Agents**
  - `DQNAgent` (default) couples the dueling head, Double DQN targets, and prioritized replay.
  - `QRDQNAgent` (optional) adds quantile regression, CVaR-based action selection, and n-step prioritized replay for tail-risk sensitivity.
- **Trainer & Monitoring** – `AMLTrainer` orchestrates the curriculum, evaluation sweeps, checkpointing (`outputs/checkpoints/*.pt`), and rolling metrics stored in `training_stats.npz`.

:::{admonition} Swap-in Paths
:class: tip
- Choose the encoder via `config.gnn.gnn_type` (`sage`, `gat`, or `rmganets`).
- Switch to the distributional agent by instantiating `QRDQNAgent` in `scripts/train_agent.py` (the trainer API remains unchanged).
- Run `scripts/compare_gnn_architectures.sh` to train all three backbones and compare metrics under identical settings.
:::

## RLlib Distributed Training

For production-scale deployments, the `rllib_integration/` module provides Ray RLlib wrappers:

```{mermaid}
flowchart TB
    subgraph RLlib["Ray RLlib Integration"]
        GraphSpace["GraphSpace<br/>(custom gym space)"]
        EnvWrapper["RLlibAMLEnv<br/>(config-based wrapper)"]
        GNNModule["GNNDQNModule<br/>(custom RLModule)"]
        FraudReplay["FraudAwareReplayBuffer"]
        Callbacks["FraudDetectionCallbacks"]
    end
    subgraph Core["Core Components"]
        AMLEnv["AMLDetectionEnv"]
        StateEnc["StateEncoder"]
    end
    subgraph Ray["Ray Infrastructure"]
        Workers["Parallel Workers<br/>(2-8 CPUs)"]
        Tune["Ray Tune<br/>(hyperparameter search)"]
        Serve["Ray Serve<br/>(deployment)"]
    end

    AMLEnv --> EnvWrapper
    StateEnc --> GNNModule
    GraphSpace --> EnvWrapper
    EnvWrapper --> GNNModule
    GNNModule --> FraudReplay
    FraudReplay --> Callbacks
    Callbacks --> Workers
    Workers --> Tune
    Workers --> Serve
```

Key components:

- **GraphSpace** – Custom Gymnasium space handling variable-size graphs with padding/masking
- **RLlibAMLEnv** – Config-based wrapper adapting `AMLDetectionEnv` for RLlib's vectorized API
- **GNNDQNModule** – Custom RLModule (Ray 2.40+ new API) integrating GNN encoder with DQN
- **FraudAwareReplayBuffer** – Extends RLlib's replay buffer with fraud-aware sampling
- **FraudDetectionCallbacks** – Tracks detection rate, false positives, and alert budgets during training

Usage:
```bash
python scripts/train_rllib.py \
  --dataset data/amlnet/AMLNet_August_2025.csv \
  --num-workers 4 \
  --num-gpus 1 \
  --training-iterations 1000
```

Benefits over standalone training:
- **2-8x speedup** from parallel environment rollouts
- **Multi-GPU scaling** for large graphs
- **Hyperparameter tuning** via Ray Tune (Population Based Training, ASHA, etc.)
- **Production deployment** with Ray Serve (REST API, model versioning, A/B testing)
- **Advanced algorithms** – easy to swap DQN → PPO, SAC, APPO, R2D2

See `docs/RLLIB_INTEGRATION.md` for detailed setup and benchmarks.

## Budgeted Evaluation Metrics

The `evaluation/` module provides compliance-oriented metrics that reflect real-world constraints:

```python
from rl_money_laundering.evaluation import BudgetedMetrics

metrics = BudgetedMetrics()

# Scenario: compliance team can investigate 100 transactions/day
recall_100 = metrics.recall_at_k(predictions, labels, k=100)
precision_100 = metrics.precision_at_k(predictions, labels, k=100)
aupr = metrics.aupr(predictions, labels)

print(f"With 100 alerts/day: catch {recall_100*100:.1f}% of fraud")
print(f"Precision: {precision_100*100:.1f}% of alerts are fraud")
print(f"AUPR (area under PR curve): {aupr:.3f}")
```

**Why budgeted metrics matter:**

Traditional metrics (accuracy, AUROC) are misleading for extreme imbalance (0.16% fraud rate):
- 99.84% accuracy achieved by always predicting "not fraud"
- AUROC insensitive to class distribution

Budgeted metrics answer the questions compliance teams actually ask:
- **Recall@K** – "If I investigate K cases/day, what % of fraud will I catch?"
- **Precision@K** – "What % of my K daily alerts will be true fraud?"
- **AUPR** – Area under precision-recall curve (better than AUROC for imbalanced data)

These metrics directly inform resource allocation and operational decisions.

## Control Loop

```{mermaid}
sequenceDiagram
    participant Trainer
    participant Encoder as StateEncoder
    participant Env as AMLDetectionEnv
    participant Agent as Policy (DQN or QRDQN)
    participant Replay as Replay Buffer
    participant Target as Target Net
    participant Eval as Evaluator

    Trainer->>Env: reset(seed)
    Env-->>Trainer: obs, info
    loop Episode Steps
        Trainer->>Encoder: encode_state(graph, current_node, history)
        Encoder-->>Trainer: state vector
        Trainer->>Agent: select_action(state, valid_actions, epsilon)
        Agent-->>Trainer: action
        Trainer->>Env: step(action)
        Env-->>Trainer: next_obs, reward, terminated?, truncated?, info
        Trainer->>Agent: store_transition(state, action, reward, next_state, done)
        alt buffer ready
            Agent->>Replay: sample(batch_size)
            Replay-->>Agent: transitions + weights
            Agent->>Target: compute bootstrap targets
            Agent->>Agent: optimise parameters
            Agent->>Replay: update_priorities(td_error)
        end
        alt target update tick
            Agent->>Target: sync weights
        end
    end
    alt evaluation tick
        Trainer->>Eval: run_eval(episodes=20)
        Eval-->>Trainer: metrics (reward, detection rate, FPR)
    end
    Trainer->>Checkpoint: save_model()
```

## Curriculum & Evaluation Schedule

`AMLTrainer` defaults to a seven-stage fraud-exposure curriculum. Early episodes seed fraud-heavy neighbourhoods; later stages approximate the real 2 % positive rate.

```{mermaid}
flowchart LR
    S1["Stage 1<br/>fraud_rate = 1.00<br/>seeded fraud hubs"] --> S2["Stage 2<br/>0.80"]
    S2 --> S3["Stage 3<br/>0.60"]
    S3 --> S4["Stage 4<br/>0.40"]
    S4 --> S5["Stage 5<br/>0.20"]
    S5 --> S6["Stage 6<br/>0.10"]
    S6 --> S7["Stage 7<br/>0.02<br/>production prior"]
```

- **Evaluation** – every `eval_frequency` episodes (default 50) the trainer runs 20 greedy episodes (`epsilon = 0`) and logs detection rate, false-positive rate, and reward.
- **Checkpointing** – `checkpoint_frequency` (default 100) persists intermediate weights; the final run emits `final_model.pt` for deployment.
- **Statistics** – aggregated rewards, lengths, and detection metrics populate compliance notebooks via `EvaluationReport` helpers.
