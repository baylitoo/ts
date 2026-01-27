# Extending the RL Money Laundering Pipeline

This note captures the key integration patterns we validated while wiring the Elliptic Bitcoin dataset into the AML pipeline. Use it as a reference whenever you need to add a new dataset, plug in a custom agent, or experiment with alternative GNN encoders (including RMGANets variants).

---

## 1. System Overview & Modularity

```
dataset loader ─┐     raw transactions  ─┐
                ├─ feature pipeline ─┐   ├─ graph builder ─┐
processor utils ┘                    │   ┘                │
                                      ▼                   ▼
                                node attributes     AMLDetectionEnv
                                      │                   │
                                      └─ feature extractor│
                                                          ▼
                              StateEncoder (SAGE/GAT/RMGANets)
                                                          │
                             ┌────────────────────────────┴────────────────────────────┐
                             │                                                        │
                         Agent (DQN / QR-DQN / custom)                          AMLTrainer
                             │                                                        │
                        Replay buffers, PER,                                 Curriculum, rewards,
                        QR head, etc.                                        checkpoints, logging
```

Everything flows through `build_pipeline` (`src/rl_money_laundering/pipeline.py`):

1. **DatasetArtifacts** – pairs a loader, processed transactions, and a `networkx.DiGraph`.
2. **Feature extractor** – `NodeFeatureExtractor` for AMLNet or `EllipticNodeFeatureExtractor` for Elliptic; you can inject your own extractor via the trainer.
3. **StateEncoder** – currently supports `sage`, `gat`, and `rmganets` (multi-branch or single). Any encoder just needs to expose `get_state_dim` and produce embeddings for the trainer.
4. **Agent** – `DQNAgent` or `QRDQNAgent` implement the `AgentProtocol`. New agents only need to honour the same interface.
5. **AMLTrainer** – orchestrates curriculum sampling, intrinsic rewards, PER updates, and logging. It expects an environment, agent, state encoder, and feature extractor.

Because each layer only depends on the abstraction below it, you can swap pieces independently (e.g., plug in a new agent while reusing the dataset or vice versa).

---

## 2. Adding Your Own Agent

Implement the `AgentProtocol` declared in `src/rl_money_laundering/agent.py`:

```python
@runtime_checkable
class AgentProtocol(Protocol):
    epsilon: float
    replay_buffer: Any
    training_steps: int
    episode_rewards: List[float]

    def select_action(...): ...
    def store_transition(...): ...
    def train_step(...): ...
    def update_target_network(...): ...
    def decay_epsilon(...): ...
    def get_statistics(...): ...
    def compute_q_values(...): ...
    def compute_target_q_values(...): ...
    def save(...): ...
    def load(...): ...
```

Minimum steps:

1. **Create the class** (e.g., `src/rl_money_laundering/agent_custom.py`) that implements the interface.
2. **Extend `_build_agent_from_config`** in `pipeline.py` so it can instantiate your agent based on a config knob (e.g., `agent_type="rainbow"`).
3. **Update configs** to declare the new type and any hyperparameters.
4. **(Optional) CLI overrides** – add parser flags in `scripts/train_agent.py` to let you tweak agent-specific fields without editing JSON.

As long as the protocol is satisfied, the trainer and environment do not care whether you are using vanilla DQN, QR-DQN, or a more exotic architecture.

---

## 3. Bringing Your Own Dataset

Follow the checklist under *Bring Your Own Dataset* in `docs/architecture.md`, plus the lessons from the Elliptic integration:

1. **Loader**
   - Subclass `BaseGraphDataset`.
   - Implement `load_raw()` (reads raw tables), `feature_pipeline()` (engineering + scaling), and optional helpers like `create_temporal_splits()`.
   - Decide whether to keep unlabeled rows. For AML-like sparsity you typically want `include_unlabeled=True`; otherwise the graph becomes fraud-heavy.

2. **Graph builder (`build_graph`)**
   - Always normalize your node IDs so they match between `transactions`, `processed_features`, and edges (we had to strip `.0` in Elliptic).
   - Accept `processed_features` and attach the engineered features to nodes so the feature extractor can pull them directly from the graph.
   - Tag edges with a consistent set of fraud labels:
     ```python
     fraud_flag = int(node_u.get("label_illicit", 0) or node_v.get("label_illicit", 0))
     graph.add_edge(u, v,
                    is_fraud=fraud_flag,
                    is_money_laundering=fraud_flag,
                    isFraud=fraud_flag)
     ```
     The environment only inspects those three keys; without them you will see zero positives (as we initially observed).

3. **Pipeline registration**
   - Add a `_build_<dataset>_dataset` helper in `pipeline.py` that returns `DatasetArtifacts`.
   - Wire it into `build_pipeline` (`cfg.dataset_type == "<name>"`).
   - Provide a config preset in `configs/*.json` setting `dataset_type`, `dataset_path`, `gnn.node_feature_dim`, curriculum schedule, etc.

4. **Feature extractor selection**
   - If the default 20-D AMLNet extractor doesn’t apply, create a `BaseNodeFeatureExtractor` subclass (see `features/elliptic_extractor.py`).
   - Pass it into the trainer via `build_pipeline` so history/state encodings use the right dimensionality.

5. **Reward tuning**
   - Verify the true fraud rate by counting `label`/`is_fraud` in your graph; if it’s much higher than AMLNet’s 0.27 %, consider lowering the intrinsic bonuses or adjusting the curriculum schedule (`TrainerConfig.curriculum_schedule`) to avoid trivial detection.

---

## 4. Custom Encoders & RMGANets Tweaks

`StateEncoder` is the sole consumer of the node feature dimension. You can:

1. **Switch backbones** – set `gnn.gnn_type` in the config (`"sage"`, `"gat"`, or `"rmganets"`). The encoder will select the corresponding PyG module or RMGANets implementation.
2. **Use multi-branch RMGANets** – enable `gnn.multi_branch` and configure `trainer.multi_branch` (variant, lambda, epsilons). The trainer will instantiate `RMGANetsMultiBranchEncoder`, auxiliary heads, and the `MultiBranchLoss`.
3. **Create a new encoder** – extend `StateEncoder` to accept your module:
   - Add a constructor branch (e.g., `elif gnn_type == "graphormer": self.gnn = GraphormerEncoder(...)`).
   - Ensure `forward_state` returns embeddings consistent with `embedding_dim`.
4. **Add feature dimensions** – update `gnn.node_feature_dim` to match the extractor output. The encoder validates the dimension and will raise a descriptive error if there’s a mismatch.

For RMGANets specifically you can tweak:

- Attention heads (`gnn.heads`), hidden width (`hidden_channels`), enhancement flags (`use_dqn_enhancement`).
- Multi-branch loss parameters (`trainer.multi_branch.lambda_branch`, `epsilon_dqn`, `beta_reg`, etc.) to rebalance the auxiliary heads.

---

## 5. Agent & Curriculum Diagnostics

When plugging unfamiliar datasets (e.g., Elliptic) you should:

1. **Check fraud prevalence** – `_precompute_fraud_info` prints the # of fraud nodes; if it’s zero you probably forgot to tag edges or kept no positives.
2. **Inspect `logs.logs`** – evaluation blocks list “Episodes With Fraud Present” and “Fraud Edge Encounter Rate”. If they read 0 % while the dataset definitely contains positives, revisit the loader.
3. **Align curriculum** – use `curriculum_schedule` to control how often the trainer seeds fraud-heavy episodes. Dense datasets should start at lower fractions; sparse ones can keep the AMLNet defaults.
4. **Match reward scale** – the flag reward/penalty magnitudes (±5 to ±10) assume rare positives. If you intentionally work on denser graphs, consider downscaling these values to prevent trivial positive rewards.

---

## 6. Summary Checklist

- [ ] New dataset → loader (normalize IDs, attach fraud labels), feature pipeline, graph builder, config preset.
- [ ] Feature extractor → ensure `node_feature_dim` matches config + encoder.
- [ ] Agent → implement `AgentProtocol`, register in pipeline, expose CLI overrides as needed.
- [ ] Encoder tweaks → adjust `StateEncoder` branches or create new modules in `gnn_modules/`.
- [ ] Documentation → update the “Bring Your Own Dataset” section (as done for Elliptic) so future contributors know the expected hooks.

These patterns kept the AMLNet and Elliptic integrations aligned without touching the trainer or reward code. Reuse them whenever you need to extend the system. Happy hunting!
