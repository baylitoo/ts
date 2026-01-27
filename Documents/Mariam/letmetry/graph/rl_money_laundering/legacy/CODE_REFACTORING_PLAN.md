# Code Refactoring Plan - Modularity & Reusability

## Current State Analysis

### ✅ Well-Structured Components

1. **datasets/** module (GOOD!)
   - ✓ Base classes (`BaseGraphDataset`)
   - ✓ Config dataclasses (`AMLNetConfig`, `EllipticConfig`)
   - ✓ Shared feature utilities (`features.py`)
   - ✓ Clean separation (amlnet.py, elliptic.py)

2. **Core Components**
   - `agent.py` - DQN agent (429 lines)
   - `gnn_encoder.py` - State encoding (382 lines)
   - `environment.py` - RL environment (419 lines)
   - `trainer.py` - Training pipeline (389 lines)
   - `baselines.py` - Baseline methods (442 lines)
   - `evaluator.py` - Metrics (251 lines)

### ⚠️ Issues Identified

1. **Duplicate Data Loaders**
   - `data_loader.py` (68 lines) - OLD AMLNet loader
   - `elliptic_loader.py` (72 lines) - OLD Elliptic loader
   - `datasets/amlnet.py` - NEW AMLNet loader
   - `datasets/elliptic.py` - NEW Elliptic loader
   - **Problem**: Two versions of same functionality!

2. **Missing Config Management**
   - Agent, Trainer, GNN have hardcoded params
   - No centralized configuration
   - Difficult to reproduce experiments

3. **Code Duplication**
   - Graph building logic duplicated
   - Feature extraction duplicated
   - Node feature extraction in multiple places

4. **Missing Utilities**
   - No common graph utilities
   - No experiment tracking
   - No checkpoint management utilities

5. **Inconsistent Interfaces**
   - Some modules use configs, some use kwargs
   - Inconsistent return types
   - Mixed naming conventions

---

## Refactoring Plan

### Phase 1: Consolidate Data Loaders ✅

**Action**: Remove old loaders, keep datasets/ module

```bash
# DELETE (redundant):
rm src/rl_money_laundering/data_loader.py
rm src/rl_money_laundering/elliptic_loader.py

# KEEP (modern):
datasets/amlnet.py
datasets/elliptic.py
datasets/base.py
datasets/features.py
```

### Phase 2: Create Config System

**Create**: `src/rl_money_laundering/config.py`

```python
@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 10000
    use_prioritized_replay: bool = True
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])

@dataclass
class GNNConfig:
    node_feature_dim: int
    gnn_type: str = "sage"  # "sage" or "gat"
    embedding_dim: int = 32
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.2

@dataclass
class TrainerConfig:
    num_episodes: int = 1000
    batch_size: int = 64
    eval_frequency: int = 50
    checkpoint_frequency: int = 100
    target_update_frequency: int = 10
    curriculum_schedule: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.02])
    max_steps_per_episode: int = 20

@dataclass
class ExperimentConfig:
    """Master config for full experiment"""
    dataset: AMLNetConfig | EllipticConfig
    agent: AgentConfig
    gnn: GNNConfig
    trainer: TrainerConfig
    output_dir: Path
    device: str = "auto"
    seed: int = 42
```

**Benefits**:
- Single source of truth
- Easy to save/load experiments
- Type-checked configurations
- Self-documenting

### Phase 3: Extract Common Utilities

**Create**: `src/rl_money_laundering/utils/`

```
utils/
├── __init__.py
├── graph.py          # Graph operations (subgraph extraction, etc.)
├── metrics.py        # Common metric calculations
├── visualization.py  # Plotting utilities
├── checkpoint.py     # Save/load models
└── experiment.py     # Experiment tracking
```

**graph.py**:
```python
def extract_k_hop_subgraph(
    graph: nx.DiGraph,
    center_node: str,
    k: int = 2
) -> nx.DiGraph:
    """Extract k-hop neighborhood around node"""
    # Move from gnn_encoder.py

def build_pyg_data(
    graph: nx.DiGraph,
    nodes: List[str],
    node_feature_extractor: Callable
) -> Data:
    """Convert NetworkX to PyTorch Geometric"""
    # Move from multiple places

def get_fraud_subgraphs(
    graph: nx.DiGraph,
    fraud_edges: List[Tuple],
    k_hop: int = 2
) -> List[nx.DiGraph]:
    """Extract subgraphs around fraud"""
    # Move from loaders
```

**checkpoint.py**:
```python
class CheckpointManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / "checkpoints"

    def save(
        self,
        agent: DQNAgent,
        config: ExperimentConfig,
        episode: int,
        metrics: Dict
    ):
        """Save model + config + metrics"""

    def load_latest(self) -> Tuple[DQNAgent, ExperimentConfig, int]:
        """Load most recent checkpoint"""

    def load_best(self, metric: str = "f1_score") -> Tuple[DQNAgent, ExperimentConfig]:
        """Load checkpoint with best metric"""
```

**experiment.py**:
```python
class ExperimentTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_file = output_dir / "metrics.json"

    def log_episode(self, episode: int, metrics: Dict):
        """Log episode metrics"""

    def log_evaluation(self, episode: int, results: Dict):
        """Log evaluation results"""

    def get_summary(self) -> Dict:
        """Get experiment summary"""
```

### Phase 4: Standardize Interfaces

**Before** (inconsistent):
```python
# agent.py
agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99, ...)

# gnn_encoder.py
encoder = StateEncoder(node_feature_dim, gnn_type="sage", ...)

# trainer.py
trainer = AMLTrainer(graph, agent, encoder, env, fraud_subgraphs, ...)
```

**After** (consistent):
```python
# All use config objects
agent = DQNAgent(config=agent_config)
encoder = StateEncoder(config=gnn_config)
trainer = AMLTrainer(config=trainer_config, ...)
```

### Phase 5: Reduce Duplication

**Duplicate Feature Extraction** (4 places):
1. `datasets/features.py` (shared utilities) ✓
2. `trainer.py` (node_feature_extractor method)
3. `baselines.py` (extract_features in RF)
4. `gnn_encoder.py` (node_feature_extractor param)

**Solution**: Create `FeatureExtractor` class
```python
class NodeFeatureExtractor:
    """Standardized node feature extraction"""

    def __init__(self, feature_set: str = "default"):
        self.feature_set = feature_set

    def extract(self, node_data: Dict) -> np.ndarray:
        """Extract features from node"""
        if self.feature_set == "default":
            return self._default_features(node_data)
        elif self.feature_set == "extended":
            return self._extended_features(node_data)

    def _default_features(self, node_data: Dict) -> np.ndarray:
        # Account type (5 features)
        # Transaction stats (7 features)
        # Risk scores (2 features)
        pass
```

---

## Implementation Order

### Week 1: Critical Cleanup
1. ✅ Remove duplicate loaders
2. ✅ Create config.py
3. ✅ Update agent.py to use config
4. ✅ Update scripts to use config

### Week 2: Utilities
5. ✅ Create utils/ module
6. ✅ Extract graph utilities
7. ✅ Create checkpoint manager
8. ✅ Add experiment tracker

### Week 3: Standardization
9. ✅ Standardize all interfaces
10. ✅ Create feature extractor class
11. ✅ Update documentation
12. ✅ Add type hints everywhere

---

## File Structure (After Refactoring)

```
src/rl_money_laundering/
├── __init__.py
│
├── datasets/                  # ✓ Already good!
│   ├── __init__.py
│   ├── base.py
│   ├── features.py
│   ├── amlnet.py
│   └── elliptic.py
│
├── models/                    # Rename from scattered files
│   ├── __init__.py
│   ├── agent.py              # DQN agent
│   ├── networks.py           # Neural network architectures
│   └── gnn_encoder.py        # GNN-based encoders
│
├── environment/               # RL environment
│   ├── __init__.py
│   ├── base.py               # Base environment
│   └── aml_env.py           # AML-specific environment
│
├── training/                  # Training components
│   ├── __init__.py
│   ├── trainer.py            # Main trainer
│   └── replay_buffer.py      # Experience replay
│
├── baselines/                 # Baseline methods
│   ├── __init__.py
│   ├── static.py             # PageRank, etc.
│   └── gnn.py               # Static GNN baselines
│
├── evaluation/                # Evaluation framework
│   ├── __init__.py
│   ├── metrics.py            # Metric calculations
│   └── evaluator.py          # Main evaluator
│
├── utils/                     # NEW! Utilities
│   ├── __init__.py
│   ├── graph.py              # Graph operations
│   ├── features.py           # Feature extraction
│   ├── checkpoint.py         # Model saving/loading
│   ├── experiment.py         # Experiment tracking
│   └── visualization.py      # Plotting
│
└── config.py                  # NEW! Config classes
```

---

## Benefits

### 1. Maintainability
- Single source of truth for each functionality
- Easy to find and fix bugs
- Clear module boundaries

### 2. Reusability
- Utils can be used across experiments
- Configs can be shared/reused
- Standard interfaces for all components

### 3. Testability
- Each module can be tested independently
- Mock dependencies easily
- Clear input/output contracts

### 4. Reproducibility
- Save entire experiment config
- Load exact same setup
- Version control configs

### 5. Extensibility
- Easy to add new datasets (inherit from base)
- Easy to add new agents (implement interface)
- Plug-and-play components

---

## Backward Compatibility

**Keep for now** (mark deprecated):
- `data_generator.py` (synthetic generator)
- Scripts maintain same CLI

**Migration path**:
```python
# Old way (still works)
from rl_money_laundering.data_loader import AMLNetDataLoader
loader = AMLNetDataLoader(path)

# New way (recommended)
from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
config = AMLNetConfig(csv_path=path)
loader = AMLNetLoader(config)
```

---

## Testing Strategy

```python
# tests/test_config.py
def test_agent_config_defaults():
    config = AgentConfig(state_dim=10, action_dim=5)
    assert config.learning_rate == 1e-3
    assert config.gamma == 0.99

# tests/test_utils_graph.py
def test_k_hop_subgraph():
    graph = create_test_graph()
    subgraph = extract_k_hop_subgraph(graph, "node1", k=2)
    assert len(subgraph.nodes()) == expected_size

# tests/test_checkpoint.py
def test_save_load_checkpoint():
    manager = CheckpointManager(tmp_path)
    manager.save(agent, config, episode=100, metrics={})
    loaded_agent, loaded_config, episode = manager.load_latest()
    assert episode == 100
```

---

## Documentation Updates

Update docstrings to follow Google style:

```python
def extract_k_hop_subgraph(
    graph: nx.DiGraph,
    center_node: str,
    k: int = 2
) -> nx.DiGraph:
    """Extract k-hop neighborhood subgraph around a node.

    Args:
        graph: Full transaction graph
        center_node: Node ID to center subgraph around
        k: Number of hops to include (default: 2)

    Returns:
        Subgraph containing k-hop neighborhood

    Raises:
        ValueError: If center_node not in graph

    Example:
        >>> graph = create_graph()
        >>> subgraph = extract_k_hop_subgraph(graph, "ACC_0001", k=2)
        >>> print(subgraph.number_of_nodes())
        25
    """
```

---

## Next Steps

1. Review this plan
2. Approve refactoring approach
3. Execute phase-by-phase
4. Test each phase before moving on
5. Update documentation as we go

**Estimated time**: 2-3 days for full refactor
**Priority**: Medium (code works, but will improve long-term)

Should I proceed with Phase 1 (consolidation)?
