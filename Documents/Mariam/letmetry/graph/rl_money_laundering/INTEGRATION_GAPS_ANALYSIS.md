# Full Integration Gap Analysis

**Date**: 2025-10-23
**Goal**: Unify agent creation via AgentProtocol, maintain backward compatibility, support both DQN and QR-DQN

---

## ✅ What's Already Done

### 1. AgentProtocol Definition ✅
**File**: `src/rl_money_laundering/agent.py` (Lines 29-73)

```python
@runtime_checkable
class AgentProtocol(Protocol):
    """Shared interface for RL agents used in the pipeline/trainer."""

    epsilon: float
    replay_buffer: Any
    training_steps: int
    episode_rewards: List[float]

    def select_action(...) -> int: ...
    def store_transition(...) -> None: ...
    def train_step(batch_size: int = 64) -> float: ...
    def update_target_network() -> None: ...
    def decay_epsilon() -> None: ...
    def get_statistics() -> Dict[str, Any]: ...
    def compute_q_values(states: torch.Tensor) -> torch.Tensor: ...
    def compute_target_q_values(states: torch.Tensor) -> torch.Tensor: ...
```

**Status**: ✅ Complete, well-defined, `@runtime_checkable`

### 2. AgentConfig with agent_type ✅
**File**: `src/rl_money_laundering/config.py` (Lines 18-69)

```python
@dataclass
class AgentConfig:
    agent_type: Literal["dqn", "qrdqn"] = "dqn"  # Line 55
    state_dim: int = 48
    action_dim: int = 6
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 100000
    device: str = "auto"

    def __post_init__(self):
        if self.agent_type not in ("dqn", "qrdqn"):  # Lines 68-69
            raise ValueError(f"Unknown agent_type: {self.agent_type}")
```

**Status**: ✅ Complete, validation included

### 3. Pipeline Agent Factory ✅
**File**: `src/rl_money_laundering/pipeline.py` (Lines 21, 175-226)

```python
from .agent import AgentProtocol, DQNAgent, QRDQNAgent  # Line 21

# Agent creation (Lines 175-226):
if config.agent.agent_type == "qrdqn":
    return QRDQNAgent(...)
else:
    return DQNAgent(...)
```

**Status**: ✅ Factory pattern implemented, returns AgentProtocol

---

## ❌ What's Missing

### 1. Trainer Type Annotations ❌
**File**: `src/rl_money_laundering/trainer.py` (Lines 15, 42)

**Current**:
```python
from .agent import DQNAgent  # Line 15

def __init__(
    self,
    graph: nx.DiGraph,
    agent: DQNAgent,  # Line 42 - CONCRETE TYPE!
    state_encoder: StateEncoder,
    ...
```

**Issue**: Trainer hardcoded to `DQNAgent`, not `AgentProtocol`

**What You Need to Change**:
```python
# Line 15: Import AgentProtocol
from .agent import AgentProtocol

# Line 42: Use protocol type
agent: AgentProtocol,
```

**Impact**:
- Trainer will accept any agent implementing the protocol
- Type checker will verify protocol compliance
- Enables DQN/QR-DQN/future agents without trainer changes

---

### 2. train_agent.py CLI Arguments ❌
**File**: `scripts/train_agent.py`

**Current**: Script gets agent from `pipeline.build_pipeline()` without CLI control

**What's Missing**:
```python
# Add CLI argument for agent type selection
parser.add_argument(
    "--agent-type",
    type=str,
    choices=["dqn", "qrdqn"],
    default="dqn",
    help="Agent type to use (dqn or qrdqn)"
)

# Override config with CLI arg
if args.agent_type is not None:
    config.agent.agent_type = args.agent_type
```

**Impact**:
- Users can select agent via CLI: `--agent-type qrdqn`
- Overrides JSON config for quick experiments
- Consistent with existing `--device`, `--nrows` overrides

---

### 3. train_agent_qrdqn.py Unification ❌
**File**: `scripts/train_agent_qrdqn.py` (313 lines)

**Current**: Separate script for QR-DQN with duplicate logic

**Issue**: Code duplication, maintenance burden, inconsistency

**What You Need**:
- **Option A (Recommended)**: Merge into `train_agent.py` with `--agent-type qrdqn`
  - Add QR-DQN-specific args (conditionally parsed)
  - Single entry point for all agents

- **Option B**: Keep separate but refactor common logic
  - Extract shared code into `_build_and_train()` helper
  - Both scripts call same underlying function

**QR-DQN Specific Args to Add** (if merging):
```python
# Only parsed if --agent-type qrdqn
qrdqn_group = parser.add_argument_group("QR-DQN options")
qrdqn_group.add_argument("--num-quantiles", type=int, default=128)
qrdqn_group.add_argument("--n-step", type=int, default=5)
qrdqn_group.add_argument("--risk-measure", choices=["mean", "cvar_90", "cvar_95"], default="mean")
qrdqn_group.add_argument("--positive-fraction", type=float, default=0.25)
qrdqn_group.add_argument("--hidden-dims", type=str, default="512,512,256")
qrdqn_group.add_argument("--prioritized-alpha", type=float, default=0.6)
qrdqn_group.add_argument("--prioritized-beta", type=float, default=0.4)
```

---

### 4. Config Validation for QR-DQN Parameters ❌
**File**: `src/rl_money_laundering/config.py`

**Current**: AgentConfig doesn't have QR-DQN-specific fields

**What's Missing**:
```python
@dataclass
class AgentConfig:
    agent_type: Literal["dqn", "qrdqn"] = "dqn"

    # Existing DQN params...
    learning_rate: float = 1e-4
    gamma: float = 0.99
    # ...

    # QR-DQN specific (optional fields)
    num_quantiles: Optional[int] = None  # Default: 128 for qrdqn
    n_step: Optional[int] = None  # Default: 5 for qrdqn
    risk_measure: Optional[str] = None  # Default: "mean"
    positive_fraction: Optional[float] = None  # Default: 0.25
    hidden_dims: Optional[List[int]] = None  # Default: [512, 512, 256]
    prioritized_alpha: Optional[float] = None  # Default: 0.6
    prioritized_beta: Optional[float] = None  # Default: 0.4

    def __post_init__(self):
        # Existing validation
        if self.agent_type not in ("dqn", "qrdqn"):
            raise ValueError(f"Unknown agent_type: {self.agent_type}")

        # Set QR-DQN defaults if type is qrdqn
        if self.agent_type == "qrdqn":
            if self.num_quantiles is None:
                self.num_quantiles = 128
            if self.n_step is None:
                self.n_step = 5
            if self.risk_measure is None:
                self.risk_measure = "mean"
            # ... etc
```

**Alternative**: Create separate `QRDQNConfig` dataclass
```python
@dataclass
class QRDQNConfig:
    num_quantiles: int = 128
    n_step: int = 5
    risk_measure: Literal["mean", "cvar_90", "cvar_95"] = "mean"
    positive_fraction: float = 0.25
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    prioritized_alpha: float = 0.6
    prioritized_beta: float = 0.4

@dataclass
class AgentConfig:
    agent_type: Literal["dqn", "qrdqn"] = "dqn"
    # ... existing fields ...
    qrdqn: QRDQNConfig = field(default_factory=QRDQNConfig)
```

---

### 5. Pipeline Agent Creation Needs QR-DQN Config ❌
**File**: `src/rl_money_laundering/pipeline.py` (Lines 175-226)

**Current**: QR-DQN creation uses hardcoded defaults

**What's Needed**:
```python
if config.agent.agent_type == "qrdqn":
    return QRDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_quantiles=config.agent.num_quantiles or 128,  # Use config
        n_step=config.agent.n_step or 5,
        risk_measure=config.agent.risk_measure or "mean",
        positive_fraction=config.agent.positive_fraction or 0.25,
        hidden_dims=config.agent.hidden_dims or [512, 512, 256],
        # ... rest from config
    )
```

---

### 6. Tests Need AgentProtocol Compatibility ❌
**Files**: `tests/test_*.py`

**Current**: Tests likely use concrete `DQNAgent` types

**What to Check**:
1. `tests/test_full_integration.py`: Does it test both agent types?
2. Do fixture return types use `AgentProtocol` or `DQNAgent`?
3. Are there tests for agent_type selection via config?

**What You Need**:
```python
@pytest.mark.parametrize("agent_type", ["dqn", "qrdqn"])
def test_pipeline_with_agent_type(agent_type):
    config = ExperimentConfig(...)
    config.agent.agent_type = agent_type

    artifacts = build_pipeline(config, ...)

    # Verify correct agent type created
    if agent_type == "dqn":
        assert isinstance(artifacts.agent, DQNAgent)
    else:
        assert isinstance(artifacts.agent, QRDQNAgent)

    # Verify protocol compliance
    assert isinstance(artifacts.agent, AgentProtocol)
```

---

### 7. Documentation Updates ❌
**Files**: `docs/`, `README.md`, `QUICKSTART.md`

**What's Missing**:
1. **AgentProtocol documentation**: How to implement new agents
2. **agent_type usage guide**: How to select agent in configs/CLI
3. **QR-DQN parameter guide**: What each parameter does
4. **Migration guide**: How to update existing configs

**Example Doc Section Needed**:
```markdown
## Agent Types

The framework supports multiple RL agents via `AgentProtocol`:

### Available Agents

1. **DQN** (`agent_type: "dqn"`)
   - Standard Deep Q-Network
   - Parameters: learning_rate, gamma, epsilon_*

2. **QR-DQN** (`agent_type: "qrdqn"`)
   - Distributional RL with Quantile Regression
   - Additional parameters: num_quantiles, n_step, risk_measure

### Using via Config

```json
{
  "agent": {
    "agent_type": "qrdqn",
    "num_quantiles": 128,
    "n_step": 5,
    "risk_measure": "cvar_90"
  }
}
```

### Using via CLI

```bash
python scripts/train_agent.py --agent-type qrdqn \
    --num-quantiles 128 --risk-measure cvar_90
```
```

---

### 8. JSON Config Examples Need agent_type ❌
**Files**: `configs/*.json`

**Current**: Configs may not have `agent_type` field

**What You Need**: Update all 5 configs
```json
// configs/amlnet_sage.json
{
  "name": "amlnet_sage",
  "agent": {
    "agent_type": "dqn",  // ADD THIS
    "learning_rate": 1e-4,
    "gamma": 0.99
  }
}

// configs/amlnet_rmganets_qrdqn.json (NEW)
{
  "name": "amlnet_rmganets_qrdqn",
  "agent": {
    "agent_type": "qrdqn",  // QR-DQN variant
    "learning_rate": 1e-4,
    "num_quantiles": 128,
    "n_step": 5,
    "risk_measure": "cvar_90",
    "positive_fraction": 0.3
  },
  "gnn": {
    "gnn_type": "rmganets",
    "multi_branch": true
  }
}
```

---

## 📋 Implementation Checklist

### High Priority (Must Do)

- [ ] **1. Update Trainer Type Annotation**
  - File: `src/rl_money_laundering/trainer.py`
  - Change: `agent: DQNAgent` → `agent: AgentProtocol`
  - Line: 15 (import), 42 (type hint)
  - Impact: Core unification, enables protocol-based design

- [ ] **2. Add CLI Agent Type Selection**
  - File: `scripts/train_agent.py`
  - Add: `--agent-type` argument with choices ["dqn", "qrdqn"]
  - Override: `config.agent.agent_type = args.agent_type`
  - Impact: User-facing feature, easy agent switching

- [ ] **3. Extend AgentConfig for QR-DQN**
  - File: `src/rl_money_laundering/config.py`
  - Add: Optional QR-DQN fields or separate QRDQNConfig
  - Validate: Set defaults in `__post_init__` if agent_type=="qrdqn"
  - Impact: Config-driven QR-DQN parameters

- [ ] **4. Update Pipeline Agent Creation**
  - File: `src/rl_money_laundering/pipeline.py`
  - Change: Use config values for QR-DQN parameters
  - Lines: 175-226
  - Impact: Config values actually used, not hardcoded

### Medium Priority (Should Do)

- [ ] **5. Unify or Refactor train_agent_qrdqn.py**
  - Option A: Merge into train_agent.py with conditional args
  - Option B: Extract common logic to shared helper
  - Impact: Reduce code duplication, single entry point

- [ ] **6. Add Agent Type Tests**
  - File: `tests/test_full_integration.py`
  - Add: Parameterized tests for ["dqn", "qrdqn"]
  - Verify: Protocol compliance, correct instantiation
  - Impact: Regression prevention, confidence in changes

- [ ] **7. Update JSON Configs**
  - Files: `configs/*.json` (all 5)
  - Add: `agent_type` field to each
  - Create: New QR-DQN variant configs
  - Impact: Config completeness, examples for users

### Low Priority (Nice to Have)

- [ ] **8. Document AgentProtocol**
  - File: `docs/agent.rst` or similar
  - Content: How to implement new agents, protocol spec
  - Impact: Extensibility documentation

- [ ] **9. Update QUICKSTART.md**
  - Add: Agent type selection guide
  - Add: QR-DQN parameter explanations
  - Impact: User-facing documentation

- [ ] **10. Create Migration Guide**
  - File: `MIGRATION.md`
  - Content: How to update old configs for agent_type
  - Impact: Backward compatibility communication

---

## 🔍 Current State Summary

### What Works:
✅ AgentProtocol defined with @runtime_checkable
✅ AgentConfig has agent_type field with validation
✅ Pipeline factory creates correct agent based on config
✅ Both DQN and QR-DQN implement protocol
✅ Type safety: mypy compliant agent implementations

### What's Broken/Incomplete:
❌ Trainer still typed as `DQNAgent` not `AgentProtocol`
❌ No CLI selection of agent type (must edit config)
❌ QR-DQN parameters hardcoded in pipeline, not from config
❌ train_agent_qrdqn.py duplicates train_agent.py logic
❌ Tests don't verify agent_type switching works
❌ JSON configs don't specify agent_type explicitly
❌ Documentation doesn't explain agent system

---

## 🎯 Recommended Implementation Order

### Phase 1: Core Unification (Must Do First)
1. **Trainer Type Update** (5 min)
   - Change line 15: `from .agent import AgentProtocol`
   - Change line 42: `agent: AgentProtocol,`
   - Run mypy to verify

2. **AgentConfig Extension** (15 min)
   - Add Optional[...] fields for QR-DQN params
   - Add defaults in __post_init__ if agent_type=="qrdqn"
   - Test config loading/validation

3. **Pipeline Config Usage** (10 min)
   - Update lines 175-226 to use config.agent.* values
   - Remove hardcoded QR-DQN defaults
   - Test with both agent types

### Phase 2: User-Facing Features
4. **CLI Agent Selection** (10 min)
   - Add --agent-type argument
   - Add QR-DQN specific args (conditional)
   - Override config with CLI values

5. **JSON Config Updates** (10 min)
   - Add agent_type to all 5 existing configs
   - Create 2 new QR-DQN variant configs
   - Verify all load correctly

### Phase 3: Quality & Docs
6. **Integration Tests** (20 min)
   - Parameterized test for ["dqn", "qrdqn"]
   - Verify protocol compliance
   - Test CLI overrides

7. **Documentation** (30 min)
   - AgentProtocol design doc
   - agent_type usage guide
   - QR-DQN parameter reference

8. **Script Unification** (30 min - optional)
   - Merge train_agent_qrdqn.py into train_agent.py
   - Or refactor common logic to helper
   - Update benchmark scripts to use --agent-type

---

## 🚦 Integration Verification

After implementing, verify with:

```bash
# 1. Type checking
mypy src/rl_money_laundering/trainer.py
mypy src/rl_money_laundering/pipeline.py

# 2. Test agent type switching
pytest tests/test_full_integration.py -k agent_type -v

# 3. CLI agent selection
python scripts/train_agent.py --config configs/quick_test.json --agent-type dqn --episodes 5
python scripts/train_agent.py --config configs/quick_test.json --agent-type qrdqn --episodes 5

# 4. Config-driven agent selection
python scripts/train_agent.py --config configs/amlnet_rmganets_qrdqn.json --episodes 10

# 5. Verify protocol compliance at runtime
python -c "
from rl_money_laundering.agent import AgentProtocol, DQNAgent, QRDQNAgent
assert isinstance(DQNAgent(...), AgentProtocol)
assert isinstance(QRDQNAgent(...), AgentProtocol)
print('✅ All agents implement AgentProtocol')
"
```

---

## 📊 Impact Matrix

| Change | Files | Lines | Mypy | Tests | Docs | Priority |
|--------|-------|-------|------|-------|------|----------|
| Trainer type | 1 | 2 | ✅ | ⚠️ | - | HIGH |
| CLI args | 1 | 20 | - | ⚠️ | ⚠️ | HIGH |
| Config fields | 1 | 30 | ✅ | ✅ | ⚠️ | HIGH |
| Pipeline usage | 1 | 15 | ✅ | - | - | HIGH |
| JSON configs | 5 | 25 | - | - | - | MED |
| Tests | 1 | 50 | ✅ | ✅ | - | MED |
| Script merge | 2 | 200 | ✅ | ✅ | ⚠️ | LOW |
| Documentation | 3 | 100 | - | - | ✅ | LOW |

**Total Estimate**: 2-3 hours for full integration

---

## 🎓 Key Design Decisions

### 1. Protocol vs ABC
**Choice**: `Protocol` with `@runtime_checkable`
**Rationale**: Structural subtyping, no inheritance required, mypy-friendly

### 2. Config Design
**Choice**: Optional fields in AgentConfig + defaults in __post_init__
**Alternative**: Separate QRDQNConfig dataclass
**Rationale**: Single config object simpler, less nesting

### 3. CLI Args
**Choice**: Conditional parsing based on --agent-type
**Alternative**: Always parse all args, ignore if not used
**Rationale**: Cleaner help output, less confusion

### 4. Script Unification
**Recommendation**: Merge train_agent_qrdqn.py into train_agent.py
**Rationale**: Single entry point, DRY, consistent UX

---

**Status**: Analysis complete. You have a clear roadmap for full integration!
