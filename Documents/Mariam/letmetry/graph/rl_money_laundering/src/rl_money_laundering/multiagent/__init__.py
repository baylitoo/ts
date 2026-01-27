"""
Multi-Agent Reinforcement Learning for AML Detection

This module provides a complete multi-agent RL framework for Anti-Money Laundering
detection, including:

1. **Learned Judge Model**: ModernBERT-based reward model for episode evaluation
   - JudgeModel: Fine-tuned encoder with reward + fraud-type heads
   - JudgeConfig: Configuration for model architecture
   - EpisodeSerializer: Canonical episode-to-text serialization (order-invariant)
   - GraphTextFusion: Hybrid graph + text encoding

2. **Multi-Agent Environment**: Detector vs Adversary game-theoretic training
   - MultiAgentAMLEnvironment: Gymnasium-compatible multi-agent env
   - DetectorAdversaryConfig: Configuration for agent interactions

3. **Conservative Reward Computation**: Safe reward shaping with hard metric anchoring
   - ConservativeRewardComputer: Combines hard metrics + judge + safety
   - RewardConfig: Configuration for reward formula weights

4. **Training Infrastructure**: Two-timescale training with safety monitoring
   - TwoTimescaleJudgeCallback: Slow judge updates, fast policy updates
   - SafetyMonitorCallback: Drift detection, calibration monitoring
   - JudgeMetricsCallback: Detailed judge performance logging

5. **Multi-Agent Coordination**: Coordination layer for GNN, Judge, and RL agents
   - MultiAgentCoordinator: Coordinates all agents during training/execution
   - GNNProposerAgent: Proposes suspicious nodes/edges based on graph structure
   - MessageBus: Communication protocol between agents
   - TrainingMode: CENTRALIZED, INDEPENDENT, CTDE modes

Architecture Overview:
----------------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                    Multi-Agent Coordination                      │
    │                                                                  │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
    │  │ GNN Agent   │───▶│  Message    │◀───│   Judge     │         │
    │  │ (Proposer)  │    │    Bus      │    │  (Scorer)   │         │
    │  └─────────────┘    └──────┬──────┘    └─────────────┘         │
    │        │                   │                   │                 │
    │        │    proposals      │    scores         │                 │
    │        └───────────────────┼───────────────────┘                 │
    │                            ▼                                     │
    │                    ┌─────────────┐                               │
    │                    │  RL Agent   │                               │
    │                    │(Investigator)│                              │
    │                    └──────┬──────┘                               │
    │                           │                                      │
    │                           ▼                                      │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
    │  │ Environment │◀──│   Reward    │◀──│  Callbacks  │           │
    │  │   (Graph)   │   │  Computer   │   │  (Safety)   │           │
    │  └─────────────┘   └─────────────┘   └─────────────┘           │
    └─────────────────────────────────────────────────────────────────┘

Key Safety Features:
-------------------
- Hard metric anchoring prevents pure judge exploitation
- Conservative reward bounds (clipping, uncertainty penalties)
- Two-timescale training prevents co-adaptation instability
- Drift detection alarms when judge diverges from true metrics
- Calibration monitoring catches reward degradation early

Recommended Usage:
-----------------

```python
from rl_money_laundering.multiagent import (
    JudgeModel,
    JudgeConfig,
    MultiAgentAMLEnvironment,
    DetectorAdversaryConfig,
    ConservativeRewardComputer,
    RewardConfig,
    TwoTimescaleJudgeCallback,
    SafetyMonitorCallback,
)

# 1. Initialize judge
judge_config = JudgeConfig(
    model_name="answerdotai/ModernBERT-base",
    use_lora=True,
    reward_clip=0.5,
)
judge = JudgeModel(judge_config)

# 2. Initialize environment
env_config = DetectorAdversaryConfig(
    max_steps=50,
    enable_adversary=False,  # Start with single-agent
)
env = MultiAgentAMLEnvironment(graph, env_config)

# 3. Initialize reward computer
reward_config = RewardConfig(
    judge_weight_start=0.1,
    judge_weight_end=0.3,
    judge_clip_max=0.5,
)
reward_computer = ConservativeRewardComputer(
    reward_config,
    judge_fn=judge.compute_reward,
)

# 4. Initialize callbacks
judge_callback = TwoTimescaleJudgeCallback(
    judge_model=judge,
    config=TwoTimescaleConfig(
        judge_update_frequency=100,
        judge_freeze_after=5000,
    ),
)
safety_callback = SafetyMonitorCallback(
    config=SafetyConfig(
        hard_metric_floor=0.1,
        drift_correlation_threshold=-0.3,
    ),
)

# 5. Training loop
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Get episode data for judge
    episode_data = env.get_episode_data()
    true_labels = env.get_true_labels()

    # Compute combined reward
    total_reward, metrics = reward_computer.compute_episode_reward(
        episode_data, true_labels
    )

    # Update callbacks
    judge_callback.on_episode_end(episode, episode_data, true_labels)
    safety_result = safety_callback.on_episode_end(
        episode,
        hard_metric=metrics["components"]["hard_metric"]["f1"],
        judge_score=metrics["components"]["judge"]["clipped_reward"],
        combined_reward=total_reward,
    )

    if safety_result["should_stop"]:
        print("Safety violation - stopping training")
        break
```

References:
----------
- Christiano et al. (2017): Deep RL from Human Preferences
- Ouyang et al. (2022): InstructGPT / RLHF reward modeling
- Ng et al. (1999): Policy invariance under reward transformations
- Kumar et al. (2020): Conservative Q-Learning (CQL)
"""

from .judge_model import (
    JudgeModel,
    JudgeConfig,
    EpisodeSerializer,
    GraphTextFusion,
)

from .multi_agent_env import (
    MultiAgentAMLEnvironment,
    DetectorAdversaryConfig,
    RLlibMultiAgentWrapper,
    CoordinatedMultiAgentEnv,
    create_coordinated_env,
)

from .training_callbacks import (
    TwoTimescaleJudgeCallback,
    TwoTimescaleConfig,
    SafetyMonitorCallback,
    SafetyConfig,
    JudgeMetricsCallback,
)

from .reward_computation import (
    ConservativeRewardComputer,
    RewardConfig,
    create_safe_reward_formula,
)

from .coordination import (
    TrainingMode,
    AgentRole,
    AgentMessage,
    NodeProposal,
    EdgeProposal,
    JudgeScore,
    CoordinationConfig,
    MessageBus,
    BaseAgent,
    GNNProposerAgent,
    MultiAgentCoordinator,
    ObservationAugmentor,
    CTDEWrapper,
    RMGANetsCoordinationAdapter,
    create_coordinated_training_setup,
)

__all__ = [
    # Judge model components
    "JudgeModel",
    "JudgeConfig",
    "EpisodeSerializer",
    "GraphTextFusion",
    # Multi-agent environment
    "MultiAgentAMLEnvironment",
    "DetectorAdversaryConfig",
    "RLlibMultiAgentWrapper",
    "CoordinatedMultiAgentEnv",
    "create_coordinated_env",
    # Training infrastructure
    "TwoTimescaleJudgeCallback",
    "TwoTimescaleConfig",
    "SafetyMonitorCallback",
    "SafetyConfig",
    "JudgeMetricsCallback",
    # Reward computation
    "ConservativeRewardComputer",
    "RewardConfig",
    "create_safe_reward_formula",
    # Multi-agent coordination
    "TrainingMode",
    "AgentRole",
    "AgentMessage",
    "NodeProposal",
    "EdgeProposal",
    "JudgeScore",
    "CoordinationConfig",
    "MessageBus",
    "BaseAgent",
    "GNNProposerAgent",
    "MultiAgentCoordinator",
    "ObservationAugmentor",
    "CTDEWrapper",
    "RMGANetsCoordinationAdapter",
    "create_coordinated_training_setup",
]
