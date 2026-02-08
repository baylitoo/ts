"""
Integration Layer for Multi-Agent AML Detection

This module provides adapters and bridges to integrate the multiagent framework
with the existing RL-AML pipeline WITHOUT modifying the original codebase.

The integration follows an adapter pattern:
- Existing components are wrapped, not modified
- New functionality is layered on top
- Original behavior is preserved when multiagent features are disabled

Components:
----------
1. EnvironmentAdapter: Wraps AMLDetectionEnv for multiagent compatibility
2. EncoderBridge: Connects existing StateEncoder with JudgeModel
3. RewardIntegrator: Combines existing reward system with learned judge
4. TrainingOrchestrator: Coordinates training with both frameworks

Usage:
------
```python
from rl_money_laundering.multiagent.integration import (
    EnvironmentAdapter,
    EncoderBridge,
    RewardIntegrator,
    TrainingOrchestrator,
    IntegrationConfig,
)

# Wrap existing components
config = IntegrationConfig(enable_judge=True, judge_weight=0.2)
env_adapter = EnvironmentAdapter(existing_env, config)
reward_integrator = RewardIntegrator(existing_reward_fn, judge_model, config)

# Run training with integrated components
orchestrator = TrainingOrchestrator(
    env=env_adapter,
    agent=existing_agent,
    reward_integrator=reward_integrator,
    config=config,
)
orchestrator.train(num_episodes=1000)
```

Design Principles:
-----------------
- Zero modifications to existing code
- Graceful degradation when judge unavailable
- Compatible with existing checkpoints
- Preserves curriculum learning behavior
- Maintains fraud-aware sampling
"""

from .config import IntegrationConfig, JudgeIntegrationConfig, AdversaryConfig
from .environment_adapter import EnvironmentAdapter, EpisodeCollector
from .encoder_bridge import EncoderBridge, EmbeddingCache, EpisodeTextBuilder
from .reward_integrator import RewardIntegrator, RewardMetrics
from .training_orchestrator import TrainingOrchestrator, TrainingState, create_integrated_trainer

__all__ = [
    # Configuration
    "IntegrationConfig",
    "JudgeIntegrationConfig",
    "AdversaryConfig",
    # Environment
    "EnvironmentAdapter",
    "EpisodeCollector",
    # Encoder
    "EncoderBridge",
    "EmbeddingCache",
    "EpisodeTextBuilder",
    # Reward
    "RewardIntegrator",
    "RewardMetrics",
    # Training
    "TrainingOrchestrator",
    "TrainingState",
    "create_integrated_trainer",
]
