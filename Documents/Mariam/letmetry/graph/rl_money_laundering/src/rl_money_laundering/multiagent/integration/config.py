"""
Configuration for multiagent integration layer.

Provides configuration classes that control how the multiagent components
integrate with the existing RL-AML framework.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JudgeIntegrationConfig:
    """Configuration for judge model integration.

    Controls how the learned judge interacts with the existing reward system.

    Attributes:
        enable_judge: Whether to use learned judge rewards.
        judge_model_path: Path to pretrained judge checkpoint.
        judge_weight_start: Initial weight for judge component (0-1).
        judge_weight_end: Final weight after warmup.
        judge_warmup_episodes: Episodes before reaching full judge weight.
        judge_clip_max: Maximum absolute judge reward contribution.
        uncertainty_penalty: Coefficient for uncertainty-based reward reduction.
        update_frequency: Episodes between judge fine-tuning updates.
        freeze_after_episodes: Freeze judge after this many episodes.
        use_ema: Use exponential moving average for judge inference.
        ema_decay: EMA decay rate for parameter smoothing.
        calibration_check_frequency: Episodes between calibration checks.
    """
    enable_judge: bool = True
    judge_model_path: Optional[str] = None
    judge_weight_start: float = 0.1
    judge_weight_end: float = 0.3
    judge_warmup_episodes: int = 5000
    judge_clip_max: float = 0.5
    uncertainty_penalty: float = 0.1
    update_frequency: int = 100
    freeze_after_episodes: int = 5000
    use_ema: bool = True
    ema_decay: float = 0.995
    calibration_check_frequency: int = 500

    def get_judge_weight(self, episode: int) -> float:
        """Compute annealed judge weight for given episode."""
        if not self.enable_judge:
            return 0.0

        if episode >= self.judge_warmup_episodes:
            return self.judge_weight_end

        progress = episode / self.judge_warmup_episodes
        return self.judge_weight_start + progress * (self.judge_weight_end - self.judge_weight_start)


@dataclass
class AdversaryConfig:
    """Configuration for adversary agent (optional).

    Controls adversarial training for robustness.

    Attributes:
        enable_adversary: Whether to train with adversary.
        adversary_type: Type of adversary ('random', 'learned', 'curriculum').
        adversary_strength: How aggressively adversary injects fraud (0-1).
        adversary_update_frequency: Episodes between adversary updates.
        adversary_reward_scale: Scale factor for adversary rewards.
        max_injections_per_episode: Maximum fraud injections per episode.
        injection_types: Types of fraud patterns to inject.
    """
    enable_adversary: bool = False
    adversary_type: str = "learned"
    adversary_strength: float = 0.5
    adversary_update_frequency: int = 50
    adversary_reward_scale: float = 1.0
    max_injections_per_episode: int = 5
    injection_types: List[str] = field(default_factory=lambda: [
        "structuring",
        "round_tripping",
        "layering",
        "shell_company",
    ])


@dataclass
class IntegrationConfig:
    """Master configuration for multiagent integration.

    Bundles all integration settings and provides serialization.

    Attributes:
        judge_config: Judge model integration settings.
        adversary_config: Adversary agent settings.
        episode_buffer_size: Size of episode replay buffer for judge training.
        preserve_original_rewards: Keep original rewards alongside judge.
        hard_metric_weight: Weight for hard metrics (precision/recall/F1).
        precision_weight: Weight for precision in hard metric computation.
        recall_weight: Weight for recall (typically higher for AML).
        f1_weight: Weight for F1 score.
        safety_monitoring: Enable drift and calibration monitoring.
        drift_correlation_threshold: Minimum correlation before drift alarm.
        hard_metric_floor: Minimum F1 before reducing judge influence.
        log_detailed_metrics: Log per-episode judge metrics.
        checkpoint_dir: Directory for integration checkpoints.
        compatibility_mode: Preserve exact behavior when judge disabled.
    """
    judge_config: JudgeIntegrationConfig = field(default_factory=JudgeIntegrationConfig)
    adversary_config: AdversaryConfig = field(default_factory=AdversaryConfig)

    episode_buffer_size: int = 1000
    preserve_original_rewards: bool = True

    hard_metric_weight: float = 0.7
    precision_weight: float = 0.2
    recall_weight: float = 0.3
    f1_weight: float = 0.5

    safety_monitoring: bool = True
    drift_correlation_threshold: float = 0.3
    hard_metric_floor: float = 0.1

    log_detailed_metrics: bool = True
    checkpoint_dir: Optional[str] = None
    compatibility_mode: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.judge_config, dict):
            self.judge_config = JudgeIntegrationConfig(**self.judge_config)
        if isinstance(self.adversary_config, dict):
            self.adversary_config = AdversaryConfig(**self.adversary_config)

        total_weight = self.precision_weight + self.recall_weight + self.f1_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Hard metric weights sum to {total_weight}, normalizing to 1.0"
            )
            self.precision_weight /= total_weight
            self.recall_weight /= total_weight
            self.f1_weight /= total_weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationConfig":
        """Create from dictionary."""
        judge_data = data.pop("judge_config", {})
        adversary_data = data.pop("adversary_config", {})

        return cls(
            judge_config=JudgeIntegrationConfig(**judge_data),
            adversary_config=AdversaryConfig(**adversary_data),
            **data,
        )

    def save(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved integration config to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "IntegrationConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def default_conservative(cls) -> "IntegrationConfig":
        """Create conservative configuration prioritizing safety."""
        return cls(
            judge_config=JudgeIntegrationConfig(
                enable_judge=True,
                judge_weight_start=0.05,
                judge_weight_end=0.2,
                judge_warmup_episodes=10000,
                judge_clip_max=0.3,
                uncertainty_penalty=0.15,
            ),
            adversary_config=AdversaryConfig(enable_adversary=False),
            hard_metric_weight=0.8,
            safety_monitoring=True,
            drift_correlation_threshold=0.4,
        )

    @classmethod
    def default_aggressive(cls) -> "IntegrationConfig":
        """Create aggressive configuration maximizing judge influence."""
        return cls(
            judge_config=JudgeIntegrationConfig(
                enable_judge=True,
                judge_weight_start=0.2,
                judge_weight_end=0.4,
                judge_warmup_episodes=2000,
                judge_clip_max=0.7,
                uncertainty_penalty=0.05,
            ),
            adversary_config=AdversaryConfig(
                enable_adversary=True,
                adversary_strength=0.7,
            ),
            hard_metric_weight=0.6,
            safety_monitoring=True,
        )

    @classmethod
    def disabled(cls) -> "IntegrationConfig":
        """Create configuration with all multiagent features disabled."""
        return cls(
            judge_config=JudgeIntegrationConfig(enable_judge=False),
            adversary_config=AdversaryConfig(enable_adversary=False),
            compatibility_mode=True,
        )
