"""
Reward integrator combining existing rewards with learned judge.

Provides conservative reward combination with safety bounds,
uncertainty penalties, and hard metric anchoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ...protocols import JudgeModelProtocol
from ..reward_computation import BaseRewardComputer
from .config import IntegrationConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardMetrics:
    """Metrics from reward computation for logging and monitoring."""
    original_reward: float
    judge_reward: float
    judge_uncertainty: float
    hard_metric_reward: float
    combined_reward: float

    judge_weight: float
    was_clipped: bool
    uncertainty_penalty_applied: float

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    episode_id: Optional[str] = None
    step_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "original_reward": self.original_reward,
            "judge_reward": self.judge_reward,
            "judge_uncertainty": self.judge_uncertainty,
            "hard_metric_reward": self.hard_metric_reward,
            "combined_reward": self.combined_reward,
            "judge_weight": self.judge_weight,
            "was_clipped": self.was_clipped,
            "uncertainty_penalty": self.uncertainty_penalty_applied,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class RewardHistory:
    """Tracks reward history for drift detection and analysis."""
    max_size: int = 1000

    original_rewards: List[float] = field(default_factory=list)
    judge_rewards: List[float] = field(default_factory=list)
    combined_rewards: List[float] = field(default_factory=list)
    hard_metrics: List[float] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)

    def add(self, metrics: RewardMetrics) -> None:
        """Add metrics to history."""
        self.original_rewards.append(metrics.original_reward)
        self.judge_rewards.append(metrics.judge_reward)
        self.combined_rewards.append(metrics.combined_reward)
        self.hard_metrics.append(metrics.f1)
        self.uncertainties.append(metrics.judge_uncertainty)

        if len(self.original_rewards) > self.max_size:
            self.original_rewards.pop(0)
            self.judge_rewards.pop(0)
            self.combined_rewards.pop(0)
            self.hard_metrics.pop(0)
            self.uncertainties.pop(0)

    def get_correlation(self, window: int = 100) -> float:
        """Compute correlation between judge and hard metrics."""
        if len(self.judge_rewards) < window:
            return 1.0

        judge = np.array(self.judge_rewards[-window:])
        hard = np.array(self.hard_metrics[-window:])

        if np.std(judge) < 1e-6 or np.std(hard) < 1e-6:
            return 1.0

        return float(np.corrcoef(judge, hard)[0, 1])

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.original_rewards:
            return {}

        return {
            "count": len(self.original_rewards),
            "original_mean": np.mean(self.original_rewards),
            "judge_mean": np.mean(self.judge_rewards),
            "combined_mean": np.mean(self.combined_rewards),
            "hard_metric_mean": np.mean(self.hard_metrics),
            "uncertainty_mean": np.mean(self.uncertainties),
            "correlation": self.get_correlation(),
        }


class RewardIntegrator(BaseRewardComputer):
    """Integrates existing rewards with learned judge rewards.

    Implements conservative reward combination following the formula:
        R_total = R_original + α(t) * clip(R_judge - λ*σ, -τ, τ)

    Where:
        - R_original: Reward from existing environment
        - α(t): Annealed judge weight (increases over training)
        - R_judge: Learned judge reward
        - λ: Uncertainty penalty coefficient
        - σ: Judge uncertainty
        - τ: Clipping bound

    Safety features:
        - Hard metric anchoring ensures floor for detection quality
        - Clipping prevents judge exploitation
        - Uncertainty penalty discourages overconfident extrapolation
        - Drift detection monitors judge-metric alignment
    """

    def __init__(
        self,
        config: IntegrationConfig,
        judge_model: Optional[JudgeModelProtocol] = None,
        episode_text_builder: Optional[Callable[[Any], str]] = None,
    ) -> None:
        """Initialize reward integrator.

        Args:
            config: Integration configuration.
            judge_model: Judge model for episode evaluation.
            episode_text_builder: Function to convert episode to text.
        """
        self.config = config
        self._judge = judge_model
        self._text_builder = episode_text_builder

        self._current_episode = 0
        self._history = RewardHistory()
        self._drift_alarm_raised = False
        self._floor_violation_count = 0

    def set_judge(self, judge: JudgeModelProtocol) -> None:
        """Set or replace judge model."""
        self._judge = judge

    def set_text_builder(self, builder: Callable[[Any], str]) -> None:
        """Set or replace episode text builder."""
        self._text_builder = builder

    def compute_reward(
        self,
        episode_data: Any,
        *,
        original_reward: float = 0.0,
        true_labels: Optional[Dict[str, bool]] = None,
        graph_embedding: Optional[Any] = None,
        episode_text: Optional[str] = None,
    ) -> tuple[float, Dict[str, Any]]:
        """Canonical episode reward interface shared with reward computation paths."""
        del true_labels, graph_embedding
        combined_reward, metrics = self.compute_episode_reward(
            original_episode_reward=original_reward,
            episode_data=episode_data,
            episode_text=episode_text,
        )
        return combined_reward, metrics.to_dict()

    def compute_step_reward(
        self,
        original_reward: float,
        step_info: Dict[str, Any],
    ) -> float:
        """Compute reward for a single step.

        For step-level rewards, we preserve the original reward system
        since the judge operates at episode level. This maintains
        compatibility with the existing potential-based shaping.

        Args:
            original_reward: Reward from environment step.
            step_info: Info dictionary from step.

        Returns:
            Reward to use (original, possibly scaled).
        """
        if self.config.compatibility_mode:
            return original_reward

        return original_reward

    def compute_episode_reward(
        self,
        original_episode_reward: float,
        episode_data: Any,
        episode_text: Optional[str] = None,
    ) -> tuple[float, RewardMetrics]:
        """Compute combined reward for completed episode.

        This is the main entry point for episode-level reward computation.
        Combines original rewards with judge evaluation and hard metrics.

        Args:
            original_episode_reward: Sum of step rewards from environment.
            episode_data: EpisodeData object from collector.
            episode_text: Pre-built episode text (built if not provided).

        Returns:
            Tuple of (combined_reward, metrics).
        """
        self._current_episode += 1

        judge_reward = 0.0
        judge_uncertainty = 1.0
        judge_available = False

        if (
            self.config.judge_config.enable_judge
            and self._judge is not None
            and episode_data is not None
        ):
            if episode_text is None and self._text_builder is not None:
                episode_text = self._text_builder(episode_data)

            if episode_text:
                episode_dict = (
                    episode_data.to_dict()
                    if hasattr(episode_data, "to_dict")
                    else {"text": episode_text}
                )
                reward_value, metadata = self._judge.compute_reward(
                    episode=episode_dict,
                    conservative=True,
                )
                judge_reward = reward_value
                judge_uncertainty = metadata.get("uncertainty", 1.0)
                judge_available = True

        hard_metrics = self._compute_hard_metrics(episode_data)

        combined, metrics = self._combine_rewards(
            original_reward=original_episode_reward,
            judge_reward=judge_reward,
            judge_uncertainty=judge_uncertainty,
            judge_available=judge_available,
            hard_metrics=hard_metrics,
            episode_data=episode_data,
        )

        self._history.add(metrics)

        if self.config.safety_monitoring:
            self._check_safety(metrics)

        return combined, metrics

    def _compute_hard_metrics(self, episode_data: Any) -> Dict[str, float]:
        """Extract hard metrics from episode data."""
        if episode_data is None:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision = getattr(episode_data, "precision", 0.0)
        recall = getattr(episode_data, "recall", 0.0)
        f1 = getattr(episode_data, "f1_score", 0.0)

        return {"precision": precision, "recall": recall, "f1": f1}

    def _combine_rewards(
        self,
        original_reward: float,
        judge_reward: float,
        judge_uncertainty: float,
        judge_available: bool,
        hard_metrics: Dict[str, float],
        episode_data: Any,
    ) -> tuple[float, RewardMetrics]:
        """Combine reward components with safety bounds.

        Implements:
            R_total = w_hard * R_hard + α(t) * clip(R_judge - λ*σ, -τ, τ) + (1-w_hard) * R_original
        """
        jc = self.config.judge_config

        judge_weight = jc.get_judge_weight(self._current_episode)

        hard_metric_reward = (
            self.config.precision_weight * hard_metrics["precision"] +
            self.config.recall_weight * hard_metrics["recall"] +
            self.config.f1_weight * hard_metrics["f1"]
        )

        hard_metric_reward = 2 * hard_metric_reward - 1

        uncertainty_penalty = jc.uncertainty_penalty * judge_uncertainty
        penalized_judge = judge_reward - uncertainty_penalty

        clipped_judge = np.clip(penalized_judge, -jc.judge_clip_max, jc.judge_clip_max)
        was_clipped = abs(penalized_judge - clipped_judge) > 1e-6

        if hard_metrics["f1"] < self.config.hard_metric_floor:
            judge_weight = min(judge_weight, 0.1)
            self._floor_violation_count += 1

        if judge_available:
            combined = (
                self.config.hard_metric_weight * hard_metric_reward +
                judge_weight * clipped_judge +
                (1 - self.config.hard_metric_weight - judge_weight) * np.tanh(original_reward / 10)
            )
        else:
            combined = (
                self.config.hard_metric_weight * hard_metric_reward +
                (1 - self.config.hard_metric_weight) * np.tanh(original_reward / 10)
            )

        metrics = RewardMetrics(
            original_reward=original_reward,
            judge_reward=judge_reward,
            judge_uncertainty=judge_uncertainty,
            hard_metric_reward=hard_metric_reward,
            combined_reward=combined,
            judge_weight=judge_weight,
            was_clipped=was_clipped,
            uncertainty_penalty_applied=uncertainty_penalty,
            precision=hard_metrics["precision"],
            recall=hard_metrics["recall"],
            f1=hard_metrics["f1"],
            episode_id=getattr(episode_data, "episode_id", None),
            step_number=self._current_episode,
        )

        return combined, metrics

    def _check_safety(self, metrics: RewardMetrics) -> None:
        """Check safety conditions and raise alarms if needed."""
        correlation = self._history.get_correlation()

        if (
            correlation < self.config.drift_correlation_threshold
            and not self._drift_alarm_raised
            and len(self._history.judge_rewards) >= 100
        ):
            self._drift_alarm_raised = True
            logger.warning(
                f"DRIFT ALARM: Judge-metric correlation {correlation:.3f} "
                f"below threshold {self.config.drift_correlation_threshold}"
            )

        if metrics.f1 < self.config.hard_metric_floor:
            logger.warning(
                f"Hard metric floor violation: F1={metrics.f1:.3f} "
                f"(floor={self.config.hard_metric_floor})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get integrator statistics."""
        history_stats = self._history.get_stats()
        return {
            "current_episode": self._current_episode,
            "current_judge_weight": self.config.judge_config.get_judge_weight(
                self._current_episode
            ),
            "drift_alarm_raised": self._drift_alarm_raised,
            "floor_violations": self._floor_violation_count,
            "judge_enabled": self.config.judge_config.enable_judge,
            "judge_available": self._judge is not None,
            **history_stats,
        }

    def reset_episode_counter(self) -> None:
        """Reset episode counter (e.g., for new training run)."""
        self._current_episode = 0

    def clear_history(self) -> None:
        """Clear reward history."""
        self._history = RewardHistory()
        self._drift_alarm_raised = False
        self._floor_violation_count = 0

    def should_stop_training(self) -> tuple[bool, str]:
        """Check if training should stop due to safety violation.

        Returns:
            Tuple of (should_stop, reason).
        """
        if self._drift_alarm_raised:
            correlation = self._history.get_correlation()
            if correlation < 0:
                return True, f"Negative judge-metric correlation: {correlation:.3f}"

        if self._floor_violation_count > 100:
            recent_f1s = self._history.hard_metrics[-50:]
            if recent_f1s and np.mean(recent_f1s) < self.config.hard_metric_floor:
                return True, "Sustained hard metric floor violation"

        return False, ""
