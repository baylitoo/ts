"""
Conservative Reward Computation for Multi-Agent AML Detection

This module implements safe reward shaping that combines:
1. Hard metric anchors (precision, recall, F1)
2. Learned judge shaping (soft labels, interpretability)
3. Safety mechanisms (clipping, uncertainty penalties, drift detection)

Key principle: R_total = R_hard_metric + α * R_judge_shaping
where α is scheduled/clipped to prevent reward hacking.

Safety Features:
- Hard metric anchoring prevents pure judge exploitation
- Conservative reward bounds penalize false positives heavily
- Ensemble disagreement penalties for uncertain predictions
- Diversity bonuses prevent repetitive flagging strategies
- Calibration monitoring detects reward drift

References:
- Ng et al. (1999): Policy invariance under reward transformations
- Hadfield-Menell et al. (2017): Inverse reward design
- Kumar et al. (2020): Conservative Q-Learning (CQL)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for conservative reward computation.

    The reward formula is:
        R_total = w_hard * R_hard + w_judge * R_judge + w_diversity * R_diversity
                  - w_uncertainty * uncertainty_penalty

    With safety constraints:
        - R_judge clipped to [judge_clip_min, judge_clip_max]
        - w_judge annealed from judge_weight_start to judge_weight_end
        - Alarm triggered if judge_score ↑ but hard_metric ↓

    Args:
        # Hard metric weights
        precision_weight: Weight for precision in hard metric
        recall_weight: Weight for recall in hard metric
        f1_weight: Weight for F1 in hard metric (typically dominant)
        false_positive_penalty: Additional penalty per false positive

        # Judge integration
        judge_weight_start: Initial weight for judge reward (start low)
        judge_weight_end: Final weight for judge reward (max allowed)
        judge_weight_warmup_episodes: Episodes to anneal judge weight
        judge_clip_min: Minimum clipped judge reward
        judge_clip_max: Maximum clipped judge reward

        # Safety mechanisms
        uncertainty_penalty_weight: Weight for uncertainty penalty
        uncertainty_threshold: Threshold above which to apply penalty
        diversity_bonus_weight: Weight for action diversity bonus
        hard_metric_floor: Minimum hard metric required (else alarm)

        # Monitoring
        drift_detection_window: Window size for drift detection
        drift_alarm_threshold: Correlation threshold for drift alarm
        log_frequency: How often to log reward statistics
    """

    # Hard metric weights
    precision_weight: float = 0.3
    recall_weight: float = 0.2
    f1_weight: float = 0.5
    false_positive_penalty: float = -0.5

    # Judge integration (conservative defaults)
    judge_weight_start: float = 0.1  # Start with low judge influence
    judge_weight_end: float = 0.3  # Cap judge influence
    judge_weight_warmup_episodes: int = 1000
    judge_clip_min: float = -0.5
    judge_clip_max: float = 0.5

    # Safety mechanisms
    uncertainty_penalty_weight: float = 0.1
    uncertainty_threshold: float = 0.3
    diversity_bonus_weight: float = 0.1
    hard_metric_floor: float = 0.1  # Minimum acceptable F1

    # Monitoring
    drift_detection_window: int = 100
    drift_alarm_threshold: float = -0.3  # Negative correlation = alarm
    log_frequency: int = 50

    # Episode-level settings
    max_flags_per_episode: int = 5
    flag_cooldown_steps: int = 3

    def get_judge_weight(self, episode: int) -> float:
        """Get annealed judge weight for given episode."""
        if episode >= self.judge_weight_warmup_episodes:
            return self.judge_weight_end

        progress = episode / self.judge_weight_warmup_episodes
        return self.judge_weight_start + progress * (
            self.judge_weight_end - self.judge_weight_start
        )


class BaseRewardComputer(ABC):
    """Abstract base for all episode-level reward computation paths.

    Subclasses must implement ``compute_reward`` with the canonical signature
    defined by ``RewardComputationProtocol``.  Two optional sub-component hooks
    are provided for subclasses that want shared helper structure.
    """

    @abstractmethod
    def compute_reward(
        self,
        episode_data: Any,
        *,
        original_reward: float = 0.0,
        true_labels: Optional[Dict[str, bool]] = None,
        graph_embedding: Optional[Any] = None,
        episode_text: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute episode reward and return detailed metadata."""
        ...

    def _compute_hard_metrics(self, episode_data: Any) -> Dict[str, float]:
        """Extract hard metrics from episode data.  Override in subclasses."""
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def _compute_judge_component(
        self,
        episode_data: Any,
        graph_embedding: Optional[Any] = None,
        episode_text: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return (judge_reward, uncertainty).  Override if a judge is available."""
        return 0.0, 1.0


class ConservativeRewardComputer(BaseRewardComputer):
    """Compute safe, anchored rewards for AML detection.

    This class combines hard metrics (precision/recall) with learned
    judge rewards while maintaining safety constraints to prevent
    reward hacking.

    Key safety features:
    1. Hard metric anchoring: Judge can only provide shaping, not replace metrics
    2. Conservative clipping: Judge rewards bounded to prevent exploitation
    3. Uncertainty penalties: Penalize high-uncertainty predictions
    4. Drift detection: Alarm when judge diverges from true metrics
    5. Diversity bonus: Prevent repetitive flagging strategies

    Usage:
        computer = ConservativeRewardComputer(config, judge_model)

        # During episode
        reward = computer.compute_step_reward(
            action=action,
            is_flag=is_flag,
            true_positive=true_positive,
            episode_state=state,
        )

        # End of episode
        total_reward, metrics = computer.compute_episode_reward(
            episode_data=episode_data,
            true_labels=true_labels,
        )

        # Check for alarms
        if computer.check_drift_alarm():
            logger.warning("Reward drift detected!")
    """

    def __init__(
        self,
        config: RewardConfig,
        judge_fn: Optional[Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]]] = None,
    ):
        """
        Args:
            config: RewardConfig with reward settings
            judge_fn: Optional function that takes episode dict and returns
                     (reward, metadata). If None, judge component is disabled.
        """
        self.config = config
        self.judge_fn = judge_fn

        # Episode counter for weight annealing
        self.episode_count: int = 0

        # Monitoring buffers
        self.hard_metric_history: List[float] = []
        self.judge_score_history: List[float] = []
        self.combined_reward_history: List[float] = []

        # Per-episode state
        self._episode_flags: int = 0
        self._episode_steps_since_flag: int = 0
        self._episode_flagged_nodes: set = set()
        self._episode_true_positives: int = 0
        self._episode_false_positives: int = 0

        # Alarm state
        self._drift_alarm_active: bool = False
        self._last_drift_correlation: float = 0.0

    def reset_episode(self) -> None:
        """Reset per-episode state. Call at start of each episode."""
        self._episode_flags = 0
        self._episode_steps_since_flag = 0
        self._episode_flagged_nodes = set()
        self._episode_true_positives = 0
        self._episode_false_positives = 0

    def compute_step_reward(
        self,
        action: int,
        is_flag_action: bool,
        current_node: str,
        is_true_positive: Optional[bool] = None,
        node_features: Optional[Dict[str, Any]] = None,
        visited_nodes: Optional[set] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward for a single step.

        Args:
            action: Action taken
            is_flag_action: Whether this was a FLAG action
            current_node: Current node ID
            is_true_positive: If FLAG, whether it was a true positive
            node_features: Features of current node
            visited_nodes: Set of already visited nodes

        Returns:
            Tuple of (reward, metadata_dict)
        """
        reward = 0.0
        metadata: Dict[str, Any] = {
            "action": action,
            "is_flag": is_flag_action,
            "step_reward_components": {},
        }

        self._episode_steps_since_flag += 1

        if is_flag_action:
            # FLAG action reward
            reward, flag_metadata = self._compute_flag_reward(
                current_node=current_node,
                is_true_positive=is_true_positive,
            )
            metadata.update(flag_metadata)
            self._episode_steps_since_flag = 0

        else:
            # MOVE action reward (small shaping)
            reward, move_metadata = self._compute_move_reward(
                current_node=current_node,
                node_features=node_features,
                visited_nodes=visited_nodes,
            )
            metadata.update(move_metadata)

        metadata["total_step_reward"] = reward
        return reward, metadata

    def _compute_flag_reward(
        self,
        current_node: str,
        is_true_positive: Optional[bool],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for FLAG action."""
        reward = 0.0
        metadata: Dict[str, Any] = {"flag_node": current_node}

        # Check cooldown
        if self._episode_steps_since_flag < self.config.flag_cooldown_steps:
            # Penalty for flagging too quickly
            reward = -0.5
            metadata["flag_penalty"] = "cooldown_violation"
            return reward, metadata

        # Check max flags
        if self._episode_flags >= self.config.max_flags_per_episode:
            reward = -0.3
            metadata["flag_penalty"] = "max_flags_exceeded"
            return reward, metadata

        # Check duplicate flag
        if current_node in self._episode_flagged_nodes:
            reward = -0.2
            metadata["flag_penalty"] = "duplicate_flag"
            return reward, metadata

        # Valid flag - compute reward based on correctness
        self._episode_flags += 1
        self._episode_flagged_nodes.add(current_node)

        if is_true_positive is True:
            # True positive: positive reward
            self._episode_true_positives += 1
            reward = 1.0  # Base TP reward
            metadata["flag_result"] = "true_positive"

        elif is_true_positive is False:
            # False positive: penalty
            self._episode_false_positives += 1
            reward = self.config.false_positive_penalty
            metadata["flag_result"] = "false_positive"

        else:
            # Unknown (no ground truth available at this step)
            # Give small positive reward to encourage exploration
            reward = 0.1
            metadata["flag_result"] = "unknown"

        metadata["step_reward_components"] = {
            "flag_reward": reward,
            "true_positives_so_far": self._episode_true_positives,
            "false_positives_so_far": self._episode_false_positives,
        }

        return reward, metadata

    def _compute_move_reward(
        self,
        current_node: str,
        node_features: Optional[Dict[str, Any]],
        visited_nodes: Optional[set],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for MOVE action (small shaping)."""
        reward = 0.0
        metadata: Dict[str, Any] = {}

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Diversity bonus for visiting new nodes
        if visited_nodes is not None and current_node not in visited_nodes:
            diversity_bonus = self.config.diversity_bonus_weight * 0.05
            reward += diversity_bonus
            metadata["diversity_bonus"] = diversity_bonus

        # Risk proximity bonus (if node features available)
        if node_features is not None:
            risk_score = node_features.get("risk_score", 0.0)
            if risk_score > 0.5:
                risk_bonus = self.config.diversity_bonus_weight * risk_score * 0.1
                reward += risk_bonus
                metadata["risk_proximity_bonus"] = risk_bonus

        metadata["step_reward_components"] = {
            "step_penalty": -0.01,
            "diversity_bonus": metadata.get("diversity_bonus", 0.0),
            "risk_proximity_bonus": metadata.get("risk_proximity_bonus", 0.0),
        }

        return reward, metadata

    def compute_reward(
        self,
        episode_data: Dict[str, Any],
        *,
        original_reward: float = 0.0,
        true_labels: Optional[Dict[str, bool]] = None,
        graph_embedding: Optional[Tensor] = None,
        episode_text: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Canonical reward interface compatible with integrator-style callers."""
        del original_reward, episode_text
        return self.compute_episode_reward(
            episode_data=episode_data,
            true_labels=true_labels,
            graph_embedding=graph_embedding,
        )

    def compute_episode_reward(
        self,
        episode_data: Dict[str, Any],
        true_labels: Optional[Dict[str, bool]] = None,
        graph_embedding: Optional[Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute total episode reward combining hard metrics and judge.

        Formula:
            R_total = w_hard * R_hard + w_judge * clip(R_judge) + w_div * R_diversity
                      - w_unc * uncertainty_penalty

        Args:
            episode_data: Episode dictionary with trajectory information
            true_labels: Ground truth labels for flagged nodes (for hard metrics)
            graph_embedding: Optional graph embedding for judge

        Returns:
            Tuple of (total_reward, detailed_metrics)
        """
        self.episode_count += 1
        judge_weight = self.config.get_judge_weight(self.episode_count)

        metrics: Dict[str, Any] = {
            "episode": self.episode_count,
            "judge_weight": judge_weight,
            "components": {},
        }

        # =====================================================================
        # Component 1: Hard Metrics (anchoring)
        # =====================================================================
        hard_reward, hard_metrics = self._compute_hard_metric_reward(
            episode_data=episode_data,
            true_labels=true_labels,
        )
        metrics["components"]["hard_metric"] = hard_metrics

        # =====================================================================
        # Component 2: Judge Reward (shaping, if available)
        # =====================================================================
        judge_reward = 0.0
        judge_uncertainty = 0.0
        judge_metadata: Dict[str, Any] = {}

        if self.judge_fn is not None:
            try:
                raw_judge_reward, judge_metadata = self.judge_fn(episode_data)

                # Clip judge reward
                judge_reward = np.clip(
                    raw_judge_reward,
                    self.config.judge_clip_min,
                    self.config.judge_clip_max,
                )

                judge_uncertainty = judge_metadata.get("uncertainty", 0.0)
                metrics["components"]["judge"] = {
                    "raw_reward": raw_judge_reward,
                    "clipped_reward": judge_reward,
                    "uncertainty": judge_uncertainty,
                    "fraud_type": judge_metadata.get("fraud_type"),
                }

            except Exception as e:
                logger.warning(f"Judge evaluation failed: {e}")
                judge_reward = 0.0
                metrics["components"]["judge"] = {"error": str(e)}

        # =====================================================================
        # Component 3: Diversity Bonus
        # =====================================================================
        diversity_reward = self._compute_diversity_reward(episode_data)
        metrics["components"]["diversity"] = diversity_reward

        # =====================================================================
        # Component 4: Uncertainty Penalty
        # =====================================================================
        uncertainty_penalty = 0.0
        if judge_uncertainty > self.config.uncertainty_threshold:
            uncertainty_penalty = (
                self.config.uncertainty_penalty_weight
                * (judge_uncertainty - self.config.uncertainty_threshold)
            )
        metrics["components"]["uncertainty_penalty"] = uncertainty_penalty

        # =====================================================================
        # Combine Components
        # =====================================================================
        hard_weight = 1.0 - judge_weight  # Hard metric gets remaining weight

        total_reward = (
            hard_weight * hard_reward
            + judge_weight * judge_reward
            + self.config.diversity_bonus_weight * diversity_reward
            - uncertainty_penalty
        )

        metrics["total_reward"] = total_reward
        metrics["weights"] = {
            "hard": hard_weight,
            "judge": judge_weight,
            "diversity": self.config.diversity_bonus_weight,
        }

        # =====================================================================
        # Update Monitoring
        # =====================================================================
        self._update_monitoring(
            hard_metric=hard_metrics.get("f1", 0.0),
            judge_score=judge_reward,
            combined_reward=total_reward,
        )

        # Check for floor violation
        if hard_metrics.get("f1", 0.0) < self.config.hard_metric_floor:
            metrics["alarm"] = "hard_metric_below_floor"
            logger.warning(
                f"Hard metric F1={hard_metrics.get('f1', 0.0):.3f} "
                f"below floor {self.config.hard_metric_floor}"
            )

        # Log periodically
        if self.episode_count % self.config.log_frequency == 0:
            self._log_statistics(metrics)

        return total_reward, metrics

    def _compute_hard_metric_reward(
        self,
        episode_data: Dict[str, Any],
        true_labels: Optional[Dict[str, bool]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward from hard metrics (precision, recall, F1)."""
        flagged_nodes = episode_data.get("flagged_nodes", set())
        if isinstance(flagged_nodes, list):
            flagged_nodes = set(flagged_nodes)

        metrics: Dict[str, Any] = {
            "num_flagged": len(flagged_nodes),
        }

        if not flagged_nodes or true_labels is None:
            # No flags or no ground truth - can't compute metrics
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
            return 0.0, metrics

        # Compute TP, FP, FN
        true_positives = sum(
            1 for node in flagged_nodes if true_labels.get(str(node), False)
        )
        false_positives = len(flagged_nodes) - true_positives

        # Count total positive labels in episode scope
        visited_nodes = episode_data.get("visited_nodes", set())
        if isinstance(visited_nodes, list):
            visited_nodes = set(visited_nodes)

        total_positives_in_scope = sum(
            1 for node in visited_nodes if true_labels.get(str(node), False)
        )
        false_negatives = total_positives_in_scope - true_positives

        # Compute metrics
        precision = (
            true_positives / len(flagged_nodes) if flagged_nodes else 0.0
        )
        recall = (
            true_positives / total_positives_in_scope
            if total_positives_in_scope > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics["true_positives"] = true_positives
        metrics["false_positives"] = false_positives
        metrics["false_negatives"] = false_negatives
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        # Compute reward
        reward = (
            self.config.precision_weight * precision
            + self.config.recall_weight * recall
            + self.config.f1_weight * f1
            + self.config.false_positive_penalty * false_positives
        )

        return reward, metrics

    def _compute_diversity_reward(self, episode_data: Dict[str, Any]) -> float:
        """Compute diversity bonus to prevent repetitive strategies."""
        visited_nodes = episode_data.get("visited_nodes", set())
        flagged_nodes = episode_data.get("flagged_nodes", set())
        actions = episode_data.get("actions", [])

        if isinstance(visited_nodes, list):
            visited_nodes = set(visited_nodes)
        if isinstance(flagged_nodes, list):
            flagged_nodes = set(flagged_nodes)

        diversity_score = 0.0

        # Exploration diversity: ratio of unique nodes visited
        if visited_nodes:
            episode_length = len(actions) if actions else 1
            exploration_ratio = len(visited_nodes) / max(episode_length, 1)
            diversity_score += min(exploration_ratio, 1.0) * 0.5

        # Flag diversity: penalize if all flags are clustered
        if len(flagged_nodes) > 1 and len(visited_nodes) > 0:
            flag_spread = len(flagged_nodes) / len(visited_nodes)
            diversity_score += min(flag_spread * 2, 0.5)

        # Action diversity: entropy of action distribution
        if actions:
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            total = len(actions)
            probs = [c / total for c in action_counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            max_entropy = np.log(len(action_counts) + 1)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            diversity_score += normalized_entropy * 0.3

        return diversity_score

    def _update_monitoring(
        self,
        hard_metric: float,
        judge_score: float,
        combined_reward: float,
    ) -> None:
        """Update monitoring buffers for drift detection."""
        self.hard_metric_history.append(hard_metric)
        self.judge_score_history.append(judge_score)
        self.combined_reward_history.append(combined_reward)

        # Keep only recent history
        window = self.config.drift_detection_window
        if len(self.hard_metric_history) > window:
            self.hard_metric_history = self.hard_metric_history[-window:]
            self.judge_score_history = self.judge_score_history[-window:]
            self.combined_reward_history = self.combined_reward_history[-window:]

    def check_drift_alarm(self) -> bool:
        """
        Check for reward drift (judge diverging from hard metrics).

        Alarm condition: Judge score increasing while hard metric decreasing
        (negative correlation over recent window).

        Returns:
            True if drift alarm is active
        """
        if len(self.hard_metric_history) < self.config.drift_detection_window // 2:
            return False

        # Compute correlation between judge score changes and hard metric changes
        hard_array = np.array(self.hard_metric_history)
        judge_array = np.array(self.judge_score_history)

        # Use differences to detect trends
        hard_diff = np.diff(hard_array)
        judge_diff = np.diff(judge_array)

        if len(hard_diff) < 10:
            return False

        # Correlation of changes
        if np.std(hard_diff) > 1e-6 and np.std(judge_diff) > 1e-6:
            correlation = np.corrcoef(hard_diff, judge_diff)[0, 1]
            self._last_drift_correlation = correlation

            # Alarm if strong negative correlation (judge ↑ while hard ↓)
            if correlation < self.config.drift_alarm_threshold:
                self._drift_alarm_active = True
                logger.warning(
                    f"DRIFT ALARM: Judge-HardMetric correlation = {correlation:.3f}"
                )
                return True

        self._drift_alarm_active = False
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get reward computation statistics for monitoring."""
        stats: Dict[str, Any] = {
            "episode_count": self.episode_count,
            "current_judge_weight": self.config.get_judge_weight(self.episode_count),
            "drift_alarm_active": self._drift_alarm_active,
            "last_drift_correlation": self._last_drift_correlation,
        }

        if self.hard_metric_history:
            stats["hard_metric_mean"] = np.mean(self.hard_metric_history)
            stats["hard_metric_std"] = np.std(self.hard_metric_history)

        if self.judge_score_history:
            stats["judge_score_mean"] = np.mean(self.judge_score_history)
            stats["judge_score_std"] = np.std(self.judge_score_history)

        if self.combined_reward_history:
            stats["combined_reward_mean"] = np.mean(self.combined_reward_history)
            stats["combined_reward_std"] = np.std(self.combined_reward_history)

        return stats

    def _log_statistics(self, metrics: Dict[str, Any]) -> None:
        """Log reward statistics."""
        stats = self.get_statistics()
        logger.info(
            f"Episode {self.episode_count}: "
            f"hard_metric={stats.get('hard_metric_mean', 0):.3f}, "
            f"judge_score={stats.get('judge_score_mean', 0):.3f}, "
            f"total_reward={metrics['total_reward']:.3f}, "
            f"drift_corr={stats['last_drift_correlation']:.3f}"
        )


# =============================================================================
# Utility Functions
# =============================================================================


def create_safe_reward_formula(
    hard_precision: float,
    hard_recall: float,
    judge_score: float,
    diversity_bonus: float,
    uncertainty: float,
    judge_weight: float = 0.3,
    uncertainty_threshold: float = 0.3,
    uncertainty_penalty_weight: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """
    Create a safe reward combining hard metrics and judge.

    Recommended formula:
        R_total = (
            0.6 * hard_precision +           # Ground truth anchor
            0.3 * judge_score +               # Learned shaping
            0.1 * diversity_bonus -           # Prevent repetition
            penalty_if_judge_uncertain        # Safety valve
        )

    Args:
        hard_precision: Precision from ground truth
        hard_recall: Recall from ground truth
        judge_score: Score from learned judge (should be pre-clipped)
        diversity_bonus: Bonus for diverse exploration
        uncertainty: Judge's uncertainty estimate
        judge_weight: Weight for judge component
        uncertainty_threshold: Threshold for applying uncertainty penalty
        uncertainty_penalty_weight: Weight for uncertainty penalty

    Returns:
        Tuple of (total_reward, component_breakdown)
    """
    # Compute hard F1
    hard_f1 = (
        2 * hard_precision * hard_recall / (hard_precision + hard_recall)
        if (hard_precision + hard_recall) > 0
        else 0.0
    )

    # Compute uncertainty penalty
    uncertainty_penalty = 0.0
    if uncertainty > uncertainty_threshold:
        uncertainty_penalty = (
            uncertainty_penalty_weight * (uncertainty - uncertainty_threshold)
        )

    # Combine components
    hard_weight = 1.0 - judge_weight - 0.1  # Reserve 0.1 for diversity
    total = (
        hard_weight * hard_f1
        + judge_weight * judge_score
        + 0.1 * diversity_bonus
        - uncertainty_penalty
    )

    components = {
        "hard_f1": hard_f1,
        "hard_contribution": hard_weight * hard_f1,
        "judge_contribution": judge_weight * judge_score,
        "diversity_contribution": 0.1 * diversity_bonus,
        "uncertainty_penalty": uncertainty_penalty,
        "total": total,
    }

    return total, components
