"""
Training Callbacks for Multi-Agent AML Detection

This module provides callbacks for:
1. Two-timescale training (detector fast, judge slow)
2. Safety monitoring (drift detection, calibration)
3. Judge metrics logging

Key Principles:
- Judge updates less frequently than policy (prevents co-adaptation instability)
- Hard metric floors prevent judge exploitation
- Calibration monitoring catches reward drift early

References:
- Borkar (1997): Two-timescale stochastic approximation
- Haarnoja et al. (2018): SAC for stable off-policy learning
- Kumar et al. (2020): Conservative Q-Learning
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwoTimescaleConfig:
    """Configuration for two-timescale judge training.

    The key insight: Policy updates are fast (every step), while judge
    updates are slow (every N episodes) to prevent co-adaptation.

    Args:
        judge_update_frequency: Episodes between judge updates
        judge_warmup_episodes: Episodes before first judge update
        judge_freeze_after: Episodes after which judge is frozen
        judge_learning_rate: Learning rate for judge updates
        judge_batch_size: Batch size for judge training
        judge_epochs_per_update: Training epochs per update
        min_episodes_for_update: Minimum episodes in buffer before training
        use_exponential_moving_average: Use EMA for judge parameters
        ema_decay: EMA decay rate (if using EMA)
    """

    judge_update_frequency: int = 100  # Update judge every 100 episodes
    judge_warmup_episodes: int = 200  # Don't train judge before this
    judge_freeze_after: int = 5000  # Freeze judge after this many episodes
    judge_learning_rate: float = 1e-5  # Conservative LR for stability
    judge_batch_size: int = 32
    judge_epochs_per_update: int = 3
    min_episodes_for_update: int = 50
    use_exponential_moving_average: bool = True
    ema_decay: float = 0.995


class TwoTimescaleJudgeCallback:
    """
    Callback for two-timescale training of judge and policy.

    The policy (detector) updates every step using the current judge.
    The judge updates less frequently using collected episode data.

    This prevents the instability where:
    - Judge overfits to current policy distribution
    - Policy exploits judge weaknesses (reward hacking)

    Usage:
        callback = TwoTimescaleJudgeCallback(
            judge_model=judge,
            config=TwoTimescaleConfig(),
        )

        # In training loop:
        for episode in range(num_episodes):
            # Run episode
            episode_data = run_episode(env, agent)

            # Call callback
            callback.on_episode_end(
                episode=episode,
                episode_data=episode_data,
                true_labels=true_labels,
            )
    """

    def __init__(
        self,
        judge_model: Any,  # JudgeModel
        config: Optional[TwoTimescaleConfig] = None,
        episode_buffer_capacity: int = 1000,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            judge_model: JudgeModel instance
            config: Two-timescale configuration
            episode_buffer_capacity: Max episodes to store for judge training
            checkpoint_dir: Directory for judge checkpoints
        """
        self.judge_model = judge_model
        self.config = config or TwoTimescaleConfig()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Episode buffer for judge training
        self.episode_buffer: List[Dict[str, Any]] = []
        self.episode_buffer_capacity = episode_buffer_capacity

        # Training state
        self.total_episodes: int = 0
        self.judge_updates: int = 0
        self.judge_frozen: bool = False

        # EMA model (if enabled)
        self._ema_state_dict: Optional[Dict[str, Any]] = None

        # Metrics
        self.judge_losses: List[float] = []
        self.update_timestamps: List[float] = []

        logger.info(
            f"TwoTimescaleJudgeCallback initialized: "
            f"update_freq={self.config.judge_update_frequency}, "
            f"warmup={self.config.judge_warmup_episodes}, "
            f"freeze_after={self.config.judge_freeze_after}"
        )

    def on_episode_end(
        self,
        episode: int,
        episode_data: Dict[str, Any],
        true_labels: Optional[Dict[str, bool]] = None,
        policy_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Called at the end of each episode.

        Args:
            episode: Episode number
            episode_data: Episode trajectory data
            true_labels: Ground truth labels (for training only)
            policy_metrics: Current policy performance metrics

        Returns:
            Callback metrics
        """
        self.total_episodes = episode
        metrics: Dict[str, Any] = {"episode": episode}

        # Store episode in buffer
        self._add_to_buffer(episode_data, true_labels)

        # Check if judge should be updated
        should_update = self._should_update_judge(episode)

        if should_update:
            update_metrics = self._update_judge()
            metrics["judge_update"] = update_metrics
            self.judge_updates += 1

            # Save checkpoint
            if self.checkpoint_dir:
                self._save_checkpoint(episode)

        # Check if judge should be frozen
        if not self.judge_frozen and episode >= self.config.judge_freeze_after:
            self._freeze_judge()
            metrics["judge_frozen"] = True

        metrics["judge_updates_total"] = self.judge_updates
        metrics["buffer_size"] = len(self.episode_buffer)
        metrics["judge_frozen"] = self.judge_frozen

        return metrics

    def _add_to_buffer(
        self,
        episode_data: Dict[str, Any],
        true_labels: Optional[Dict[str, bool]],
    ) -> None:
        """Add episode to training buffer."""
        entry = {
            "episode_data": episode_data,
            "true_labels": true_labels,
            "timestamp": time.time(),
        }

        self.episode_buffer.append(entry)

        # Maintain capacity
        if len(self.episode_buffer) > self.episode_buffer_capacity:
            # Remove oldest entries
            self.episode_buffer = self.episode_buffer[-self.episode_buffer_capacity :]

    def _should_update_judge(self, episode: int) -> bool:
        """Determine if judge should be updated this episode."""
        if self.judge_frozen:
            return False

        if episode < self.config.judge_warmup_episodes:
            return False

        if len(self.episode_buffer) < self.config.min_episodes_for_update:
            return False

        if (episode - self.config.judge_warmup_episodes) % self.config.judge_update_frequency != 0:
            return False

        return True

    def _update_judge(self) -> Dict[str, Any]:
        """Update judge model using buffered episodes."""
        import torch
        import torch.nn.functional as F

        logger.info(
            f"Updating judge (update #{self.judge_updates + 1}, "
            f"buffer_size={len(self.episode_buffer)})"
        )

        start_time = time.time()

        # Sample batch from buffer
        batch_size = min(self.config.judge_batch_size, len(self.episode_buffer))
        indices = np.random.choice(
            len(self.episode_buffer), size=batch_size, replace=False
        )
        batch = [self.episode_buffer[i] for i in indices]

        # Prepare training data
        episodes = [entry["episode_data"] for entry in batch]
        labels = [entry["true_labels"] for entry in batch]

        # Compute target rewards (based on hard metrics)
        target_rewards = []
        for ep_data, true_labs in zip(episodes, labels):
            reward = self._compute_target_reward(ep_data, true_labs)
            target_rewards.append(reward)

        target_rewards_tensor = torch.tensor(
            target_rewards, dtype=torch.float32, device=self.judge_model.device
        ).unsqueeze(1)

        # Training loop
        self.judge_model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.judge_model.parameters()),
            lr=self.config.judge_learning_rate,
        )

        total_loss = 0.0
        for epoch in range(self.config.judge_epochs_per_update):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.judge_model(
                episodes=episodes,
                return_uncertainty=True,
                return_fraud_types=False,
            )

            predicted_rewards = outputs["reward"]

            # MSE loss
            loss = F.mse_loss(predicted_rewards, target_rewards_tensor)

            # Add uncertainty regularization (prevent overconfident wrong predictions)
            if "uncertainty" in outputs:
                uncertainty = outputs["uncertainty"]
                errors = (predicted_rewards - target_rewards_tensor).abs()
                uncertainty_loss = F.mse_loss(uncertainty, errors.detach())
                loss = loss + 0.1 * uncertainty_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.judge_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / self.config.judge_epochs_per_update
        self.judge_losses.append(avg_loss)

        # Update EMA if enabled
        if self.config.use_exponential_moving_average:
            self._update_ema()

        elapsed = time.time() - start_time
        self.update_timestamps.append(elapsed)

        metrics = {
            "loss": avg_loss,
            "elapsed_time": elapsed,
            "batch_size": batch_size,
            "epochs": self.config.judge_epochs_per_update,
        }

        logger.info(f"Judge update complete: loss={avg_loss:.4f}, time={elapsed:.2f}s")

        return metrics

    def _compute_target_reward(
        self,
        episode_data: Dict[str, Any],
        true_labels: Optional[Dict[str, bool]],
    ) -> float:
        """Compute target reward from hard metrics."""
        flagged_nodes = episode_data.get("flagged_nodes", [])
        if isinstance(flagged_nodes, set):
            flagged_nodes = list(flagged_nodes)

        if not flagged_nodes or true_labels is None:
            return 0.0

        # Compute precision
        true_positives = sum(
            1 for node in flagged_nodes if true_labels.get(str(node), False)
        )
        precision = true_positives / len(flagged_nodes) if flagged_nodes else 0.0

        # Compute recall over visited scope
        visited_nodes = episode_data.get("visited_nodes", [])
        if isinstance(visited_nodes, set):
            visited_nodes = list(visited_nodes)

        total_positives = sum(
            1 for node in visited_nodes if true_labels.get(str(node), False)
        )
        recall = true_positives / total_positives if total_positives > 0 else 0.0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Target reward in [-1, 1]
        return 2 * f1 - 1  # Map [0, 1] to [-1, 1]

    def _update_ema(self) -> None:
        """Update exponential moving average of judge parameters."""
        if self._ema_state_dict is None:
            self._ema_state_dict = {
                k: v.clone().detach()
                for k, v in self.judge_model.state_dict().items()
            }
        else:
            current_state = self.judge_model.state_dict()
            for k in self._ema_state_dict:
                if k in current_state:
                    self._ema_state_dict[k] = (
                        self.config.ema_decay * self._ema_state_dict[k]
                        + (1 - self.config.ema_decay) * current_state[k]
                    )

    def _freeze_judge(self) -> None:
        """Freeze judge model (no more updates)."""
        self.judge_frozen = True

        # Optionally load EMA weights
        if self.config.use_exponential_moving_average and self._ema_state_dict:
            self.judge_model.load_state_dict(self._ema_state_dict)
            logger.info("Loaded EMA weights into judge")

        # Freeze all parameters
        for param in self.judge_model.parameters():
            param.requires_grad = False

        logger.info(
            f"Judge frozen after {self.total_episodes} episodes, "
            f"{self.judge_updates} updates"
        )

    def _save_checkpoint(self, episode: int) -> None:
        """Save judge checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"judge_episode_{episode}.pt"
        self.judge_model.save(path)
        logger.info(f"Saved judge checkpoint: {path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get callback statistics."""
        stats = {
            "total_episodes": self.total_episodes,
            "judge_updates": self.judge_updates,
            "judge_frozen": self.judge_frozen,
            "buffer_size": len(self.episode_buffer),
        }

        if self.judge_losses:
            stats["avg_judge_loss"] = np.mean(self.judge_losses[-10:])
            stats["judge_loss_trend"] = (
                np.mean(self.judge_losses[-5:]) - np.mean(self.judge_losses[-10:-5])
                if len(self.judge_losses) >= 10
                else 0.0
            )

        return stats


@dataclass
class SafetyConfig:
    """Configuration for safety monitoring.

    Args:
        hard_metric_floor: Minimum required hard metric (F1)
        drift_correlation_threshold: Threshold for drift alarm
        calibration_check_frequency: Episodes between calibration checks
        max_consecutive_alarms: Alarms before intervention
        early_stopping_enabled: Enable early stopping on safety violations
        log_frequency: Episodes between log outputs
    """

    hard_metric_floor: float = 0.1
    drift_correlation_threshold: float = -0.3
    calibration_check_frequency: int = 50
    max_consecutive_alarms: int = 3
    early_stopping_enabled: bool = True
    log_frequency: int = 50


class SafetyMonitorCallback:
    """
    Monitor for safety violations during training.

    Detects:
    1. Hard metric floor violations (F1 too low)
    2. Reward drift (judge diverging from true metrics)
    3. Calibration degradation
    4. Reward hacking patterns

    Actions:
    - Log warnings
    - Trigger alarms
    - Optionally stop training
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        reward_computer: Optional[Any] = None,  # ConservativeRewardComputer
    ):
        """
        Args:
            config: Safety configuration
            reward_computer: ConservativeRewardComputer for drift detection
        """
        self.config = config or SafetyConfig()
        self.reward_computer = reward_computer

        # Monitoring state
        self.hard_metrics: List[float] = []
        self.judge_scores: List[float] = []
        self.combined_rewards: List[float] = []

        # Alarm state
        self.alarm_count: int = 0
        self.consecutive_alarms: int = 0
        self.should_stop: bool = False
        self.alarms: List[Dict[str, Any]] = []

        logger.info(
            f"SafetyMonitorCallback initialized: "
            f"floor={self.config.hard_metric_floor}, "
            f"drift_threshold={self.config.drift_correlation_threshold}"
        )

    def on_episode_end(
        self,
        episode: int,
        hard_metric: float,
        judge_score: float,
        combined_reward: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Called at end of each episode to check safety.

        Args:
            episode: Episode number
            hard_metric: Hard metric value (e.g., F1)
            judge_score: Judge's reward prediction
            combined_reward: Combined reward used for training
            additional_metrics: Any additional metrics

        Returns:
            Safety check results
        """
        # Store metrics
        self.hard_metrics.append(hard_metric)
        self.judge_scores.append(judge_score)
        self.combined_rewards.append(combined_reward)

        results: Dict[str, Any] = {
            "episode": episode,
            "alarms": [],
            "should_stop": False,
        }

        # Check hard metric floor
        floor_alarm = self._check_hard_metric_floor(episode, hard_metric)
        if floor_alarm:
            results["alarms"].append(floor_alarm)

        # Check drift (periodically)
        if episode % self.config.calibration_check_frequency == 0:
            drift_alarm = self._check_drift(episode)
            if drift_alarm:
                results["alarms"].append(drift_alarm)

            calibration_alarm = self._check_calibration(episode)
            if calibration_alarm:
                results["alarms"].append(calibration_alarm)

        # Handle alarms
        if results["alarms"]:
            self.consecutive_alarms += 1
            self.alarm_count += len(results["alarms"])
            self.alarms.extend(results["alarms"])

            if (
                self.config.early_stopping_enabled
                and self.consecutive_alarms >= self.config.max_consecutive_alarms
            ):
                self.should_stop = True
                results["should_stop"] = True
                logger.error(
                    f"SAFETY STOP: {self.consecutive_alarms} consecutive alarms"
                )
        else:
            self.consecutive_alarms = 0

        # Log periodically
        if episode % self.config.log_frequency == 0:
            self._log_status(episode)

        results["total_alarms"] = self.alarm_count
        results["consecutive_alarms"] = self.consecutive_alarms

        return results

    def _check_hard_metric_floor(
        self, episode: int, hard_metric: float
    ) -> Optional[Dict[str, Any]]:
        """Check if hard metric is above floor."""
        if hard_metric < self.config.hard_metric_floor:
            alarm = {
                "type": "hard_metric_floor_violation",
                "episode": episode,
                "value": hard_metric,
                "threshold": self.config.hard_metric_floor,
                "message": f"Hard metric {hard_metric:.3f} below floor {self.config.hard_metric_floor}",
            }
            logger.warning(f"ALARM: {alarm['message']}")
            return alarm
        return None

    def _check_drift(self, episode: int) -> Optional[Dict[str, Any]]:
        """Check for reward drift (judge diverging from hard metrics)."""
        if len(self.hard_metrics) < 20:
            return None

        # Get recent window
        window = min(50, len(self.hard_metrics))
        hard_recent = np.array(self.hard_metrics[-window:])
        judge_recent = np.array(self.judge_scores[-window:])

        # Compute correlation of changes
        hard_diff = np.diff(hard_recent)
        judge_diff = np.diff(judge_recent)

        if np.std(hard_diff) < 1e-6 or np.std(judge_diff) < 1e-6:
            return None

        correlation = np.corrcoef(hard_diff, judge_diff)[0, 1]

        if correlation < self.config.drift_correlation_threshold:
            alarm = {
                "type": "reward_drift",
                "episode": episode,
                "correlation": correlation,
                "threshold": self.config.drift_correlation_threshold,
                "message": (
                    f"Reward drift detected: correlation={correlation:.3f} "
                    f"(threshold={self.config.drift_correlation_threshold})"
                ),
            }
            logger.warning(f"ALARM: {alarm['message']}")
            return alarm
        return None

    def _check_calibration(self, episode: int) -> Optional[Dict[str, Any]]:
        """Check judge calibration (prediction error trend)."""
        if len(self.hard_metrics) < 20:
            return None

        window = min(50, len(self.hard_metrics))
        hard_recent = np.array(self.hard_metrics[-window:])
        judge_recent = np.array(self.judge_scores[-window:])

        # Compute prediction errors
        errors = np.abs(judge_recent - hard_recent)

        # Check if error is increasing
        if len(errors) >= 20:
            first_half = np.mean(errors[: len(errors) // 2])
            second_half = np.mean(errors[len(errors) // 2 :])

            if second_half > first_half * 1.5:  # 50% increase
                alarm = {
                    "type": "calibration_degradation",
                    "episode": episode,
                    "error_increase": second_half / first_half,
                    "message": (
                        f"Calibration degradation: error increased by "
                        f"{(second_half / first_half - 1) * 100:.1f}%"
                    ),
                }
                logger.warning(f"ALARM: {alarm['message']}")
                return alarm

        return None

    def _log_status(self, episode: int) -> None:
        """Log current safety status."""
        if not self.hard_metrics:
            return

        recent_hard = np.mean(self.hard_metrics[-10:])
        recent_judge = np.mean(self.judge_scores[-10:]) if self.judge_scores else 0
        recent_combined = np.mean(self.combined_rewards[-10:]) if self.combined_rewards else 0

        logger.info(
            f"Safety status (ep={episode}): "
            f"hard_metric={recent_hard:.3f}, "
            f"judge_score={recent_judge:.3f}, "
            f"combined={recent_combined:.3f}, "
            f"alarms={self.alarm_count}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety monitoring statistics."""
        stats = {
            "total_alarms": self.alarm_count,
            "consecutive_alarms": self.consecutive_alarms,
            "should_stop": self.should_stop,
            "alarm_history": self.alarms[-10:],  # Last 10 alarms
        }

        if self.hard_metrics:
            stats["hard_metric_mean"] = np.mean(self.hard_metrics)
            stats["hard_metric_min"] = np.min(self.hard_metrics)
            stats["hard_metric_recent"] = np.mean(self.hard_metrics[-10:])

        return stats


class JudgeMetricsCallback:
    """
    Callback for logging judge-specific metrics.

    Tracks:
    - Reward distribution statistics
    - Fraud type predictions
    - Uncertainty calibration
    - Latency measurements
    """

    def __init__(
        self,
        log_frequency: int = 10,
        detailed_logging: bool = False,
    ):
        """
        Args:
            log_frequency: Episodes between detailed logging
            detailed_logging: Enable detailed per-episode logging
        """
        self.log_frequency = log_frequency
        self.detailed_logging = detailed_logging

        # Metrics storage
        self.rewards: List[float] = []
        self.uncertainties: List[float] = []
        self.fraud_types: List[str] = []
        self.latencies: List[float] = []

        # Per-fraud-type tracking
        self.fraud_type_counts: Dict[str, int] = {}
        self.fraud_type_rewards: Dict[str, List[float]] = {}

    def on_judge_evaluation(
        self,
        episode: int,
        reward: float,
        metadata: Dict[str, Any],
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Called after judge evaluates an episode.

        Args:
            episode: Episode number
            reward: Judge's reward output
            metadata: Judge metadata (fraud_type, uncertainty, etc.)
            latency_ms: Evaluation latency in milliseconds
        """
        self.rewards.append(reward)
        self.uncertainties.append(metadata.get("uncertainty", 0.0))

        if latency_ms is not None:
            self.latencies.append(latency_ms)

        # Track fraud types
        fraud_type = metadata.get("fraud_type", "unknown")
        self.fraud_types.append(fraud_type)
        self.fraud_type_counts[fraud_type] = self.fraud_type_counts.get(fraud_type, 0) + 1

        if fraud_type not in self.fraud_type_rewards:
            self.fraud_type_rewards[fraud_type] = []
        self.fraud_type_rewards[fraud_type].append(reward)

        # Log periodically
        if self.detailed_logging or episode % self.log_frequency == 0:
            self._log_metrics(episode)

    def _log_metrics(self, episode: int) -> None:
        """Log current metrics."""
        if not self.rewards:
            return

        recent_rewards = self.rewards[-10:]
        recent_uncertainties = self.uncertainties[-10:]

        log_msg = (
            f"Judge metrics (ep={episode}): "
            f"reward={np.mean(recent_rewards):.3f}±{np.std(recent_rewards):.3f}, "
            f"uncertainty={np.mean(recent_uncertainties):.3f}"
        )

        if self.latencies:
            recent_latencies = self.latencies[-10:]
            log_msg += f", latency={np.mean(recent_latencies):.1f}ms"

        logger.info(log_msg)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive judge metrics."""
        stats: Dict[str, Any] = {}

        if self.rewards:
            stats["reward_mean"] = np.mean(self.rewards)
            stats["reward_std"] = np.std(self.rewards)
            stats["reward_min"] = np.min(self.rewards)
            stats["reward_max"] = np.max(self.rewards)

        if self.uncertainties:
            stats["uncertainty_mean"] = np.mean(self.uncertainties)
            stats["uncertainty_std"] = np.std(self.uncertainties)

        if self.latencies:
            stats["latency_mean_ms"] = np.mean(self.latencies)
            stats["latency_p99_ms"] = np.percentile(self.latencies, 99)

        stats["fraud_type_distribution"] = self.fraud_type_counts

        # Per-fraud-type reward statistics
        stats["fraud_type_reward_means"] = {
            ft: np.mean(rewards) for ft, rewards in self.fraud_type_rewards.items()
        }

        return stats

    def get_report(self) -> str:
        """Generate human-readable report."""
        stats = self.get_statistics()

        lines = [
            "=" * 60,
            "JUDGE METRICS REPORT",
            "=" * 60,
            "",
            "Reward Statistics:",
            f"  Mean:  {stats.get('reward_mean', 0):.4f}",
            f"  Std:   {stats.get('reward_std', 0):.4f}",
            f"  Range: [{stats.get('reward_min', 0):.4f}, {stats.get('reward_max', 0):.4f}]",
            "",
            "Uncertainty Statistics:",
            f"  Mean: {stats.get('uncertainty_mean', 0):.4f}",
            f"  Std:  {stats.get('uncertainty_std', 0):.4f}",
            "",
        ]

        if self.latencies:
            lines.extend([
                "Latency Statistics:",
                f"  Mean: {stats.get('latency_mean_ms', 0):.1f} ms",
                f"  P99:  {stats.get('latency_p99_ms', 0):.1f} ms",
                "",
            ])

        lines.extend([
            "Fraud Type Distribution:",
        ])
        for ft, count in sorted(
            stats.get("fraud_type_distribution", {}).items(),
            key=lambda x: -x[1],
        ):
            mean_reward = stats.get("fraud_type_reward_means", {}).get(ft, 0)
            lines.append(f"  {ft}: {count} ({mean_reward:.3f} avg reward)")

        lines.append("=" * 60)

        return "\n".join(lines)
