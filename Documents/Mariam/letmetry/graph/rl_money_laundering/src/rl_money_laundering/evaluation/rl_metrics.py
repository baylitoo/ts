"""
RL-Specific Metrics for AML Investigation Agent

Implements reinforcement learning evaluation metrics:
1. Episode Performance: Return, length, success rate
2. Investigation Efficiency: Steps per detection, coverage
3. Learning Dynamics: Convergence, stability, sample efficiency
4. Budget Utilization: Cost-effectiveness under constraints
5. Multi-Agent Metrics: Coordination, specialization

These metrics assess the RL agent's effectiveness at
learning optimal investigation strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
from collections import deque
import warnings


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    # Basic episode stats
    episode_return: float = 0.0
    episode_length: int = 0
    discounted_return: float = 0.0

    # Investigation outcomes
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    # Efficiency
    steps_per_detection: float = float('inf')
    investigation_coverage: float = 0.0  # Fraction of graph explored
    budget_utilization: float = 0.0  # Fraction of budget used

    # Actions taken
    action_distribution: Dict[int, int] = field(default_factory=dict)

    def precision(self) -> float:
        """Precision of investigations."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    def recall(self) -> float:
        """Recall (detection rate)."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    def f1(self) -> float:
        """F1 score."""
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class RLPerformanceMetrics:
    """Aggregated RL performance metrics across episodes."""

    # Episode statistics
    mean_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0
    median_return: float = 0.0

    mean_length: float = 0.0
    std_length: float = 0.0

    # Success metrics
    success_rate: float = 0.0  # Episodes with positive outcome
    completion_rate: float = 0.0  # Episodes not truncated

    # Investigation metrics
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_steps_per_detection: float = 0.0

    # Efficiency
    mean_coverage: float = 0.0
    mean_budget_utilization: float = 0.0

    # Learning progress
    return_improvement: float = 0.0  # Recent vs early returns
    convergence_score: float = 0.0  # Stability measure

    # Sample efficiency
    episodes_to_threshold: int = -1  # Episodes to reach target return

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "median_return": self.median_return,
            "mean_length": self.mean_length,
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_steps_per_detection": self.mean_steps_per_detection,
            "mean_coverage": self.mean_coverage,
            "mean_budget_utilization": self.mean_budget_utilization,
            "return_improvement": self.return_improvement,
            "convergence_score": self.convergence_score,
            "episodes_to_threshold": self.episodes_to_threshold,
        }


class EpisodeTracker:
    """
    Tracks episode-level metrics during training/evaluation.

    Usage:
        tracker = EpisodeTracker()
        tracker.start_episode()
        for step in episode:
            tracker.record_step(reward, action, info)
        episode_metrics = tracker.end_episode(info)
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        """Reset tracker state."""
        self._current_return = 0.0
        self._current_discounted = 0.0
        self._step_count = 0
        self._discount_factor = 1.0
        self._action_counts: Dict[int, int] = {}
        self._in_episode = False

        # Investigation tracking
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self._nodes_visited: set = set()

    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self.reset()
        self._in_episode = True

    def record_step(
        self,
        reward: float,
        action: int,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single step.

        Args:
            reward: Step reward
            action: Action taken
            info: Step info dict with optional keys:
                - is_true_positive: bool
                - is_false_positive: bool
                - is_false_negative: bool
                - node_id: Current node being investigated
        """
        if not self._in_episode:
            warnings.warn("Recording step without starting episode")
            self.start_episode()

        self._current_return += reward
        self._current_discounted += self._discount_factor * reward
        self._discount_factor *= self.gamma
        self._step_count += 1

        # Track action distribution
        self._action_counts[action] = self._action_counts.get(action, 0) + 1

        # Track investigation outcomes from info
        if info:
            if info.get("is_true_positive", False):
                self._tp += 1
            if info.get("is_false_positive", False):
                self._fp += 1
            if info.get("is_false_negative", False):
                self._fn += 1
            if info.get("is_true_negative", False):
                self._tn += 1
            if "node_id" in info:
                self._nodes_visited.add(info["node_id"])

    def end_episode(
        self,
        final_info: Optional[Dict[str, Any]] = None
    ) -> EpisodeMetrics:
        """
        End episode and compute metrics.

        Args:
            final_info: Final step info with optional keys:
                - total_nodes: Total nodes in graph
                - budget_used: Fraction of budget used
                - success: Whether episode was successful
        """
        self._in_episode = False

        # Compute steps per detection
        detections = self._tp
        steps_per_det = (
            self._step_count / detections if detections > 0
            else float('inf')
        )

        # Coverage
        total_nodes = 1
        budget_util = 0.0
        if final_info:
            total_nodes = final_info.get("total_nodes", len(self._nodes_visited) or 1)
            budget_util = final_info.get("budget_used", 0.0)

        coverage = len(self._nodes_visited) / total_nodes

        return EpisodeMetrics(
            episode_return=self._current_return,
            episode_length=self._step_count,
            discounted_return=self._current_discounted,
            true_positives=self._tp,
            false_positives=self._fp,
            false_negatives=self._fn,
            true_negatives=self._tn,
            steps_per_detection=steps_per_det,
            investigation_coverage=coverage,
            budget_utilization=budget_util,
            action_distribution=self._action_counts.copy(),
        )


class RLMetricsAggregator:
    """
    Aggregates episode metrics and computes training statistics.

    Maintains rolling windows for computing learning progress
    and convergence metrics.
    """

    def __init__(
        self,
        window_size: int = 100,
        success_threshold: float = 0.0
    ):
        """
        Args:
            window_size: Window for rolling statistics
            success_threshold: Minimum return for success
        """
        self.window_size = window_size
        self.success_threshold = success_threshold

        # Episode history
        self._episodes: List[EpisodeMetrics] = []
        self._returns: deque = deque(maxlen=window_size)
        self._lengths: deque = deque(maxlen=window_size)

        # Early episodes for improvement calculation
        self._early_returns: List[float] = []
        self._early_window = 20

    def add_episode(self, metrics: EpisodeMetrics) -> None:
        """Add an episode's metrics."""
        self._episodes.append(metrics)
        self._returns.append(metrics.episode_return)
        self._lengths.append(metrics.episode_length)

        # Track early returns
        if len(self._early_returns) < self._early_window:
            self._early_returns.append(metrics.episode_return)

    def compute_aggregated(self) -> RLPerformanceMetrics:
        """Compute aggregated metrics across all episodes."""
        if not self._episodes:
            return RLPerformanceMetrics()

        returns = [e.episode_return for e in self._episodes]
        lengths = [e.episode_length for e in self._episodes]

        # Basic statistics
        metrics = RLPerformanceMetrics(
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            median_return=float(np.median(returns)),
            mean_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
        )

        # Success and completion rates
        successes = sum(
            1 for e in self._episodes
            if e.episode_return >= self.success_threshold
        )
        metrics.success_rate = successes / len(self._episodes)

        # Investigation metrics
        precisions = [e.precision() for e in self._episodes]
        recalls = [e.recall() for e in self._episodes]
        f1s = [e.f1() for e in self._episodes]
        spd = [e.steps_per_detection for e in self._episodes if e.steps_per_detection < float('inf')]

        metrics.mean_precision = float(np.mean(precisions)) if precisions else 0.0
        metrics.mean_recall = float(np.mean(recalls)) if recalls else 0.0
        metrics.mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        metrics.mean_steps_per_detection = float(np.mean(spd)) if spd else float('inf')

        # Efficiency metrics
        coverages = [e.investigation_coverage for e in self._episodes]
        budgets = [e.budget_utilization for e in self._episodes]
        metrics.mean_coverage = float(np.mean(coverages))
        metrics.mean_budget_utilization = float(np.mean(budgets))

        # Learning progress
        if len(self._episodes) >= self._early_window:
            early_mean = np.mean(self._early_returns)
            recent_mean = np.mean(list(self._returns)[-self._early_window:])
            metrics.return_improvement = float(recent_mean - early_mean)

        # Convergence score (inverse of recent variance)
        if len(self._returns) >= 10:
            recent_std = np.std(list(self._returns)[-50:])
            metrics.convergence_score = 1.0 / (1.0 + recent_std)

        return metrics

    def compute_rolling(self) -> RLPerformanceMetrics:
        """Compute metrics over recent window only."""
        if not self._returns:
            return RLPerformanceMetrics()

        returns = list(self._returns)
        lengths = list(self._lengths)

        return RLPerformanceMetrics(
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            median_return=float(np.median(returns)),
            mean_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
            success_rate=sum(1 for r in returns if r >= self.success_threshold) / len(returns),
        )

    def get_learning_curve(
        self,
        smoothing: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Get smoothed learning curve.

        Returns:
            episodes: Episode numbers
            returns: Smoothed returns
        """
        returns = [e.episode_return for e in self._episodes]

        if len(returns) < smoothing:
            return list(range(len(returns))), returns

        # Moving average smoothing
        smoothed = []
        for i in range(len(returns)):
            start = max(0, i - smoothing + 1)
            smoothed.append(np.mean(returns[start:i+1]))

        return list(range(len(returns))), smoothed

    def episodes_to_target(self, target_return: float) -> int:
        """
        Find first episode where rolling average exceeds target.

        Returns:
            Episode number, or -1 if never reached
        """
        _, smoothed = self.get_learning_curve()

        for i, ret in enumerate(smoothed):
            if ret >= target_return:
                return i

        return -1


class InvestigationEfficiency:
    """
    Metrics for investigation efficiency under budget constraints.

    Measures how effectively the agent uses its investigation
    budget to detect money laundering.
    """

    def __init__(self, total_budget: int):
        self.total_budget = total_budget

    def compute_efficiency(
        self,
        detections: int,
        budget_used: int,
        total_illicit: int
    ) -> Dict[str, float]:
        """
        Compute investigation efficiency metrics.

        Args:
            detections: Number of true positives found
            budget_used: Budget consumed
            total_illicit: Total illicit nodes in graph

        Returns:
            Dict with efficiency metrics
        """
        # Detection rate
        detection_rate = detections / total_illicit if total_illicit > 0 else 0.0

        # Cost per detection
        cost_per_detection = budget_used / detections if detections > 0 else float('inf')

        # Budget efficiency (detections per unit budget)
        budget_efficiency = detections / budget_used if budget_used > 0 else 0.0

        # Normalized efficiency (compared to random baseline)
        expected_random = (budget_used / self.total_budget) * total_illicit
        normalized_efficiency = (
            detections / expected_random if expected_random > 0 else 0.0
        )

        # Remaining budget
        budget_remaining = (self.total_budget - budget_used) / self.total_budget

        return {
            "detection_rate": detection_rate,
            "cost_per_detection": cost_per_detection,
            "budget_efficiency": budget_efficiency,
            "normalized_efficiency": normalized_efficiency,
            "budget_remaining": budget_remaining,
            "budget_utilization": budget_used / self.total_budget,
        }

    def compute_recall_at_budget(
        self,
        detections_at_steps: List[int],
        total_illicit: int
    ) -> List[float]:
        """
        Compute recall at each budget step.

        Args:
            detections_at_steps: Cumulative detections at each step
            total_illicit: Total illicit nodes

        Returns:
            Recall at each step
        """
        if total_illicit == 0:
            return [0.0] * len(detections_at_steps)

        return [d / total_illicit for d in detections_at_steps]


@dataclass
class MultiAgentRLMetrics:
    """Metrics for multi-agent RL systems."""

    # Per-agent returns
    agent_returns: Dict[str, float] = field(default_factory=dict)

    # Coordination metrics
    joint_success_rate: float = 0.0  # All agents succeed together
    coordination_efficiency: float = 0.0  # Overlap avoidance

    # Specialization
    agent_specialization: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # Agent -> metric scores

    # Communication
    message_count: int = 0
    message_utility: float = 0.0  # Improvement from messages

    # Fairness
    return_gini: float = 0.0  # Gini coefficient of returns
    workload_balance: float = 0.0  # How evenly work is distributed


class MultiAgentMetricsComputer:
    """Computes metrics for multi-agent RL systems."""

    def compute_coordination_efficiency(
        self,
        agent_visits: Dict[str, set],
        total_nodes: int
    ) -> float:
        """
        Compute coordination efficiency (overlap avoidance).

        Low overlap = high efficiency (agents not duplicating work).
        """
        all_visits = set()
        total_visits = 0

        for visits in agent_visits.values():
            total_visits += len(visits)
            all_visits.update(visits)

        if total_visits == 0:
            return 1.0

        # Efficiency = unique visits / total visits
        # 1.0 means no overlap
        return len(all_visits) / total_visits

    def compute_gini(self, values: Sequence[float]) -> float:
        """Compute Gini coefficient for fairness."""
        values = np.array(sorted(values))
        n = len(values)

        if n == 0 or values.sum() == 0:
            return 0.0

        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))

    def compute_workload_balance(
        self,
        agent_steps: Dict[str, int]
    ) -> float:
        """
        Compute workload balance across agents.

        Returns:
            1.0 = perfectly balanced, 0.0 = one agent does everything
        """
        steps = list(agent_steps.values())
        if not steps or sum(steps) == 0:
            return 1.0

        # Use coefficient of variation
        cv = np.std(steps) / np.mean(steps) if np.mean(steps) > 0 else 0.0

        # Convert to 0-1 scale (lower CV = higher balance)
        return float(1.0 / (1.0 + cv))

    def compute_all(
        self,
        agent_returns: Dict[str, float],
        agent_visits: Dict[str, set],
        agent_steps: Dict[str, int],
        total_nodes: int
    ) -> MultiAgentRLMetrics:
        """Compute all multi-agent metrics."""
        return MultiAgentRLMetrics(
            agent_returns=agent_returns,
            coordination_efficiency=self.compute_coordination_efficiency(
                agent_visits, total_nodes
            ),
            return_gini=self.compute_gini(list(agent_returns.values())),
            workload_balance=self.compute_workload_balance(agent_steps),
        )


def compute_rl_metrics(
    episodes: List[EpisodeMetrics],
    window_size: int = 100,
    success_threshold: float = 0.0
) -> RLPerformanceMetrics:
    """
    Convenience function to compute RL metrics from episode list.

    Args:
        episodes: List of episode metrics
        window_size: Window for rolling stats
        success_threshold: Min return for success

    Returns:
        Aggregated RL performance metrics
    """
    aggregator = RLMetricsAggregator(
        window_size=window_size,
        success_threshold=success_threshold
    )

    for ep in episodes:
        aggregator.add_episode(ep)

    return aggregator.compute_aggregated()
