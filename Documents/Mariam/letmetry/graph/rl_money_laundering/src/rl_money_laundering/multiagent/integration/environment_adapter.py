"""
Environment adapter for multiagent integration.

Wraps the existing AMLDetectionEnv to collect episode data for judge evaluation
without modifying the original environment behavior.
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EnvironmentProtocol(Protocol):
    """Protocol defining expected environment interface."""

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]: ...
    def step(self, action: int, **kwargs: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]: ...
    @property
    def graph(self) -> Any: ...
    @property
    def current_node(self) -> Any: ...
    @property
    def visited_nodes(self) -> set: ...
    @property
    def flagged_nodes(self) -> set: ...


@dataclass
class StepRecord:
    """Record of a single environment step.

    Captures all information needed for judge evaluation.
    """
    step_number: int
    node_id: Any
    action: int
    action_type: str
    reward: float
    confidence: float
    node_features: Dict[str, Any]
    neighbors: List[Any]
    is_fraud: bool
    is_flagged: bool
    cumulative_reward: float


@dataclass
class EpisodeData:
    """Complete episode data for judge evaluation.

    Contains all information about an episode in a format suitable
    for serialization and judge model input.
    """
    episode_id: str
    steps: List[StepRecord] = field(default_factory=list)
    start_node: Any = None
    end_node: Any = None
    total_reward: float = 0.0
    episode_length: int = 0

    graph_stats: Dict[str, Any] = field(default_factory=dict)
    temporal_window: Tuple[Optional[float], Optional[float]] = (None, None)

    flagged_nodes: List[Any] = field(default_factory=list)
    visited_nodes: List[Any] = field(default_factory=list)
    fraud_nodes_encountered: List[Any] = field(default_factory=list)

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        """Compute precision from confusion matrix."""
        total_flagged = self.true_positives + self.false_positives
        if total_flagged == 0:
            return 0.0
        return self.true_positives / total_flagged

    @property
    def recall(self) -> float:
        """Compute recall from confusion matrix."""
        total_fraud = self.true_positives + self.false_negatives
        if total_fraud == 0:
            return 0.0
        return self.true_positives / total_fraud

    @property
    def f1_score(self) -> float:
        """Compute F1 score from precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "steps": [
                {
                    "step_number": s.step_number,
                    "node_id": str(s.node_id),
                    "action": s.action,
                    "action_type": s.action_type,
                    "reward": s.reward,
                    "confidence": s.confidence,
                    "is_fraud": s.is_fraud,
                    "is_flagged": s.is_flagged,
                }
                for s in self.steps
            ],
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "graph_stats": self.graph_stats,
            "flagged_nodes": [str(n) for n in self.flagged_nodes],
            "visited_nodes": [str(n) for n in self.visited_nodes],
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1_score,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
            },
        }


class EpisodeCollector:
    """Collects episode data without modifying environment behavior.

    This class observes environment interactions and builds EpisodeData
    objects for judge evaluation. It does not alter the environment's
    state transitions or rewards.
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        track_node_features: bool = True,
        anonymize_ids: bool = True,
    ) -> None:
        """Initialize episode collector.

        Args:
            buffer_size: Maximum episodes to keep in buffer.
            track_node_features: Whether to capture node features.
            anonymize_ids: Whether to anonymize node IDs for judge.
        """
        self.buffer_size = buffer_size
        self.track_node_features = track_node_features
        self.anonymize_ids = anonymize_ids

        self._episode_buffer: List[EpisodeData] = []
        self._current_episode: Optional[EpisodeData] = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._id_map: Dict[Any, str] = {}
        self._episode_counter = 0

    def start_episode(
        self,
        env: Any,
        start_node: Any,
        graph_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Begin collecting data for a new episode.

        Args:
            env: Environment instance (for graph access).
            start_node: Starting node for the episode.
            graph_stats: Optional pre-computed graph statistics.
        """
        self._episode_counter += 1
        episode_id = self._generate_episode_id(start_node)

        self._current_episode = EpisodeData(
            episode_id=episode_id,
            start_node=self._anonymize(start_node),
            graph_stats=graph_stats or self._compute_graph_stats(env),
        )
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._id_map.clear()

    def record_step(
        self,
        node_id: Any,
        action: int,
        reward: float,
        confidence: float,
        env: Any,
        info: Dict[str, Any],
    ) -> None:
        """Record a single step in the current episode.

        Args:
            node_id: Current node ID.
            action: Action taken.
            reward: Reward received.
            confidence: Agent's confidence in action.
            env: Environment instance.
            info: Step info dictionary from environment.
        """
        if self._current_episode is None:
            logger.warning("record_step called without start_episode")
            return

        self._cumulative_reward += reward
        self._step_count += 1

        action_type = self._classify_action(action, env)
        is_fraud = self._check_fraud(node_id, env)
        is_flagged = action_type == "FLAG"

        node_features = {}
        if self.track_node_features:
            node_features = self._extract_node_features(node_id, env)

        neighbors = self._get_neighbors(node_id, env)

        step_record = StepRecord(
            step_number=self._step_count,
            node_id=self._anonymize(node_id),
            action=action,
            action_type=action_type,
            reward=reward,
            confidence=confidence,
            node_features=node_features,
            neighbors=[self._anonymize(n) for n in neighbors],
            is_fraud=is_fraud,
            is_flagged=is_flagged,
            cumulative_reward=self._cumulative_reward,
        )

        self._current_episode.steps.append(step_record)

        if is_flagged:
            self._current_episode.flagged_nodes.append(self._anonymize(node_id))
            if is_fraud:
                self._current_episode.true_positives += 1
            else:
                self._current_episode.false_positives += 1

        if is_fraud and self._anonymize(node_id) not in [
            self._anonymize(n) for n in self._current_episode.fraud_nodes_encountered
        ]:
            self._current_episode.fraud_nodes_encountered.append(
                self._anonymize(node_id)
            )

    def end_episode(
        self,
        env: Any,
        final_info: Optional[Dict[str, Any]] = None,
    ) -> EpisodeData:
        """Finalize and store the current episode.

        Args:
            env: Environment instance.
            final_info: Final step info with episode statistics.

        Returns:
            Completed EpisodeData object.
        """
        if self._current_episode is None:
            raise RuntimeError("end_episode called without start_episode")

        episode = self._current_episode

        episode.end_node = self._anonymize(getattr(env, "current_node", None))
        episode.total_reward = self._cumulative_reward
        episode.episode_length = self._step_count
        episode.visited_nodes = [
            self._anonymize(n) for n in getattr(env, "visited_nodes", set())
        ]

        self._compute_false_negatives(env, episode)

        if final_info:
            episode.metadata.update(final_info)

        self._episode_buffer.append(episode)
        if len(self._episode_buffer) > self.buffer_size:
            self._episode_buffer.pop(0)

        self._current_episode = None
        return episode

    def get_recent_episodes(self, n: int = 100) -> List[EpisodeData]:
        """Get the N most recent episodes from buffer."""
        return self._episode_buffer[-n:]

    def get_all_episodes(self) -> List[EpisodeData]:
        """Get all episodes in buffer."""
        return list(self._episode_buffer)

    def clear_buffer(self) -> None:
        """Clear the episode buffer."""
        self._episode_buffer.clear()

    def _generate_episode_id(self, start_node: Any) -> str:
        """Generate unique episode identifier."""
        content = f"{self._episode_counter}:{start_node}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _anonymize(self, node_id: Any) -> str:
        """Anonymize node ID to prevent leakage."""
        if not self.anonymize_ids:
            return str(node_id)

        if node_id is None:
            return "NONE"

        if node_id not in self._id_map:
            idx = len(self._id_map)
            self._id_map[node_id] = f"ENTITY_{chr(65 + idx % 26)}{idx // 26 or ''}"

        return self._id_map[node_id]

    def _classify_action(self, action: int, env: Any) -> str:
        """Classify action type (MOVE or FLAG)."""
        max_neighbors = getattr(env, "max_neighbors", 5)
        if action == max_neighbors:
            return "FLAG"
        return "MOVE"

    def _check_fraud(self, node_id: Any, env: Any) -> bool:
        """Check if node is fraudulent."""
        graph = getattr(env, "graph", None)
        if graph is None:
            return False

        try:
            node_data = graph.nodes.get(node_id, {})
            for label_key in ["is_fraud", "isFraud", "is_money_laundering", "label"]:
                if label_key in node_data:
                    return bool(node_data[label_key])
        except Exception:
            pass

        fraud_nodes = getattr(env, "fraud_nodes", set())
        return node_id in fraud_nodes

    def _extract_node_features(self, node_id: Any, env: Any) -> Dict[str, Any]:
        """Extract features for a node."""
        graph = getattr(env, "graph", None)
        if graph is None:
            return {}

        try:
            node_data = dict(graph.nodes.get(node_id, {}))
            safe_features = {}
            for k, v in node_data.items():
                if k not in ["is_fraud", "isFraud", "label", "is_money_laundering"]:
                    if isinstance(v, (int, float, bool, str)):
                        safe_features[k] = v
                    elif isinstance(v, np.ndarray):
                        safe_features[k] = v.tolist()
            return safe_features
        except Exception:
            return {}

    def _get_neighbors(self, node_id: Any, env: Any) -> List[Any]:
        """Get neighboring nodes."""
        graph = getattr(env, "graph", None)
        if graph is None:
            return []

        try:
            return list(graph.successors(node_id))[:10]
        except Exception:
            return []

    def _compute_graph_stats(self, env: Any) -> Dict[str, Any]:
        """Compute basic graph statistics."""
        graph = getattr(env, "graph", None)
        if graph is None:
            return {}

        try:
            return {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": graph.number_of_edges() / max(1, graph.number_of_nodes() ** 2),
            }
        except Exception:
            return {}

    def _compute_false_negatives(self, env: Any, episode: EpisodeData) -> None:
        """Compute false negatives from visited but unflagged fraud nodes."""
        fraud_encountered = set(episode.fraud_nodes_encountered)
        flagged = set(episode.flagged_nodes)

        unflagged_fraud = fraud_encountered - flagged
        episode.false_negatives = len(unflagged_fraud)


class EnvironmentAdapter:
    """Adapter that wraps existing environment for multiagent integration.

    This adapter transparently wraps an AMLDetectionEnv instance, adding
    episode collection without modifying the underlying environment's
    behavior. All original methods pass through unchanged.
    """

    def __init__(
        self,
        env: EnvironmentProtocol,
        collector: Optional[EpisodeCollector] = None,
        enable_collection: bool = True,
    ) -> None:
        """Initialize environment adapter.

        Args:
            env: Original environment instance.
            collector: Episode collector (created if not provided).
            enable_collection: Whether to collect episode data.
        """
        self._env = env
        self._collector = collector or EpisodeCollector()
        self._enable_collection = enable_collection
        self._in_episode = False
        self._last_node: Any = None

    @property
    def env(self) -> EnvironmentProtocol:
        """Access underlying environment."""
        return self._env

    @property
    def collector(self) -> EpisodeCollector:
        """Access episode collector."""
        return self._collector

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and start episode collection.

        Passes through to underlying environment, then begins collection.
        """
        obs, info = self._env.reset(**kwargs)

        if self._enable_collection:
            start_node = getattr(self._env, "current_node", None)
            self._collector.start_episode(self._env, start_node)
            self._in_episode = True
            self._last_node = start_node

        return obs, info

    def step(
        self,
        action: int,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute step and record for collection.

        Passes through to underlying environment, then records step data.

        Args:
            action: Action to take.
            confidence: Agent confidence in action (0-1).
            **kwargs: Additional arguments for environment step.

        Returns:
            Standard Gymnasium step output.
        """
        obs, reward, terminated, truncated, info = self._env.step(action, **kwargs)

        if self._enable_collection and self._in_episode:
            current_node = getattr(self._env, "current_node", self._last_node)
            self._collector.record_step(
                node_id=current_node,
                action=action,
                reward=reward,
                confidence=confidence,
                env=self._env,
                info=info,
            )
            self._last_node = current_node

            if terminated or truncated:
                self._in_episode = False

        return obs, reward, terminated, truncated, info

    def get_episode_data(self) -> Optional[EpisodeData]:
        """Finalize and return current episode data.

        Call this after episode ends to get collected data.
        """
        if not self._enable_collection:
            return None

        if self._in_episode:
            self._in_episode = False
            return self._collector.end_episode(self._env)

        if self._collector._current_episode is not None:
            return self._collector.end_episode(self._env)

        return None

    def get_recent_episodes(self, n: int = 100) -> List[EpisodeData]:
        """Get recent episodes from collector buffer."""
        return self._collector.get_recent_episodes(n)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying environment."""
        return getattr(self._env, name)

    def enable_collection(self) -> None:
        """Enable episode collection."""
        self._enable_collection = True

    def disable_collection(self) -> None:
        """Disable episode collection."""
        self._enable_collection = False
