"""
Bridge between existing StateEncoder and JudgeModel.

Provides caching, batching, and conversion utilities to efficiently
connect the GNN-based state encoder with the text-based judge model.
"""

from __future__ import annotations

import logging
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from ...protocols import EncoderBridgeProtocol, JudgeModelProtocol

logger = logging.getLogger(__name__)


class StateEncoderProtocol(Protocol):
    """Protocol for state encoder interface."""

    def forward_state(
        self,
        graph: Any,
        current_node: Any,
        visited_nodes: set,
        visited_edges: list,
        node_feature_extractor: Any,
        return_auxiliary: bool = False,
    ) -> Tuple[Any, Any, Optional[Dict[str, Any]]]: ...


@dataclass
class EmbeddingCache:
    """LRU cache for node embeddings.

    Caches GNN embeddings to avoid redundant computation when the same
    node is visited multiple times across episodes.
    """
    max_size: int = 10000
    _cache: OrderedDict = field(default_factory=OrderedDict)
    _hits: int = 0
    _misses: int = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = embedding

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Compute cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


@dataclass
class EncoderOutput:
    """Output from encoder bridge combining GNN and judge signals."""
    embedding: np.ndarray
    history_features: np.ndarray
    judge_features: Optional[np.ndarray] = None
    auxiliary_outputs: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_state(self) -> np.ndarray:
        """Concatenate all features into full state vector."""
        components = [self.embedding, self.history_features]
        if self.judge_features is not None:
            components.append(self.judge_features)
        return np.concatenate(components)


class EncoderBridge(EncoderBridgeProtocol):
    """Bridge connecting StateEncoder with JudgeModel.

    This bridge:
    1. Wraps the existing StateEncoder for GNN-based state encoding
    2. Maintains embedding cache for efficiency
    3. Provides episode-level features from judge model
    4. Supports batched encoding for training efficiency
    """

    def __init__(
        self,
        state_encoder: Optional[StateEncoderProtocol] = None,
        judge_model: Optional[JudgeModelProtocol] = None,
        cache_embeddings: bool = True,
        cache_size: int = 10000,
        include_judge_features: bool = False,
        node_feature_extractor: Optional[Any] = None,
    ) -> None:
        """Initialize encoder bridge.

        Args:
            state_encoder: Existing GNN state encoder.
            judge_model: Judge model for episode evaluation.
            cache_embeddings: Whether to cache node embeddings.
            cache_size: Maximum cache size.
            include_judge_features: Add judge-derived features to state.
            node_feature_extractor: Callable to extract features for PyG conversion.
        """
        self._state_encoder = state_encoder
        self._judge_model = judge_model
        self._include_judge_features = include_judge_features
        self._node_feature_extractor = node_feature_extractor

        self._cache: Optional[EmbeddingCache] = None
        if cache_embeddings:
            self._cache = EmbeddingCache(max_size=cache_size)

        self._episode_judge_cache: Dict[str, Dict[str, Any]] = {}
        self._current_episode_id: Optional[str] = None

    def set_state_encoder(self, encoder: StateEncoderProtocol) -> None:
        """Set or replace the state encoder."""
        self._state_encoder = encoder

    def set_judge_model(self, judge: JudgeModelProtocol) -> None:
        """Set or replace the judge model."""
        self._judge_model = judge

    def encode_state(
        self,
        graph: Any,
        node_id: Any,
        visited_nodes: set,
        visited_edges: set,
        edge_time: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> EncoderOutput:
        """Encode state using GNN encoder with optional caching.

        Args:
            graph: NetworkX graph.
            node_id: Current node ID.
            visited_nodes: Set of visited node IDs.
            visited_edges: Set of visited edge tuples.
            edge_time: Current temporal position.
            episode_id: Episode identifier for judge caching.

        Returns:
            EncoderOutput with embedding and features.
        """
        if self._state_encoder is None:
            raise RuntimeError("State encoder not set")

        cache_key = None
        if self._cache is not None:
            cache_key = self._compute_cache_key(node_id, visited_nodes, edge_time)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return EncoderOutput(
                    embedding=cached["embedding"],
                    history_features=cached["history"],
                    auxiliary_outputs=cached.get("aux", {}),
                )

        embedding, history, aux = self._state_encoder.forward_state(
            graph=graph,
            current_node=node_id,
            visited_nodes=visited_nodes,
            visited_edges=list(visited_edges),
            node_feature_extractor=self._node_feature_extractor,
            return_auxiliary=True,
        )

        embedding_np = self._to_numpy(embedding)
        history_np = self._to_numpy(history)

        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, {
                "embedding": embedding_np,
                "history": history_np,
                "aux": aux,
            })

        judge_features = None
        if self._include_judge_features and episode_id:
            judge_features = self._get_judge_features(episode_id)

        return EncoderOutput(
            embedding=embedding_np,
            history_features=history_np,
            judge_features=judge_features,
            auxiliary_outputs=aux or {},
        )

    def encode_batch(
        self,
        graph: Any,
        node_ids: List[Any],
        visited_nodes_list: List[set],
        visited_edges_list: List[set],
        edge_times: Optional[List[float]] = None,
    ) -> List[EncoderOutput]:
        """Batch encode multiple states.

        Args:
            graph: NetworkX graph.
            node_ids: List of node IDs.
            visited_nodes_list: List of visited node sets.
            visited_edges_list: List of visited edge sets.
            edge_times: Optional list of temporal positions.

        Returns:
            List of EncoderOutput objects.
        """
        if edge_times is None:
            edge_times = [None] * len(node_ids)

        results = []
        for node_id, visited, edges, time in zip(
            node_ids, visited_nodes_list, visited_edges_list, edge_times
        ):
            output = self.encode_state(
                graph=graph,
                node_id=node_id,
                visited_nodes=visited,
                visited_edges=edges,
                edge_time=time,
            )
            results.append(output)

        return results

    def evaluate_episode(
        self,
        episode_text: str,
        episode_id: str,
    ) -> Dict[str, Any]:
        """Evaluate episode using judge model.

        Args:
            episode_text: Serialized episode text.
            episode_id: Episode identifier for caching.

        Returns:
            Judge evaluation with reward and uncertainty.
        """
        if self._judge_model is None:
            return {"reward": 0.0, "uncertainty": 1.0, "available": False}

        if episode_id in self._episode_judge_cache:
            return self._episode_judge_cache[episode_id]

        episode_dict = {"text": episode_text}
        reward_value, metadata = self._judge_model.compute_reward(
            episode=episode_dict,
            conservative=True,
        )

        result = {
            "reward": reward_value,
            "uncertainty": metadata.get("uncertainty", 1.0),
            "available": True,
            **metadata,
        }

        self._episode_judge_cache[episode_id] = result

        if len(self._episode_judge_cache) > 1000:
            keys = list(self._episode_judge_cache.keys())
            for key in keys[:500]:
                del self._episode_judge_cache[key]

        return result

    def start_episode(self, episode_id: str) -> None:
        """Signal start of new episode for caching."""
        self._current_episode_id = episode_id

    def end_episode(self) -> None:
        """Signal end of episode."""
        self._current_episode_id = None

    def clear_caches(self) -> None:
        """Clear all caches."""
        if self._cache is not None:
            self._cache.clear()
        self._episode_judge_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {"judge_cache_size": len(self._episode_judge_cache)}
        if self._cache is not None:
            stats["embedding_cache"] = self._cache.get_stats()
        return stats

    def _compute_cache_key(
        self,
        node_id: Any,
        visited_nodes: set,
        edge_time: Optional[float],
    ) -> str:
        """Compute cache key for embedding."""
        visited_hash = hashlib.md5(
            str(sorted(str(n) for n in visited_nodes)).encode()
        ).hexdigest()[:8]

        time_str = f"{edge_time:.2f}" if edge_time else "none"
        return f"{node_id}:{visited_hash}:{time_str}"

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        if tensor is None:
            return np.array([])

        if isinstance(tensor, np.ndarray):
            return tensor

        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()

        if hasattr(tensor, "numpy"):
            return tensor.numpy()

        return np.array(tensor)

    def _get_judge_features(self, episode_id: str) -> Optional[np.ndarray]:
        """Get judge-derived features for episode.

        Returns features derived from running judge on recent episodes,
        providing signal about episode quality to the policy.
        """
        if episode_id not in self._episode_judge_cache:
            return None

        judge_output = self._episode_judge_cache[episode_id]
        return np.array([
            judge_output.get("reward", 0.0),
            judge_output.get("uncertainty", 1.0),
            float(judge_output.get("available", False)),
        ])


class EpisodeTextBuilder:
    """Builds text representation of episodes for judge input.

    Provides utilities to convert EpisodeData into the text format
    expected by the judge model, compatible with EpisodeSerializer.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        include_features: bool = True,
        include_actions: bool = True,
    ) -> None:
        """Initialize text builder.

        Args:
            max_tokens: Maximum approximate tokens in output.
            include_features: Include node features in text.
            include_actions: Include action history in text.
        """
        self.max_tokens = max_tokens
        self.include_features = include_features
        self.include_actions = include_actions

    def build(self, episode_data: Any) -> str:
        """Build text representation of episode.

        Args:
            episode_data: EpisodeData object from collector.

        Returns:
            Text representation for judge model.
        """
        lines = []

        lines.append(f"[EPISODE id={episode_data.episode_id}]")

        if episode_data.graph_stats:
            stats = episode_data.graph_stats
            lines.append(
                f"[GRAPH stats: {stats.get('num_nodes', 0)} nodes, "
                f"{stats.get('num_edges', 0)} edges, "
                f"density={stats.get('density', 0):.4f}]"
            )

        if episode_data.flagged_nodes:
            lines.append(f"\n[FLAGGED_TRANSACTIONS count={len(episode_data.flagged_nodes)}]")
            for node_id in episode_data.flagged_nodes[:20]:
                step = self._find_flag_step(episode_data, node_id)
                if step:
                    features_str = ""
                    if self.include_features and step.node_features:
                        features_str = f", features={list(step.node_features.keys())[:5]}"
                    lines.append(
                        f"- TXN node={node_id}, "
                        f"confidence={step.confidence:.2f}, "
                        f"is_fraud={step.is_fraud}"
                        f"{features_str}"
                    )

        if self.include_actions and episode_data.steps:
            lines.append(f"\n[AGENT_ACTIONS count={len(episode_data.steps)}]")
            for step in episode_data.steps[:30]:
                lines.append(
                    f"- STEP {step.step_number}: {step.action_type} "
                    f"node={step.node_id} "
                    f"reward={step.reward:.3f}"
                )

        lines.append("\n[METRICS]")
        lines.append(f"precision={episode_data.precision:.3f}")
        lines.append(f"recall={episode_data.recall:.3f}")
        lines.append(f"f1={episode_data.f1_score:.3f}")
        lines.append(f"true_positives={episode_data.true_positives}")
        lines.append(f"false_positives={episode_data.false_positives}")

        text = "\n".join(lines)

        approx_tokens = len(text.split())
        if approx_tokens > self.max_tokens:
            text = self._truncate(text)

        return text

    def _find_flag_step(self, episode_data: Any, node_id: str) -> Optional[Any]:
        """Find the step where a node was flagged."""
        for step in episode_data.steps:
            if step.node_id == node_id and step.action_type == "FLAG":
                return step
        return None

    def _truncate(self, text: str) -> str:
        """Truncate text to approximate token limit."""
        words = text.split()
        if len(words) <= self.max_tokens:
            return text

        truncated = words[:self.max_tokens]
        return " ".join(truncated) + "\n[TRUNCATED]"
