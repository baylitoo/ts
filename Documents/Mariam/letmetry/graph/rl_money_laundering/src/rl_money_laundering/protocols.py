"""Shared structural protocols for component interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class AgentActionProtocol(Protocol):
    """Canonical action-selection interface for RL agents."""

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[List[int]],
        epsilon: Optional[float],
    ) -> int:
        """Select an action for a state under an exploration rate."""
        ...


@runtime_checkable
class JudgeModelProtocol(Protocol):
    """Canonical judge model interface for reward and inference paths."""

    def forward(
        self,
        episodes: Dict[str, Any] | Sequence[Dict[str, Any]],
        graph_embeddings: Optional[Any] = None,
        return_uncertainty: bool = True,
        return_fraud_types: bool = True,
    ) -> Dict[str, Any]:
        """Run forward inference for one or many serialized episodes."""
        ...

    def compute_reward(
        self,
        episode: Dict[str, Any],
        graph_embedding: Optional[Any] = None,
        conservative: bool = True,
    ) -> tuple[float, Dict[str, Any]]:
        """Compute scalar reward and supporting metadata for an episode."""
        ...


@runtime_checkable
class RewardComputationProtocol(Protocol):
    """Canonical episode reward interface across reward components."""

    def compute_reward(
        self,
        episode_data: Any,
        *,
        original_reward: float = 0.0,
        true_labels: Optional[Dict[str, bool]] = None,
        graph_embedding: Optional[Any] = None,
        episode_text: Optional[str] = None,
    ) -> tuple[float, Dict[str, Any]]:
        """Compute episode reward and return detailed reward metadata."""
        ...

@runtime_checkable
class CallbackLifecycleProtocol(Protocol):
    """Canonical callback lifecycle shared by custom and RLlib callback paths."""

    def on_episode_start(self, **kwargs: Any) -> None:
        ...

    def on_episode_step(self, **kwargs: Any) -> None:
        ...

    def on_episode_end(self, **kwargs: Any) -> Dict[str, Any]:
        ...

    def should_update(self, episode: Optional[int] = None) -> bool:
        ...

    def on_train_result(self, result: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...


@runtime_checkable
class EncoderBridgeProtocol(Protocol):
    """Canonical interface for encoder bridge wrappers used by integration flows."""

    def encode_state(
        self,
        graph: Any,
        node_id: Any,
        visited_nodes: set[Any],
        visited_edges: set[Any],
        edge_time: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> Any:
        ...

    def encode_batch(
        self,
        graph: Any,
        node_ids: List[Any],
        visited_nodes_list: List[set[Any]],
        visited_edges_list: List[set[Any]],
        edge_times: Optional[List[float]] = None,
    ) -> List[Any]:
        ...

