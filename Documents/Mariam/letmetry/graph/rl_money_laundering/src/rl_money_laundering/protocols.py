"""Shared structural protocols for component interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

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

