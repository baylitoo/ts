"""Shared structural protocols for component interfaces."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

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
