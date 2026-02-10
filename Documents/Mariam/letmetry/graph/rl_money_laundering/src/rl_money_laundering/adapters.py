"""Compatibility adapters for protocol-conforming interfaces."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional

import numpy as np

from .protocols import AgentActionProtocol, CallbackLifecycleProtocol


class LegacyQValueAgentAdapter(AgentActionProtocol):
    """Adapt legacy q-value callables to the canonical select_action protocol."""

    def __init__(
        self,
        q_value_fn: Callable[[np.ndarray], np.ndarray],
        action_dim: int,
    ) -> None:
        self._q_value_fn = q_value_fn
        self._action_dim = action_dim

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[list[int]],
        epsilon: Optional[float],
    ) -> int:
        if epsilon is None:
            epsilon = 0.0

        if valid_actions is not None and len(valid_actions) == 0:
            raise ValueError("valid_actions cannot be an empty list")

        if random.random() < epsilon:
            if valid_actions is None:
                return random.randint(0, self._action_dim - 1)
            return random.choice(valid_actions)

        q_values = self._q_value_fn(state)
        if valid_actions is not None:
            mask = np.full(self._action_dim, -np.inf)
            mask[valid_actions] = 0.0
            q_values = q_values + mask
        return int(np.argmax(q_values))


class CallbackLifecycleAdapter(CallbackLifecycleProtocol):
    """Adapt callbacks with partially-overlapping method names to one lifecycle."""

    def __init__(self, callback: Any) -> None:
        self._callback = callback

    def on_episode_start(self, **kwargs: Any) -> None:
        hook = getattr(self._callback, "on_episode_start", None)
        if callable(hook):
            hook(**kwargs)

    def on_episode_step(self, **kwargs: Any) -> None:
        hook = getattr(self._callback, "on_episode_step", None)
        if callable(hook):
            hook(**kwargs)

    def on_episode_end(self, **kwargs: Any) -> Dict[str, Any]:
        hook = getattr(self._callback, "on_episode_end", None)
        if callable(hook):
            result = hook(**kwargs)
            if isinstance(result, dict):
                return result
        return {}

    def should_update(self, episode: Optional[int] = None) -> bool:
        hook = getattr(self._callback, "should_update", None)
        if callable(hook):
            try:
                return bool(hook(episode))
            except TypeError:
                return bool(hook())

        fallback = getattr(self._callback, "_should_update_judge", None)
        if callable(fallback) and episode is not None:
            return bool(fallback(episode))
        return False

    def on_train_result(self, result: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        hook = getattr(self._callback, "on_train_result", None)
        if callable(hook):
            output = hook(result=result, **kwargs)
            if isinstance(output, dict):
                return output
        return result
