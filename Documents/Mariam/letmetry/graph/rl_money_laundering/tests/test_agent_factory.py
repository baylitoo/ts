"""Tests for pipeline agent instantiation."""

from __future__ import annotations

import pytest

from rl_money_laundering.agent import DQNAgent, QRDQNAgent
from rl_money_laundering.config import AgentConfig
from rl_money_laundering.pipeline import _build_agent_from_config


@pytest.mark.parametrize(
    ("agent_type", "expected_cls"),
    [
        ("dqn", DQNAgent),
        ("qrdqn", QRDQNAgent),
    ],
)
def test_build_agent_from_config_returns_expected_type(agent_type: str, expected_cls: type[object]) -> None:
    """Factory should instantiate the correct agent implementation."""
    cfg = AgentConfig(
        state_dim=32,
        action_dim=6,
        agent_type=agent_type,
        device="cpu",
        hidden_dims=[64, 32],
        use_double_dqn=False,
        positive_fraction=0.3,
    )

    agent = _build_agent_from_config(cfg)

    assert isinstance(agent, expected_cls)
    assert getattr(agent, "action_dim") == cfg.action_dim
    assert getattr(agent, "positive_fraction") == pytest.approx(cfg.positive_fraction)
    assert getattr(agent, "use_double_dqn") is cfg.use_double_dqn


def test_build_agent_from_config_applies_qrdqn_specific_parameters() -> None:
    """QR-DQN specific settings should reach the instantiated agent."""
    cfg = AgentConfig(
        state_dim=24,
        action_dim=5,
        agent_type="qrdqn",
        device="cpu",
        hidden_dims=[128],
        num_quantiles=32,
        n_step=5,
        risk_measure="cvar_90",
        prioritized_alpha=0.7,
        prioritized_beta=0.5,
        prioritized_beta_annealing=0.01,
        positive_fraction=0.4,
        use_double_dqn=False,
    )

    agent = _build_agent_from_config(cfg)

    assert isinstance(agent, QRDQNAgent)
    assert agent.num_quantiles == cfg.num_quantiles
    assert agent.n_step == cfg.n_step
    assert agent.risk_measure == cfg.risk_measure
    assert agent.prioritized_alpha == pytest.approx(cfg.prioritized_alpha)
    assert agent.prioritized_beta == pytest.approx(cfg.prioritized_beta)
    assert agent.prioritized_beta_annealing == pytest.approx(cfg.prioritized_beta_annealing)
    assert agent.use_double_dqn is cfg.use_double_dqn
