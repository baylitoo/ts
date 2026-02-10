"""Protocol conformance tests for shared interfaces."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_money_laundering.agent import DQNAgent, QRDQNAgent
from rl_money_laundering.multiagent.integration.config import IntegrationConfig
from rl_money_laundering.multiagent.integration.reward_integrator import RewardIntegrator
from rl_money_laundering.protocols import AgentActionProtocol, RewardComputationProtocol


def test_dqn_agent_conforms_to_agent_action_protocol() -> None:
    agent = DQNAgent(state_dim=8, action_dim=6, epsilon_start=0.0)

    assert isinstance(agent, AgentActionProtocol)

    state = np.zeros(8, dtype=np.float32)
    action = agent.select_action(state=state, valid_actions=[0, 2, 4], epsilon=0.0)
    assert action in [0, 2, 4]


def test_qrdqn_agent_conforms_to_agent_action_protocol() -> None:
    agent = QRDQNAgent(state_dim=8, action_dim=6, epsilon_start=0.0, device="cpu")

    assert isinstance(agent, AgentActionProtocol)

    state = np.zeros(8, dtype=np.float32)
    action = agent.select_action(state=state, valid_actions=[1, 3, 5], epsilon=0.0)
    assert action in [1, 3, 5]


def test_reward_integrator_conforms_to_reward_computation_protocol() -> None:
    config = IntegrationConfig()
    integrator = RewardIntegrator(config=config, judge_model=None)

    assert isinstance(integrator, RewardComputationProtocol)

    reward, metadata = integrator.compute_reward(episode_data=None, original_reward=1.25)
    assert isinstance(reward, float)
    assert isinstance(metadata, dict)
    assert "combined_reward" in metadata
