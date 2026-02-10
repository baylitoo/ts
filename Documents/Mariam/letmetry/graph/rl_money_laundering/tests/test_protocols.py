"""Protocol conformance tests for shared interfaces."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_money_laundering.adapters import CallbackLifecycleAdapter, LegacyQValueAgentAdapter
from rl_money_laundering.protocols import (
    AgentActionProtocol,
    CallbackLifecycleProtocol,
    EncoderBridgeProtocol,
    JudgeModelProtocol,
    RewardComputationProtocol,
)


def _load_integrator_classes() -> tuple[type, type]:
    pytest.importorskip("torch")
    from rl_money_laundering.multiagent.integration.config import IntegrationConfig
    from rl_money_laundering.multiagent.integration.reward_integrator import RewardIntegrator

    return IntegrationConfig, RewardIntegrator


def _load_agent_classes() -> tuple[type, type]:
    pytest.importorskip("torch")
    from rl_money_laundering.agent import DQNAgent, QRDQNAgent

    return DQNAgent, QRDQNAgent


def test_dqn_agent_conforms_to_agent_action_protocol() -> None:
    DQNAgent, _ = _load_agent_classes()
    agent = DQNAgent(state_dim=8, action_dim=6, epsilon_start=0.0)

    assert isinstance(agent, AgentActionProtocol)

    state = np.zeros(8, dtype=np.float32)
    action = agent.select_action(state=state, valid_actions=[0, 2, 4], epsilon=0.0)
    assert action in [0, 2, 4]


def test_qrdqn_agent_conforms_to_agent_action_protocol() -> None:
    _, QRDQNAgent = _load_agent_classes()
    agent = QRDQNAgent(state_dim=8, action_dim=6, epsilon_start=0.0, device="cpu")

    assert isinstance(agent, AgentActionProtocol)

    state = np.zeros(8, dtype=np.float32)
    action = agent.select_action(state=state, valid_actions=[1, 3, 5], epsilon=0.0)
    assert action in [1, 3, 5]


def test_reward_integrator_conforms_to_reward_computation_protocol() -> None:
    IntegrationConfig, RewardIntegrator = _load_integrator_classes()
    config = IntegrationConfig()
    integrator = RewardIntegrator(config=config, judge_model=None)

    assert isinstance(integrator, RewardComputationProtocol)

    reward, metadata = integrator.compute_reward(episode_data=None, original_reward=1.25)
    assert isinstance(reward, float)
    assert isinstance(metadata, dict)
    assert "combined_reward" in metadata


class _DummyJudge:
    def forward(
        self,
        episodes,
        graph_embeddings=None,
        return_uncertainty: bool = True,
        return_fraud_types: bool = True,
    ) -> dict[str, float]:
        return {"reward": 0.1, "uncertainty": 0.2}

    def compute_reward(
        self,
        episode,
        graph_embedding=None,
        conservative: bool = True,
    ) -> tuple[float, dict[str, float]]:
        return 0.1, {"uncertainty": 0.2}


class _DummyStateEncoder:
    def forward_state(
        self,
        graph: Any,
        current_node: Any,
        visited_nodes: set,
        visited_edges: list,
        node_feature_extractor: Any,
        return_auxiliary: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return np.array([1.0, 2.0]), np.array([0.5]), {"node": current_node}


class _SimpleCallback:
    def on_episode_end(self, **kwargs: Any) -> dict[str, Any]:
        return {"episode": kwargs.get("episode", -1)}

    def on_train_result(self, result: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = dict(result)
        result["updated"] = True
        return result

    def _should_update_judge(self, episode: int) -> bool:
        return episode % 2 == 0


def test_judge_protocol_runtime_conformance() -> None:
    judge = _DummyJudge()
    assert isinstance(judge, JudgeModelProtocol)


def test_encoder_bridge_protocol_runtime_conformance() -> None:
    pytest.importorskip("torch")
    from rl_money_laundering.multiagent.integration.encoder_bridge import EncoderBridge

    bridge = EncoderBridge(state_encoder=_DummyStateEncoder(), judge_model=None)
    assert isinstance(bridge, EncoderBridgeProtocol)

    output = bridge.encode_state(
        graph=None,
        node_id="A",
        visited_nodes=set(),
        visited_edges=set(),
    )
    assert output.embedding.shape == (2,)


def test_callback_lifecycle_adapter_conforms_to_protocol() -> None:
    adapted = CallbackLifecycleAdapter(_SimpleCallback())
    assert isinstance(adapted, CallbackLifecycleProtocol)

    assert adapted.on_episode_end(episode=4)["episode"] == 4
    assert adapted.should_update(episode=4) is True
    assert adapted.on_train_result(result={"base": 1})["updated"] is True


def test_legacy_qvalue_adapter_conforms_to_agent_protocol() -> None:
    def q_value_fn(_: np.ndarray) -> np.ndarray:
        return np.array([0.1, -0.2, 0.9, 0.0])

    adapter = LegacyQValueAgentAdapter(q_value_fn=q_value_fn, action_dim=4)
    assert isinstance(adapter, AgentActionProtocol)

    action = adapter.select_action(
        state=np.zeros(4),
        valid_actions=[1, 3],
        epsilon=0.0,
    )
    assert action == 3
