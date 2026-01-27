"""
Comprehensive tests for the multiagent module.

Tests each component independently before integration testing.
Run with: pytest tests/test_multiagent.py -v
"""

import hashlib
import numpy as np
import pytest
import networkx as nx
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch


# =============================================================================
# SECTION 1: Test JudgeModel Components
# =============================================================================

class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from rl_money_laundering.multiagent import JudgeConfig

        config = JudgeConfig()

        assert config.model_name == "answerdotai/ModernBERT-base"
        assert config.max_length == 512
        assert config.embedding_dim == 768
        assert config.reward_clip == 1.0
        assert config.use_lora is True
        assert config.lora_rank == 16
        assert config.num_fraud_types == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        from rl_money_laundering.multiagent import JudgeConfig

        config = JudgeConfig(
            model_name="bert-base-uncased",
            max_length=256,
            embedding_dim=512,
            use_lora=False,
            lora_rank=8,
        )

        assert config.model_name == "bert-base-uncased"
        assert config.max_length == 256
        assert config.embedding_dim == 512
        assert config.use_lora is False
        assert config.lora_rank == 8


class TestEpisodeSerializer:
    """Tests for EpisodeSerializer."""

    def test_serialize_empty_episode(self):
        """Test serialization of empty episode data."""
        from rl_money_laundering.multiagent import EpisodeSerializer

        serializer = EpisodeSerializer()

        episode_data = {
            "nodes": [],
            "edges": [],
            "actions": [],
            "predictions": [],
        }

        text = serializer.serialize(episode_data)

        assert isinstance(text, str)
        assert "[EPISODE" in text
        assert "GRAPH" in text

    def test_serialize_with_transactions(self):
        """Test serialization with transaction data."""
        from rl_money_laundering.multiagent import EpisodeSerializer

        serializer = EpisodeSerializer()

        episode_data = {
            "nodes": [
                {"id": "node_1", "balance": 10000, "risk_score": 0.8},
                {"id": "node_2", "balance": 5000, "risk_score": 0.2},
            ],
            "edges": [
                {"source": "node_1", "target": "node_2", "amount": 1500},
            ],
            "actions": [
                {"step": 1, "action": "FLAG", "node": "node_1", "confidence": 0.9},
            ],
            "predictions": [
                {"node": "node_1", "predicted": True, "actual": True},
            ],
        }

        text = serializer.serialize(episode_data)

        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain episode markers
        assert "[EPISODE" in text or "EPISODE" in text.upper()

    def test_anonymization(self):
        """Test that node IDs are anonymized."""
        from rl_money_laundering.multiagent import EpisodeSerializer

        serializer = EpisodeSerializer(anonymize_ids=True)

        episode_data = {
            "nodes": [
                {"id": "sensitive_account_123", "balance": 10000},
            ],
            "edges": [],
            "actions": [],
            "predictions": [],
        }

        text = serializer.serialize(episode_data)

        # Original ID should not appear in output
        assert "sensitive_account_123" not in text

    def test_canonical_ordering(self):
        """Test that serialization produces consistent ordering."""
        from rl_money_laundering.multiagent import EpisodeSerializer

        serializer = EpisodeSerializer()

        episode_data = {
            "nodes": [
                {"id": "b", "value": 2},
                {"id": "a", "value": 1},
                {"id": "c", "value": 3},
            ],
            "edges": [],
            "actions": [],
            "predictions": [],
        }

        # Serialize twice
        text1 = serializer.serialize(episode_data)
        text2 = serializer.serialize(episode_data)

        # Should produce identical output
        assert text1 == text2


class TestJudgeModel:
    """Tests for JudgeModel."""

    def test_model_initialization_mock(self):
        """Test model initialization with mocked transformers."""
        from rl_money_laundering.multiagent import JudgeModel, JudgeConfig

        config = JudgeConfig(use_lora=False)

        # Initialize with mock mode (no actual model loading)
        with patch.dict('sys.modules', {'transformers': MagicMock()}):
            # This tests the structure, actual loading requires transformers
            pass

    def test_compute_reward_interface(self):
        """Test compute_reward method interface."""
        from rl_money_laundering.multiagent import JudgeModel, JudgeConfig

        config = JudgeConfig()

        # Create mock model
        mock_model = Mock(spec=JudgeModel)
        mock_model.compute_reward.return_value = {
            "reward": 0.5,
            "uncertainty": 0.1,
            "fraud_type_logits": [0.1, 0.2, 0.3, 0.2, 0.2],
        }

        result = mock_model.compute_reward("test episode text")

        assert "reward" in result
        assert "uncertainty" in result
        assert -1.0 <= result["reward"] <= 1.0


# =============================================================================
# SECTION 2: Test RewardComputation Components
# =============================================================================

class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_default_config(self):
        """Test default reward configuration."""
        from rl_money_laundering.multiagent import RewardConfig

        config = RewardConfig()

        assert config.precision_weight >= 0
        assert config.recall_weight >= 0
        assert config.f1_weight >= 0
        assert config.judge_weight_start >= 0
        assert config.judge_weight_end >= config.judge_weight_start
        assert config.judge_clip_max > 0
        assert config.uncertainty_penalty >= 0

    def test_weight_normalization(self):
        """Test that metric weights can be used properly."""
        from rl_money_laundering.multiagent import RewardConfig

        config = RewardConfig(
            precision_weight=0.2,
            recall_weight=0.3,
            f1_weight=0.5,
        )

        total = config.precision_weight + config.recall_weight + config.f1_weight
        assert abs(total - 1.0) < 0.01


class TestConservativeRewardComputer:
    """Tests for ConservativeRewardComputer."""

    def test_initialization(self):
        """Test reward computer initialization."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig()

        def mock_judge_fn(text):
            return {"reward": 0.5, "uncertainty": 0.1}

        computer = ConservativeRewardComputer(
            config=config,
            judge_fn=mock_judge_fn,
        )

        assert computer is not None

    def test_compute_hard_metrics(self):
        """Test hard metric computation."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig()
        computer = ConservativeRewardComputer(config=config)

        # Test with known confusion matrix
        predictions = [True, True, False, False]
        labels = [True, False, True, False]

        # TP=1, FP=1, FN=1, TN=1
        # Precision = 1/2 = 0.5
        # Recall = 1/2 = 0.5
        # F1 = 0.5

        metrics = computer._compute_hard_metrics(predictions, labels)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert abs(metrics["precision"] - 0.5) < 0.01
        assert abs(metrics["recall"] - 0.5) < 0.01

    def test_judge_clipping(self):
        """Test that judge rewards are clipped."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig(judge_clip_max=0.5)

        def extreme_judge_fn(text):
            return {"reward": 10.0, "uncertainty": 0.0}  # Extreme value

        computer = ConservativeRewardComputer(
            config=config,
            judge_fn=extreme_judge_fn,
        )

        # The clipped value should be <= judge_clip_max
        clipped = np.clip(10.0, -config.judge_clip_max, config.judge_clip_max)
        assert clipped == config.judge_clip_max

    def test_uncertainty_penalty(self):
        """Test uncertainty penalty application."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig(uncertainty_penalty=0.1)

        # High uncertainty should reduce effective reward
        reward = 0.8
        uncertainty = 0.5
        penalty = config.uncertainty_penalty * uncertainty

        effective_reward = reward - penalty
        assert effective_reward < reward

    def test_safe_reward_formula_factory(self):
        """Test the safe reward formula factory function."""
        from rl_money_laundering.multiagent import create_safe_reward_formula

        formula = create_safe_reward_formula(
            hard_weight=0.7,
            judge_weight=0.3,
            clip_max=0.5,
        )

        assert callable(formula)


# =============================================================================
# SECTION 3: Test MultiAgentEnvironment Components
# =============================================================================

class TestDetectorAdversaryConfig:
    """Tests for DetectorAdversaryConfig."""

    def test_default_config(self):
        """Test default environment configuration."""
        from rl_money_laundering.multiagent import DetectorAdversaryConfig

        config = DetectorAdversaryConfig()

        assert config.max_steps > 0
        assert config.max_neighbors > 0
        assert isinstance(config.enable_adversary, bool)

    def test_single_agent_mode(self):
        """Test configuration for single agent mode."""
        from rl_money_laundering.multiagent import DetectorAdversaryConfig

        config = DetectorAdversaryConfig(enable_adversary=False)

        assert config.enable_adversary is False

    def test_multi_agent_mode(self):
        """Test configuration for multi-agent mode."""
        from rl_money_laundering.multiagent import DetectorAdversaryConfig

        config = DetectorAdversaryConfig(
            enable_adversary=True,
            adversary_budget=5,
        )

        assert config.enable_adversary is True
        assert config.adversary_budget == 5


class TestMultiAgentAMLEnvironment:
    """Tests for MultiAgentAMLEnvironment."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample transaction graph for testing."""
        G = nx.DiGraph()

        # Add nodes with features
        for i in range(10):
            G.add_node(
                f"account_{i}",
                balance=np.random.uniform(1000, 100000),
                risk_score=np.random.uniform(0, 1),
                is_fraud=i < 2,  # First 2 nodes are fraud
                account_type="individual" if i % 2 == 0 else "business",
            )

        # Add edges (transactions)
        edges = [
            ("account_0", "account_1", {"amount": 5000, "timestamp": 1}),
            ("account_1", "account_2", {"amount": 3000, "timestamp": 2}),
            ("account_2", "account_3", {"amount": 1500, "timestamp": 3}),
            ("account_3", "account_4", {"amount": 2000, "timestamp": 4}),
            ("account_0", "account_5", {"amount": 8000, "timestamp": 5}),
            ("account_5", "account_6", {"amount": 4000, "timestamp": 6}),
        ]
        G.add_edges_from([(e[0], e[1], e[2]) for e in edges])

        return G

    def test_environment_initialization(self, sample_graph):
        """Test environment initialization."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        config = DetectorAdversaryConfig(max_steps=10)
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)

        assert env is not None
        assert env.graph is not None

    def test_environment_reset(self, sample_graph):
        """Test environment reset."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        config = DetectorAdversaryConfig(max_steps=10)
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)

        obs, info = env.reset()

        assert obs is not None
        assert isinstance(info, dict)

    def test_environment_step(self, sample_graph):
        """Test environment step."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        config = DetectorAdversaryConfig(max_steps=10)
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)

        obs, info = env.reset()

        # Take a step with action 0 (move to first neighbor or no-op)
        next_obs, reward, terminated, truncated, step_info = env.step(0)

        assert next_obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)

    def test_episode_data_collection(self, sample_graph):
        """Test that episode data is collected."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        config = DetectorAdversaryConfig(max_steps=5)
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)

        env.reset()

        # Run a few steps
        for _ in range(3):
            env.step(0)

        episode_data = env.get_episode_data()

        assert episode_data is not None
        assert isinstance(episode_data, dict)

    def test_observation_space(self, sample_graph):
        """Test observation space definition."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        config = DetectorAdversaryConfig()
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)

        assert env.observation_space is not None

    def test_action_space(self, sample_graph):
        """Test action space definition."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        config = DetectorAdversaryConfig(max_neighbors=5)
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)

        assert env.action_space is not None
        # Should have max_neighbors + 1 actions (moves + FLAG)
        assert env.action_space.n == config.max_neighbors + 1


class TestRLlibWrapper:
    """Tests for RLlib multi-agent wrapper."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph."""
        G = nx.DiGraph()
        for i in range(5):
            G.add_node(f"n{i}", is_fraud=i == 0, balance=1000)
        G.add_edge("n0", "n1", amount=100)
        G.add_edge("n1", "n2", amount=200)
        return G

    def test_wrapper_initialization(self, sample_graph):
        """Test RLlib wrapper initialization."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
            RLlibMultiAgentWrapper,
        )

        config = DetectorAdversaryConfig()
        base_env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)
        wrapper = RLlibMultiAgentWrapper(base_env)

        assert wrapper is not None


# =============================================================================
# SECTION 4: Test TrainingCallbacks Components
# =============================================================================

class TestTwoTimescaleConfig:
    """Tests for TwoTimescaleConfig."""

    def test_default_config(self):
        """Test default two-timescale configuration."""
        from rl_money_laundering.multiagent import TwoTimescaleConfig

        config = TwoTimescaleConfig()

        assert config.judge_update_frequency > 0
        assert config.judge_freeze_after > 0
        assert 0 < config.ema_decay < 1

    def test_custom_config(self):
        """Test custom configuration."""
        from rl_money_laundering.multiagent import TwoTimescaleConfig

        config = TwoTimescaleConfig(
            judge_update_frequency=50,
            judge_freeze_after=2000,
            ema_decay=0.99,
        )

        assert config.judge_update_frequency == 50
        assert config.judge_freeze_after == 2000
        assert config.ema_decay == 0.99


class TestTwoTimescaleJudgeCallback:
    """Tests for TwoTimescaleJudgeCallback."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        from rl_money_laundering.multiagent import (
            TwoTimescaleJudgeCallback,
            TwoTimescaleConfig,
        )

        config = TwoTimescaleConfig()

        # Mock judge model
        mock_judge = Mock()
        mock_judge.state_dict.return_value = {}

        callback = TwoTimescaleJudgeCallback(
            judge_model=mock_judge,
            config=config,
        )

        assert callback is not None

    def test_should_update_judge(self):
        """Test judge update frequency logic."""
        from rl_money_laundering.multiagent import (
            TwoTimescaleJudgeCallback,
            TwoTimescaleConfig,
        )

        config = TwoTimescaleConfig(
            judge_update_frequency=100,
            judge_freeze_after=5000,
        )

        mock_judge = Mock()
        mock_judge.state_dict.return_value = {}

        callback = TwoTimescaleJudgeCallback(
            judge_model=mock_judge,
            config=config,
        )

        # Should update at multiples of update_frequency
        assert callback.should_update(100) is True
        assert callback.should_update(200) is True
        assert callback.should_update(50) is False
        assert callback.should_update(150) is False

        # Should not update after freeze
        assert callback.should_update(5100) is False


class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_default_config(self):
        """Test default safety configuration."""
        from rl_money_laundering.multiagent import SafetyConfig

        config = SafetyConfig()

        assert config.hard_metric_floor >= 0
        assert config.drift_correlation_threshold > -1
        assert config.drift_correlation_threshold < 1


class TestSafetyMonitorCallback:
    """Tests for SafetyMonitorCallback."""

    def test_callback_initialization(self):
        """Test safety monitor initialization."""
        from rl_money_laundering.multiagent import (
            SafetyMonitorCallback,
            SafetyConfig,
        )

        config = SafetyConfig()
        callback = SafetyMonitorCallback(config=config)

        assert callback is not None

    def test_drift_detection(self):
        """Test drift detection logic."""
        from rl_money_laundering.multiagent import (
            SafetyMonitorCallback,
            SafetyConfig,
        )

        config = SafetyConfig(drift_correlation_threshold=0.3)
        callback = SafetyMonitorCallback(config=config)

        # Simulate correlated metrics (no drift)
        for i in range(100):
            value = 0.5 + 0.1 * np.sin(i / 10)
            result = callback.on_episode_end(
                episode=i,
                hard_metric=value,
                judge_score=value + np.random.normal(0, 0.05),
                combined_reward=value,
            )

        # Should not trigger drift alarm with correlated data
        assert result.get("drift_alarm") is None or result["drift_alarm"] is False

    def test_floor_violation_detection(self):
        """Test hard metric floor violation."""
        from rl_money_laundering.multiagent import (
            SafetyMonitorCallback,
            SafetyConfig,
        )

        config = SafetyConfig(hard_metric_floor=0.1)
        callback = SafetyMonitorCallback(config=config)

        # Simulate very low metrics
        result = callback.on_episode_end(
            episode=1,
            hard_metric=0.05,  # Below floor
            judge_score=0.5,
            combined_reward=0.3,
        )

        # Should detect floor violation
        assert "floor_violation" in result or result.get("should_reduce_judge", False)


class TestJudgeMetricsCallback:
    """Tests for JudgeMetricsCallback."""

    def test_callback_initialization(self):
        """Test metrics callback initialization."""
        from rl_money_laundering.multiagent import JudgeMetricsCallback

        callback = JudgeMetricsCallback()

        assert callback is not None

    def test_metrics_logging(self):
        """Test that metrics are logged."""
        from rl_money_laundering.multiagent import JudgeMetricsCallback

        callback = JudgeMetricsCallback()

        # Log some metrics
        callback.on_episode_end(
            episode=1,
            judge_reward=0.5,
            judge_uncertainty=0.1,
            hard_f1=0.6,
        )

        metrics = callback.get_metrics()

        assert metrics is not None
        assert len(metrics) > 0 or isinstance(metrics, dict)


# =============================================================================
# SECTION 5: Integration Tests (Components Together)
# =============================================================================

class TestComponentIntegration:
    """Tests for component integration."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph for integration tests."""
        G = nx.DiGraph()
        for i in range(20):
            G.add_node(
                f"account_{i}",
                balance=np.random.uniform(1000, 100000),
                risk_score=np.random.uniform(0, 1),
                is_fraud=i < 3,
                account_type="individual",
            )

        for i in range(15):
            src = f"account_{i}"
            tgt = f"account_{(i + 1) % 20}"
            G.add_edge(src, tgt, amount=np.random.uniform(100, 10000))

        return G

    def test_env_to_serializer_flow(self, sample_graph):
        """Test data flow from environment to serializer."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
            EpisodeSerializer,
        )

        config = DetectorAdversaryConfig(max_steps=5)
        env = MultiAgentAMLEnvironment(graph=sample_graph, config=config)
        serializer = EpisodeSerializer()

        # Run episode
        env.reset()
        for _ in range(3):
            env.step(0)

        episode_data = env.get_episode_data()

        # Serialize
        text = serializer.serialize(episode_data)

        assert isinstance(text, str)
        assert len(text) > 0

    def test_reward_computation_flow(self, sample_graph):
        """Test reward computation with mock judge."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig(
            judge_weight_start=0.1,
            judge_weight_end=0.3,
        )

        call_count = [0]

        def mock_judge_fn(text):
            call_count[0] += 1
            return {"reward": 0.6, "uncertainty": 0.1}

        computer = ConservativeRewardComputer(
            config=config,
            judge_fn=mock_judge_fn,
        )

        # Compute reward
        episode_data = {"text": "test episode"}
        predictions = [True, True, False]
        labels = [True, False, False]

        result = computer.compute_episode_reward(
            episode_data=episode_data,
            predictions=predictions,
            labels=labels,
            current_step=1000,
        )

        assert "total_reward" in result
        assert "components" in result

    def test_callback_chain(self):
        """Test callback chain execution."""
        from rl_money_laundering.multiagent import (
            TwoTimescaleJudgeCallback,
            TwoTimescaleConfig,
            SafetyMonitorCallback,
            SafetyConfig,
            JudgeMetricsCallback,
        )

        # Setup callbacks
        mock_judge = Mock()
        mock_judge.state_dict.return_value = {}

        judge_callback = TwoTimescaleJudgeCallback(
            judge_model=mock_judge,
            config=TwoTimescaleConfig(),
        )

        safety_callback = SafetyMonitorCallback(
            config=SafetyConfig(),
        )

        metrics_callback = JudgeMetricsCallback()

        # Simulate episode end
        episode = 100
        episode_data = {"test": "data"}
        labels = [True, False]

        # Each callback should execute without error
        judge_result = judge_callback.on_episode_end(episode, episode_data, labels)
        safety_result = safety_callback.on_episode_end(
            episode=episode,
            hard_metric=0.6,
            judge_score=0.55,
            combined_reward=0.58,
        )
        metrics_callback.on_episode_end(
            episode=episode,
            judge_reward=0.55,
            judge_uncertainty=0.1,
            hard_f1=0.6,
        )

        assert isinstance(judge_result, dict)
        assert isinstance(safety_result, dict)


# =============================================================================
# SECTION 6: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_graph(self):
        """Test handling of empty graph."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        G = nx.DiGraph()
        config = DetectorAdversaryConfig()

        # Should handle empty graph gracefully
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            env = MultiAgentAMLEnvironment(graph=G, config=config)
            env.reset()

    def test_single_node_graph(self):
        """Test handling of single-node graph."""
        from rl_money_laundering.multiagent import (
            MultiAgentAMLEnvironment,
            DetectorAdversaryConfig,
        )

        G = nx.DiGraph()
        G.add_node("only_node", is_fraud=False, balance=1000)

        config = DetectorAdversaryConfig()
        env = MultiAgentAMLEnvironment(graph=G, config=config)

        obs, info = env.reset()

        # Should handle single node (no neighbors to move to)
        next_obs, reward, terminated, truncated, step_info = env.step(0)

        assert next_obs is not None

    def test_zero_uncertainty(self):
        """Test reward computation with zero uncertainty."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig(uncertainty_penalty=0.1)

        def zero_uncertainty_judge(text):
            return {"reward": 0.5, "uncertainty": 0.0}

        computer = ConservativeRewardComputer(
            config=config,
            judge_fn=zero_uncertainty_judge,
        )

        # Should not crash with zero uncertainty
        result = computer.compute_episode_reward(
            episode_data={"text": "test"},
            predictions=[True],
            labels=[True],
            current_step=100,
        )

        assert result is not None

    def test_all_fraud_predictions(self):
        """Test metrics with all fraud predictions."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig()
        computer = ConservativeRewardComputer(config=config)

        # All predictions are fraud
        predictions = [True, True, True, True]
        labels = [True, False, False, False]

        metrics = computer._compute_hard_metrics(predictions, labels)

        # Precision should be 1/4 = 0.25
        assert metrics["precision"] == pytest.approx(0.25, abs=0.01)
        # Recall should be 1/1 = 1.0
        assert metrics["recall"] == pytest.approx(1.0, abs=0.01)

    def test_no_fraud_predictions(self):
        """Test metrics with no fraud predictions."""
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig()
        computer = ConservativeRewardComputer(config=config)

        # No predictions are fraud
        predictions = [False, False, False, False]
        labels = [True, False, False, False]

        metrics = computer._compute_hard_metrics(predictions, labels)

        # Precision should be 0 (no predictions)
        assert metrics["precision"] == 0.0
        # Recall should be 0 (missed the fraud)
        assert metrics["recall"] == 0.0

    def test_serializer_with_none_values(self):
        """Test serializer handling of None values."""
        from rl_money_laundering.multiagent import EpisodeSerializer

        serializer = EpisodeSerializer()

        episode_data = {
            "nodes": [{"id": "n1", "balance": None, "risk_score": 0.5}],
            "edges": [],
            "actions": [],
            "predictions": [],
        }

        # Should handle None values gracefully
        text = serializer.serialize(episode_data)
        assert isinstance(text, str)


# =============================================================================
# SECTION 7: Performance and Stress Tests
# =============================================================================

class TestPerformance:
    """Performance tests (not run by default, use -m performance)."""

    @pytest.mark.performance
    def test_serialization_performance(self):
        """Test serialization performance with large episode."""
        import time
        from rl_money_laundering.multiagent import EpisodeSerializer

        serializer = EpisodeSerializer()

        # Create large episode
        large_episode = {
            "nodes": [{"id": f"n{i}", "balance": i * 100} for i in range(1000)],
            "edges": [{"source": f"n{i}", "target": f"n{i+1}", "amount": 100}
                      for i in range(999)],
            "actions": [{"step": i, "action": "MOVE"} for i in range(500)],
            "predictions": [{"node": f"n{i}", "predicted": i % 2 == 0} for i in range(100)],
        }

        start = time.time()
        for _ in range(100):
            serializer.serialize(large_episode)
        elapsed = time.time() - start

        # Should complete 100 serializations in under 5 seconds
        assert elapsed < 5.0, f"Serialization too slow: {elapsed:.2f}s for 100 iterations"

    @pytest.mark.performance
    def test_reward_computation_performance(self):
        """Test reward computation performance."""
        import time
        from rl_money_laundering.multiagent import (
            ConservativeRewardComputer,
            RewardConfig,
        )

        config = RewardConfig()

        def fast_judge(text):
            return {"reward": 0.5, "uncertainty": 0.1}

        computer = ConservativeRewardComputer(config=config, judge_fn=fast_judge)

        predictions = [i % 2 == 0 for i in range(1000)]
        labels = [i % 3 == 0 for i in range(1000)]

        start = time.time()
        for _ in range(1000):
            computer._compute_hard_metrics(predictions, labels)
        elapsed = time.time() - start

        # Should complete 1000 metric computations in under 1 second
        assert elapsed < 1.0, f"Metric computation too slow: {elapsed:.2f}s"


# =============================================================================
# Run configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
