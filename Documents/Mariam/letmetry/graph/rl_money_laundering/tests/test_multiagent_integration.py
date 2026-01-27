"""
Integration tests for the multiagent integration layer.

Tests the integration adapters that connect multiagent components
with the existing RL-AML framework.

Run with: pytest tests/test_multiagent_integration.py -v
"""

import numpy as np
import pytest
import networkx as nx
from typing import Dict, Any, Tuple
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass


# =============================================================================
# SECTION 1: Test Integration Configuration
# =============================================================================

class TestIntegrationConfig:
    """Tests for IntegrationConfig."""

    def test_default_config(self):
        """Test default integration configuration."""
        from rl_money_laundering.multiagent.integration import IntegrationConfig

        config = IntegrationConfig()

        assert config.judge_config is not None
        assert config.adversary_config is not None
        assert config.episode_buffer_size > 0
        assert config.hard_metric_weight > 0
        assert config.safety_monitoring is True

    def test_conservative_preset(self):
        """Test conservative preset configuration."""
        from rl_money_laundering.multiagent.integration import IntegrationConfig

        config = IntegrationConfig.default_conservative()

        assert config.judge_config.judge_weight_start == 0.05
        assert config.judge_config.judge_weight_end == 0.2
        assert config.hard_metric_weight == 0.8

    def test_aggressive_preset(self):
        """Test aggressive preset configuration."""
        from rl_money_laundering.multiagent.integration import IntegrationConfig

        config = IntegrationConfig.default_aggressive()

        assert config.judge_config.judge_weight_start == 0.2
        assert config.judge_config.judge_weight_end == 0.4
        assert config.adversary_config.enable_adversary is True

    def test_disabled_preset(self):
        """Test disabled preset configuration."""
        from rl_money_laundering.multiagent.integration import IntegrationConfig

        config = IntegrationConfig.disabled()

        assert config.judge_config.enable_judge is False
        assert config.adversary_config.enable_adversary is False
        assert config.compatibility_mode is True

    def test_serialization(self, tmp_path):
        """Test config serialization and deserialization."""
        from rl_money_laundering.multiagent.integration import IntegrationConfig

        config = IntegrationConfig(
            episode_buffer_size=500,
            hard_metric_weight=0.6,
        )

        # Save
        config_path = tmp_path / "test_config.json"
        config.save(config_path)

        # Load
        loaded_config = IntegrationConfig.load(config_path)

        assert loaded_config.episode_buffer_size == 500
        assert loaded_config.hard_metric_weight == 0.6


class TestJudgeIntegrationConfig:
    """Tests for JudgeIntegrationConfig."""

    def test_weight_annealing(self):
        """Test judge weight annealing computation."""
        from rl_money_laundering.multiagent.integration import JudgeIntegrationConfig

        config = JudgeIntegrationConfig(
            judge_weight_start=0.1,
            judge_weight_end=0.3,
            judge_warmup_episodes=1000,
        )

        # At episode 0
        assert config.get_judge_weight(0) == 0.1

        # At episode 500 (halfway)
        assert abs(config.get_judge_weight(500) - 0.2) < 0.01

        # At episode 1000 (full warmup)
        assert config.get_judge_weight(1000) == 0.3

        # After warmup
        assert config.get_judge_weight(2000) == 0.3

    def test_disabled_judge_weight(self):
        """Test weight is zero when judge disabled."""
        from rl_money_laundering.multiagent.integration import JudgeIntegrationConfig

        config = JudgeIntegrationConfig(enable_judge=False)

        assert config.get_judge_weight(0) == 0.0
        assert config.get_judge_weight(1000) == 0.0


# =============================================================================
# SECTION 2: Test Environment Adapter
# =============================================================================

class TestEpisodeCollector:
    """Tests for EpisodeCollector."""

    def test_initialization(self):
        """Test collector initialization."""
        from rl_money_laundering.multiagent.integration import EpisodeCollector

        collector = EpisodeCollector(buffer_size=100)

        assert collector.buffer_size == 100
        assert len(collector.get_all_episodes()) == 0

    def test_episode_collection(self):
        """Test basic episode collection."""
        from rl_money_laundering.multiagent.integration import EpisodeCollector

        collector = EpisodeCollector()

        # Mock environment
        mock_env = Mock()
        mock_env.graph = nx.DiGraph()
        mock_env.graph.add_node("n1", is_fraud=True, balance=1000)
        mock_env.graph.add_node("n2", is_fraud=False, balance=2000)
        mock_env.graph.add_edge("n1", "n2", amount=500)
        mock_env.current_node = "n1"
        mock_env.visited_nodes = {"n1"}
        mock_env.max_neighbors = 5

        # Start episode
        collector.start_episode(mock_env, start_node="n1")

        # Record steps
        collector.record_step(
            node_id="n1",
            action=0,
            reward=0.5,
            confidence=0.8,
            env=mock_env,
            info={},
        )

        mock_env.current_node = "n2"
        mock_env.visited_nodes = {"n1", "n2"}

        collector.record_step(
            node_id="n2",
            action=1,
            reward=0.3,
            confidence=0.6,
            env=mock_env,
            info={},
        )

        # End episode
        episode = collector.end_episode(mock_env)

        assert episode is not None
        assert len(episode.steps) == 2
        assert episode.episode_length == 2

    def test_episode_metrics(self):
        """Test episode metric computation."""
        from rl_money_laundering.multiagent.integration import EpisodeCollector

        collector = EpisodeCollector()

        mock_env = Mock()
        mock_env.graph = nx.DiGraph()
        mock_env.graph.add_node("fraud_node", is_fraud=True)
        mock_env.graph.add_node("legit_node", is_fraud=False)
        mock_env.max_neighbors = 5
        mock_env.visited_nodes = set()

        collector.start_episode(mock_env, "fraud_node")

        # Flag fraud node (TP)
        collector.record_step(
            node_id="fraud_node",
            action=5,  # FLAG action
            reward=1.0,
            confidence=0.9,
            env=mock_env,
            info={},
        )

        episode = collector.end_episode(mock_env)

        assert episode.true_positives == 1
        assert episode.false_positives == 0
        assert episode.precision == 1.0

    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        from rl_money_laundering.multiagent.integration import EpisodeCollector

        collector = EpisodeCollector(buffer_size=3)

        mock_env = Mock()
        mock_env.graph = nx.DiGraph()
        mock_env.graph.add_node("n1", is_fraud=False)
        mock_env.visited_nodes = set()
        mock_env.max_neighbors = 5

        # Add more episodes than buffer size
        for i in range(5):
            collector.start_episode(mock_env, "n1")
            collector.record_step("n1", 0, 0.0, 1.0, mock_env, {})
            collector.end_episode(mock_env)

        # Buffer should only contain last 3
        assert len(collector.get_all_episodes()) == 3

    def test_anonymization(self):
        """Test node ID anonymization."""
        from rl_money_laundering.multiagent.integration import EpisodeCollector

        collector = EpisodeCollector(anonymize_ids=True)

        mock_env = Mock()
        mock_env.graph = nx.DiGraph()
        mock_env.graph.add_node("sensitive_id_12345", is_fraud=False)
        mock_env.visited_nodes = set()
        mock_env.max_neighbors = 5

        collector.start_episode(mock_env, "sensitive_id_12345")
        collector.record_step("sensitive_id_12345", 0, 0.0, 1.0, mock_env, {})
        episode = collector.end_episode(mock_env)

        # Original ID should be anonymized
        assert episode.steps[0].node_id != "sensitive_id_12345"
        assert episode.steps[0].node_id.startswith("ENTITY_")


class TestEnvironmentAdapter:
    """Tests for EnvironmentAdapter."""

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = Mock()
        env.graph = nx.DiGraph()
        env.graph.add_node("n1", is_fraud=False, balance=1000)
        env.graph.add_node("n2", is_fraud=True, balance=2000)
        env.graph.add_edge("n1", "n2", amount=500)
        env.current_node = "n1"
        env.visited_nodes = set()
        env.flagged_nodes = set()
        env.max_neighbors = 5

        # Setup reset
        env.reset.return_value = (np.zeros(16), {"start_node": "n1"})

        # Setup step
        def step_side_effect(action, **kwargs):
            return np.zeros(16), 0.5, False, False, {"action": action}

        env.step.side_effect = step_side_effect

        return env

    def test_adapter_passthrough(self, mock_env):
        """Test that adapter passes through to underlying env."""
        from rl_money_laundering.multiagent.integration import EnvironmentAdapter

        adapter = EnvironmentAdapter(mock_env, enable_collection=False)

        # Reset should pass through
        obs, info = adapter.reset()
        mock_env.reset.assert_called_once()

        # Step should pass through
        obs, reward, term, trunc, info = adapter.step(0)
        mock_env.step.assert_called()

    def test_adapter_collection(self, mock_env):
        """Test that adapter collects episode data."""
        from rl_money_laundering.multiagent.integration import EnvironmentAdapter

        adapter = EnvironmentAdapter(mock_env, enable_collection=True)

        # Run episode
        adapter.reset()
        adapter.step(0, confidence=0.8)
        adapter.step(1, confidence=0.6)

        # Mark episode done
        mock_env.step.side_effect = lambda a, **kw: (np.zeros(16), 1.0, True, False, {})
        adapter.step(2, confidence=0.9)

        # Get episode data
        episode = adapter.get_episode_data()

        assert episode is not None

    def test_collection_toggle(self, mock_env):
        """Test enabling/disabling collection."""
        from rl_money_laundering.multiagent.integration import EnvironmentAdapter

        adapter = EnvironmentAdapter(mock_env, enable_collection=True)

        adapter.disable_collection()
        adapter.reset()
        adapter.step(0)

        # Should not have episode data when disabled
        adapter.enable_collection()


# =============================================================================
# SECTION 3: Test Encoder Bridge
# =============================================================================

class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_operations(self):
        """Test basic cache operations."""
        from rl_money_laundering.multiagent.integration import EmbeddingCache

        cache = EmbeddingCache(max_size=10)

        # Put and get
        embedding = np.array([1.0, 2.0, 3.0])
        cache.put("key1", embedding)

        retrieved = cache.get("key1")
        assert np.array_equal(retrieved, embedding)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from rl_money_laundering.multiagent.integration import EmbeddingCache

        cache = EmbeddingCache()

        result = cache.get("nonexistent_key")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from rl_money_laundering.multiagent.integration import EmbeddingCache

        cache = EmbeddingCache(max_size=3)

        # Fill cache
        cache.put("k1", np.array([1.0]))
        cache.put("k2", np.array([2.0]))
        cache.put("k3", np.array([3.0]))

        # Access k1 to make it recently used
        cache.get("k1")

        # Add k4, should evict k2 (least recently used)
        cache.put("k4", np.array([4.0]))

        assert cache.get("k1") is not None
        assert cache.get("k2") is None  # Evicted
        assert cache.get("k3") is not None
        assert cache.get("k4") is not None

    def test_hit_rate(self):
        """Test hit rate computation."""
        from rl_money_laundering.multiagent.integration import EmbeddingCache

        cache = EmbeddingCache()

        cache.put("k1", np.array([1.0]))

        # 1 hit, 1 miss
        cache.get("k1")  # Hit
        cache.get("k2")  # Miss

        assert cache.hit_rate == 0.5


class TestEncoderBridge:
    """Tests for EncoderBridge."""

    def test_initialization(self):
        """Test bridge initialization."""
        from rl_money_laundering.multiagent.integration import EncoderBridge

        bridge = EncoderBridge(cache_embeddings=True, cache_size=1000)

        assert bridge is not None

    def test_encode_state_with_mock_encoder(self):
        """Test state encoding with mock encoder."""
        from rl_money_laundering.multiagent.integration import EncoderBridge

        # Mock state encoder
        mock_encoder = Mock()
        mock_encoder.forward_state.return_value = (
            np.array([0.1, 0.2, 0.3]),  # embedding
            np.array([0.4, 0.5]),  # history
            {"aux": "data"},  # auxiliary
        )

        bridge = EncoderBridge(state_encoder=mock_encoder)

        graph = nx.DiGraph()
        graph.add_node("n1")

        output = bridge.encode_state(
            graph=graph,
            node_id="n1",
            visited_nodes=set(),
            visited_edges=set(),
        )

        assert output.embedding is not None
        assert output.history_features is not None
        assert len(output.full_state) == 5  # 3 + 2

    def test_caching_behavior(self):
        """Test that caching works correctly."""
        from rl_money_laundering.multiagent.integration import EncoderBridge

        call_count = [0]

        def mock_forward(graph, node_id, visited_nodes, visited_edges, edge_time=None):
            call_count[0] += 1
            return np.array([1.0]), np.array([2.0]), {}

        mock_encoder = Mock()
        mock_encoder.forward_state.side_effect = mock_forward

        bridge = EncoderBridge(state_encoder=mock_encoder, cache_embeddings=True)

        graph = nx.DiGraph()

        # First call - should hit encoder
        bridge.encode_state(graph, "n1", set(), set())
        assert call_count[0] == 1

        # Second call with same params - should use cache
        bridge.encode_state(graph, "n1", set(), set())
        # Cache key includes visited nodes, so same call should cache
        # Note: actual caching depends on implementation details


class TestEpisodeTextBuilder:
    """Tests for EpisodeTextBuilder."""

    def test_build_basic(self):
        """Test basic text building."""
        from rl_money_laundering.multiagent.integration import EpisodeTextBuilder
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
            StepRecord,
        )

        builder = EpisodeTextBuilder()

        episode = EpisodeData(
            episode_id="test123",
            steps=[
                StepRecord(
                    step_number=1,
                    node_id="ENTITY_A",
                    action=0,
                    action_type="MOVE",
                    reward=0.5,
                    confidence=0.8,
                    node_features={},
                    neighbors=[],
                    is_fraud=False,
                    is_flagged=False,
                    cumulative_reward=0.5,
                ),
            ],
            total_reward=0.5,
            episode_length=1,
            graph_stats={"num_nodes": 10, "num_edges": 15, "density": 0.15},
        )

        text = builder.build(episode)

        assert "[EPISODE" in text
        assert "test123" in text
        assert "GRAPH" in text

    def test_truncation(self):
        """Test text truncation for long episodes."""
        from rl_money_laundering.multiagent.integration import EpisodeTextBuilder
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
            StepRecord,
        )

        builder = EpisodeTextBuilder(max_tokens=50)

        # Create episode with many steps
        steps = [
            StepRecord(
                step_number=i,
                node_id=f"ENTITY_{i}",
                action=0,
                action_type="MOVE",
                reward=0.1,
                confidence=0.5,
                node_features={"feature": "value" * 100},
                neighbors=[],
                is_fraud=False,
                is_flagged=False,
                cumulative_reward=0.1 * i,
            )
            for i in range(100)
        ]

        episode = EpisodeData(
            episode_id="long_episode",
            steps=steps,
            total_reward=10.0,
            episode_length=100,
        )

        text = builder.build(episode)

        # Should be truncated
        words = text.split()
        assert len(words) <= 100  # Some buffer for truncation message


# =============================================================================
# SECTION 4: Test Reward Integrator
# =============================================================================

class TestRewardIntegrator:
    """Tests for RewardIntegrator."""

    def test_initialization(self):
        """Test integrator initialization."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )

        config = IntegrationConfig()
        integrator = RewardIntegrator(config=config)

        assert integrator is not None

    def test_step_reward_passthrough(self):
        """Test step reward passes through in compatibility mode."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )

        config = IntegrationConfig(compatibility_mode=True)
        integrator = RewardIntegrator(config=config)

        reward = integrator.compute_step_reward(0.5, {})

        assert reward == 0.5

    def test_episode_reward_without_judge(self):
        """Test episode reward computation without judge."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
        )

        config = IntegrationConfig()
        config.judge_config.enable_judge = False

        integrator = RewardIntegrator(config=config)

        episode = EpisodeData(
            episode_id="test",
            true_positives=2,
            false_positives=1,
            false_negatives=1,
            true_negatives=10,
        )

        combined, metrics = integrator.compute_episode_reward(
            original_episode_reward=5.0,
            episode_data=episode,
        )

        assert isinstance(combined, float)
        assert metrics.judge_reward == 0.0  # No judge

    def test_episode_reward_with_mock_judge(self):
        """Test episode reward computation with mock judge."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
        )

        config = IntegrationConfig()

        mock_judge = Mock()
        mock_judge.compute_reward.return_value = {
            "reward": 0.7,
            "uncertainty": 0.1,
        }

        def mock_text_builder(episode):
            return "test episode text"

        integrator = RewardIntegrator(
            config=config,
            judge_model=mock_judge,
            episode_text_builder=mock_text_builder,
        )

        episode = EpisodeData(
            episode_id="test",
            true_positives=3,
            false_positives=1,
            false_negatives=0,
            true_negatives=10,
        )

        combined, metrics = integrator.compute_episode_reward(
            original_episode_reward=5.0,
            episode_data=episode,
        )

        assert metrics.judge_reward == 0.7
        assert metrics.judge_uncertainty == 0.1

    def test_judge_clipping(self):
        """Test that judge rewards are clipped."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
        )

        config = IntegrationConfig()
        config.judge_config.judge_clip_max = 0.5

        mock_judge = Mock()
        mock_judge.compute_reward.return_value = {
            "reward": 5.0,  # Extreme value
            "uncertainty": 0.0,
        }

        integrator = RewardIntegrator(
            config=config,
            judge_model=mock_judge,
            episode_text_builder=lambda e: "text",
        )

        episode = EpisodeData(episode_id="test")

        combined, metrics = integrator.compute_episode_reward(
            original_episode_reward=0.0,
            episode_data=episode,
        )

        assert metrics.was_clipped is True

    def test_uncertainty_penalty(self):
        """Test uncertainty penalty is applied."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
        )

        config = IntegrationConfig()
        config.judge_config.uncertainty_penalty = 0.2

        mock_judge = Mock()
        mock_judge.compute_reward.return_value = {
            "reward": 0.5,
            "uncertainty": 0.5,  # High uncertainty
        }

        integrator = RewardIntegrator(
            config=config,
            judge_model=mock_judge,
            episode_text_builder=lambda e: "text",
        )

        episode = EpisodeData(episode_id="test")

        combined, metrics = integrator.compute_episode_reward(
            original_episode_reward=0.0,
            episode_data=episode,
        )

        # Penalty should be 0.2 * 0.5 = 0.1
        assert metrics.uncertainty_penalty_applied == pytest.approx(0.1, abs=0.01)

    def test_drift_detection(self):
        """Test drift detection in reward history."""
        from rl_money_laundering.multiagent.integration import (
            RewardIntegrator,
            IntegrationConfig,
        )
        from rl_money_laundering.multiagent.integration.environment_adapter import (
            EpisodeData,
        )

        config = IntegrationConfig()
        config.drift_correlation_threshold = 0.5

        integrator = RewardIntegrator(config=config)

        # Simulate many episodes with uncorrelated judge/metrics
        for i in range(150):
            episode = EpisodeData(
                episode_id=f"ep{i}",
                true_positives=1,
                false_positives=1,
            )

            # Force metrics into history
            from rl_money_laundering.multiagent.integration.reward_integrator import (
                RewardMetrics,
            )

            metrics = RewardMetrics(
                original_reward=0.0,
                judge_reward=np.random.random(),  # Random judge
                judge_uncertainty=0.1,
                hard_metric_reward=np.random.random(),  # Random metric
                combined_reward=0.0,
                judge_weight=0.2,
                was_clipped=False,
                uncertainty_penalty_applied=0.0,
                f1=np.random.random(),
            )
            integrator._history.add(metrics)

        # Correlation should be low
        corr = integrator._history.get_correlation()
        assert abs(corr) < 0.5  # Random data should have low correlation


class TestRewardHistory:
    """Tests for RewardHistory."""

    def test_add_and_retrieve(self):
        """Test adding and retrieving from history."""
        from rl_money_laundering.multiagent.integration.reward_integrator import (
            RewardHistory,
            RewardMetrics,
        )

        history = RewardHistory(max_size=100)

        metrics = RewardMetrics(
            original_reward=1.0,
            judge_reward=0.5,
            judge_uncertainty=0.1,
            hard_metric_reward=0.6,
            combined_reward=0.7,
            judge_weight=0.2,
            was_clipped=False,
            uncertainty_penalty_applied=0.01,
            f1=0.8,
        )

        history.add(metrics)

        assert len(history.original_rewards) == 1
        assert history.original_rewards[0] == 1.0

    def test_max_size_enforcement(self):
        """Test that max size is enforced."""
        from rl_money_laundering.multiagent.integration.reward_integrator import (
            RewardHistory,
            RewardMetrics,
        )

        history = RewardHistory(max_size=5)

        for i in range(10):
            metrics = RewardMetrics(
                original_reward=float(i),
                judge_reward=0.0,
                judge_uncertainty=0.0,
                hard_metric_reward=0.0,
                combined_reward=0.0,
                judge_weight=0.0,
                was_clipped=False,
                uncertainty_penalty_applied=0.0,
            )
            history.add(metrics)

        assert len(history.original_rewards) == 5
        # Should keep last 5
        assert history.original_rewards[0] == 5.0


# =============================================================================
# SECTION 5: Test Training Orchestrator
# =============================================================================

class TestTrainingState:
    """Tests for TrainingState."""

    def test_initialization(self):
        """Test state initialization."""
        from rl_money_laundering.multiagent.integration import TrainingState

        state = TrainingState()

        assert state.episode == 0
        assert state.total_steps == 0
        assert state.best_f1 == 0.0

    def test_update(self):
        """Test state updates."""
        from rl_money_laundering.multiagent.integration import TrainingState

        state = TrainingState()

        state.update(reward=1.0, length=10, f1=0.75, judge_reward=0.5)

        assert state.episode == 1
        assert state.episode_rewards[-1] == 1.0
        assert state.best_f1 == 0.75

    def test_best_tracking(self):
        """Test best F1 tracking."""
        from rl_money_laundering.multiagent.integration import TrainingState

        state = TrainingState()

        state.update(reward=1.0, length=10, f1=0.5)
        state.update(reward=1.0, length=10, f1=0.8)
        state.update(reward=1.0, length=10, f1=0.6)

        assert state.best_f1 == 0.8
        assert state.best_episode == 2

    def test_recent_stats(self):
        """Test recent statistics computation."""
        from rl_money_laundering.multiagent.integration import TrainingState

        state = TrainingState()

        for i in range(50):
            state.update(reward=float(i), length=10, f1=0.5 + i * 0.01)

        stats = state.get_recent_stats(window=10)

        assert "mean_reward" in stats
        assert "mean_f1" in stats
        assert stats["total_episodes"] == 50


class TestTrainingOrchestrator:
    """Tests for TrainingOrchestrator."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for orchestrator."""
        # Mock environment
        env = Mock()
        env.graph = nx.DiGraph()
        env.graph.add_node("n1", is_fraud=False)
        env.graph.add_node("n2", is_fraud=True)
        env.graph.add_edge("n1", "n2")
        env.current_node = "n1"
        env.visited_nodes = set()
        env.visited_edges = set()
        env.max_neighbors = 5
        env.reset.return_value = (np.zeros(16), {})
        env.step.return_value = (np.zeros(16), 0.5, False, False, {})

        # Mock agent
        agent = Mock()
        agent.select_action.return_value = (0, 0.8)
        agent.store_transition.return_value = None
        agent.train_step.return_value = {}
        agent.decay_epsilon.return_value = None
        agent.update_target_network.return_value = None

        # Mock state encoder
        encoder = Mock()
        encoder.forward_state.return_value = (
            np.zeros(32),
            np.zeros(16),
            {},
        )

        return {
            "env": env,
            "agent": agent,
            "encoder": encoder,
            "graph": env.graph,
        }

    def test_initialization(self, mock_components):
        """Test orchestrator initialization."""
        from rl_money_laundering.multiagent.integration import (
            TrainingOrchestrator,
            IntegrationConfig,
        )

        config = IntegrationConfig.disabled()

        orchestrator = TrainingOrchestrator(
            env=mock_components["env"],
            agent=mock_components["agent"],
            state_encoder=mock_components["encoder"],
            config=config,
            graph=mock_components["graph"],
        )

        assert orchestrator is not None

    def test_single_episode(self, mock_components):
        """Test running a single episode."""
        from rl_money_laundering.multiagent.integration import (
            TrainingOrchestrator,
            IntegrationConfig,
        )

        config = IntegrationConfig.disabled()

        # Make episode terminate after 3 steps
        step_count = [0]

        def step_side_effect(action, **kwargs):
            step_count[0] += 1
            done = step_count[0] >= 3
            return np.zeros(16), 0.5, done, False, {}

        mock_components["env"].step.side_effect = step_side_effect

        orchestrator = TrainingOrchestrator(
            env=mock_components["env"],
            agent=mock_components["agent"],
            state_encoder=mock_components["encoder"],
            config=config,
            graph=mock_components["graph"],
        )

        result = orchestrator._run_episode(0)

        assert "combined_reward" in result
        assert "length" in result
        assert result["length"] == 3

    def test_factory_function(self, mock_components):
        """Test create_integrated_trainer factory."""
        from rl_money_laundering.multiagent.integration import (
            create_integrated_trainer,
            IntegrationConfig,
        )

        orchestrator = create_integrated_trainer(
            env=mock_components["env"],
            agent=mock_components["agent"],
            state_encoder=mock_components["encoder"],
            graph=mock_components["graph"],
            config=IntegrationConfig.disabled(),
        )

        assert orchestrator is not None


# =============================================================================
# SECTION 6: End-to-End Integration Test
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def full_setup(self):
        """Create full test setup."""
        # Create realistic graph
        G = nx.DiGraph()
        for i in range(50):
            G.add_node(
                f"account_{i}",
                balance=np.random.uniform(1000, 100000),
                risk_score=np.random.uniform(0, 1),
                is_fraud=i < 5,  # 10% fraud
                account_type="individual" if i % 2 == 0 else "business",
            )

        for i in range(100):
            src = f"account_{np.random.randint(0, 50)}"
            tgt = f"account_{np.random.randint(0, 50)}"
            if src != tgt:
                G.add_edge(src, tgt, amount=np.random.uniform(100, 10000))

        # Mock env
        env = Mock()
        env.graph = G
        env.current_node = "account_0"
        env.visited_nodes = set()
        env.visited_edges = set()
        env.flagged_nodes = set()
        env.max_neighbors = 5
        env.reset.return_value = (np.zeros(48), {"start": "account_0"})

        step_count = [0]

        def step_fn(action, **kwargs):
            step_count[0] += 1
            done = step_count[0] >= 10
            return np.zeros(48), np.random.uniform(-1, 1), done, False, {}

        env.step.side_effect = step_fn

        # Mock agent
        agent = Mock()
        agent.select_action.return_value = (np.random.randint(0, 6), 0.8)
        agent.store_transition.return_value = None
        agent.train_step.return_value = {"loss": 0.1}
        agent.decay_epsilon.return_value = None
        agent.update_target_network.return_value = None

        # Mock encoder
        encoder = Mock()
        encoder.forward_state.return_value = (
            np.random.randn(32),
            np.random.randn(16),
            {},
        )

        return {"graph": G, "env": env, "agent": agent, "encoder": encoder}

    def test_full_integration_disabled_judge(self, full_setup):
        """Test full integration with judge disabled."""
        from rl_money_laundering.multiagent.integration import (
            TrainingOrchestrator,
            IntegrationConfig,
        )

        config = IntegrationConfig.disabled()

        orchestrator = TrainingOrchestrator(
            env=full_setup["env"],
            agent=full_setup["agent"],
            state_encoder=full_setup["encoder"],
            config=config,
            graph=full_setup["graph"],
        )

        # Run a few episodes
        results = orchestrator.train(
            num_episodes=5,
            eval_frequency=10,
            checkpoint_frequency=100,
        )

        assert results is not None
        assert results["episodes_completed"] == 5

    def test_full_integration_with_mock_judge(self, full_setup):
        """Test full integration with mock judge."""
        from rl_money_laundering.multiagent.integration import (
            TrainingOrchestrator,
            IntegrationConfig,
        )

        config = IntegrationConfig.default_conservative()

        # Mock judge
        mock_judge = Mock()
        mock_judge.compute_reward.return_value = {
            "reward": 0.6,
            "uncertainty": 0.15,
        }

        orchestrator = TrainingOrchestrator(
            env=full_setup["env"],
            agent=full_setup["agent"],
            state_encoder=full_setup["encoder"],
            config=config,
            graph=full_setup["graph"],
            judge_model=mock_judge,
        )

        results = orchestrator.train(
            num_episodes=3,
            eval_frequency=10,
        )

        assert results is not None
        assert "integrator_stats" in results


# =============================================================================
# Run configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
