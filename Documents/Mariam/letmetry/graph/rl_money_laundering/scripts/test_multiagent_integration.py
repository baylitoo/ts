"""
Standalone tests for multiagent integration layer.

Tests the integration components that connect multiagent module
with the existing RL framework:
- IntegrationConfig and JudgeIntegrationConfig
- EnvironmentAdapter and EpisodeCollector
- EncoderBridge and EmbeddingCache
- RewardIntegrator and RewardMetrics
- TrainingOrchestrator

Run directly: python scripts/test_multiagent_integration.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np


def test_integration_config():
    """Test IntegrationConfig and presets."""
    print("\n[TEST] IntegrationConfig...")

    from rl_money_laundering.multiagent.integration import IntegrationConfig

    # Default config
    config = IntegrationConfig()
    assert config.hard_metric_weight > 0
    assert config.judge_config is not None
    print("  OK Default config created")

    # Test presets
    conservative = IntegrationConfig.default_conservative()
    assert conservative.hard_metric_weight >= 0.5
    assert conservative.safety_monitoring
    print(f"  OK Conservative preset: hard_metric_weight={conservative.hard_metric_weight}")

    aggressive = IntegrationConfig.default_aggressive()
    assert aggressive.hard_metric_weight < conservative.hard_metric_weight
    print(f"  OK Aggressive preset: hard_metric_weight={aggressive.hard_metric_weight}")

    disabled = IntegrationConfig.disabled()
    assert not disabled.judge_config.enable_judge
    print("  OK Disabled preset: judge disabled")

    # Test to_dict
    config_dict = config.to_dict()
    assert "hard_metric_weight" in config_dict
    assert "judge_config" in config_dict
    print("  OK Config serializes to dict")

    print("OK IntegrationConfig tests passed")
    return True


def test_judge_integration_config():
    """Test JudgeIntegrationConfig with weight annealing."""
    print("\n[TEST] JudgeIntegrationConfig...")

    from rl_money_laundering.multiagent.integration import JudgeIntegrationConfig

    # Use correct parameter names: judge_weight_start, judge_weight_end
    config = JudgeIntegrationConfig(
        enable_judge=True,
        judge_weight_start=0.0,
        judge_weight_end=0.3,
        judge_warmup_episodes=500,
        uncertainty_penalty=0.5,
        judge_clip_max=1.0,
    )
    print("  OK JudgeIntegrationConfig created")

    # Test weight annealing
    weight_0 = config.get_judge_weight(episode=0)
    weight_250 = config.get_judge_weight(episode=250)
    weight_500 = config.get_judge_weight(episode=500)
    weight_1000 = config.get_judge_weight(episode=1000)

    assert weight_0 == 0.0  # Initial
    assert 0.0 < weight_250 < 0.3  # During warmup
    assert weight_500 == 0.3  # At max after warmup
    assert weight_1000 == 0.3  # Stays at max

    print(f"  OK Weight annealing:")
    print(f"     ep0={weight_0:.3f}, ep250={weight_250:.3f}, ep500={weight_500:.3f}, ep1000={weight_1000:.3f}")

    # Test disabled judge
    disabled = JudgeIntegrationConfig(enable_judge=False)
    assert disabled.get_judge_weight(episode=1000) == 0.0
    print("  OK Disabled judge returns zero weight")

    print("OK JudgeIntegrationConfig tests passed")
    return True


def test_step_record():
    """Test StepRecord dataclass."""
    print("\n[TEST] StepRecord...")

    # Import directly from submodule since it's not exported in __init__.py
    from rl_money_laundering.multiagent.integration.environment_adapter import StepRecord

    step = StepRecord(
        step_number=0,
        node_id="tx_123",
        action=1,
        action_type="FLAG",
        reward=1.0,
        confidence=0.85,
        is_fraud=True,
        is_flagged=True,
        node_features={"amount": 1500.0, "degree": 3},
        neighbors=["tx_124", "tx_125"],
        cumulative_reward=1.0,
    )

    assert step.step_number == 0
    assert step.node_id == "tx_123"
    assert step.action_type == "FLAG"
    assert step.is_fraud
    print("  OK StepRecord created with all fields")

    print("OK StepRecord tests passed")
    return True


def test_episode_data():
    """Test EpisodeData dataclass."""
    print("\n[TEST] EpisodeData...")

    from rl_money_laundering.multiagent.integration.environment_adapter import EpisodeData

    episode = EpisodeData(
        episode_id="ep_001",
        start_node="tx_start",
        flagged_nodes=["tx_1", "tx_3"],
        visited_nodes=["tx_1", "tx_2", "tx_3", "tx_4"],
        fraud_nodes_encountered=["tx_1", "tx_4"],
        true_positives=1,  # tx_1 flagged and is fraud
        false_positives=1,  # tx_3 flagged but not fraud
        false_negatives=1,  # tx_4 not flagged but is fraud
        total_reward=0.0,
    )
    print("  OK EpisodeData created")

    # Check computed metrics
    assert episode.true_positives == 1
    assert episode.false_positives == 1
    assert episode.false_negatives == 1
    print(f"  OK Metrics: TP={episode.true_positives}, FP={episode.false_positives}, FN={episode.false_negatives}")

    # Check precision/recall/F1 properties
    assert episode.precision == 0.5  # 1/2
    assert episode.recall == 0.5  # 1/(1+1)
    assert 0.49 < episode.f1_score < 0.51  # ~0.5
    print(f"  OK P={episode.precision:.2f}, R={episode.recall:.2f}, F1={episode.f1_score:.2f}")

    # Test to_dict
    d = episode.to_dict()
    assert "episode_id" in d
    assert "metrics" in d
    print("  OK EpisodeData serializes to dict")

    print("OK EpisodeData tests passed")
    return True


def test_episode_collector():
    """Test EpisodeCollector for buffering episodes."""
    print("\n[TEST] EpisodeCollector...")

    from rl_money_laundering.multiagent.integration import EpisodeCollector

    collector = EpisodeCollector(buffer_size=5, anonymize_ids=False)
    print("  OK EpisodeCollector created with buffer_size=5")

    # Create a mock environment
    class MockEnv:
        def __init__(self):
            self.current_node = "tx_0"
            self.visited_nodes = {"tx_0"}
            self.flagged_nodes = set()
            self.graph = None  # No real graph needed for this test
            self.max_neighbors = 5
            self.fraud_nodes = {"tx_1"}

    mock_env = MockEnv()

    # Start an episode
    collector.start_episode(
        env=mock_env,
        start_node="tx_0",
        graph_stats={"num_nodes": 100, "num_edges": 200}
    )
    assert collector._current_episode is not None
    print("  OK Episode started")

    # Record some steps
    collector.record_step(
        node_id="tx_1",
        action=5,  # FLAG action (max_neighbors)
        reward=1.0,
        confidence=0.9,
        env=mock_env,
        info={"is_fraud": True},
    )
    collector.record_step(
        node_id="tx_2",
        action=0,  # MOVE
        reward=0.0,
        confidence=0.2,
        env=mock_env,
        info={},
    )
    print("  OK Recorded 2 steps")

    # End episode
    episode = collector.end_episode(env=mock_env, final_info={"done": True})
    assert episode is not None
    assert len(episode.steps) == 2
    print(f"  OK Episode ended: {len(episode.steps)} steps")

    # Check buffer
    assert len(collector._episode_buffer) == 1
    print("  OK Episode added to buffer")

    # Add more episodes to test buffer limit
    for i in range(6):
        collector.start_episode(env=mock_env, start_node=f"node_{i}")
        collector.record_step(
            node_id=f"node_{i}",
            action=5,
            reward=0.5,
            confidence=0.5,
            env=mock_env,
            info={},
        )
        collector.end_episode(env=mock_env)

    assert len(collector._episode_buffer) == 5  # Buffer limit
    print("  OK Buffer limited to max size")

    # Get recent episodes
    recent = collector.get_recent_episodes(n=3)
    assert len(recent) == 3
    print("  OK Retrieved recent episodes")

    print("OK EpisodeCollector tests passed")
    return True


def test_episode_collector_anonymization():
    """Test EpisodeCollector ID anonymization."""
    print("\n[TEST] EpisodeCollector anonymization...")

    from rl_money_laundering.multiagent.integration import EpisodeCollector

    collector = EpisodeCollector(anonymize_ids=True)

    # Create a mock environment
    class MockEnv:
        def __init__(self):
            self.current_node = "start"
            self.visited_nodes = set()
            self.flagged_nodes = set()
            self.graph = None
            self.max_neighbors = 5

    mock_env = MockEnv()

    collector.start_episode(env=mock_env, start_node="secret_account_123")
    collector.record_step(
        node_id="secret_account_123",
        action=5,
        reward=1.0,
        confidence=0.8,
        env=mock_env,
        info={},
    )
    episode = collector.end_episode(env=mock_env)

    # Check that IDs are anonymized
    assert episode.steps[0].node_id != "secret_account_123"
    assert "ENTITY_" in episode.steps[0].node_id
    print(f"  OK Node ID anonymized: secret_account_123 -> {episode.steps[0].node_id}")

    print("OK EpisodeCollector anonymization tests passed")
    return True


def test_environment_adapter():
    """Test EnvironmentAdapter wrapping."""
    print("\n[TEST] EnvironmentAdapter...")

    from rl_money_laundering.multiagent.integration import EnvironmentAdapter, EpisodeCollector

    # Create mock environment
    class MockEnv:
        def __init__(self):
            self.current_node = "node_0"
            self.visited_nodes = set()
            self.visited_edges = set()
            self.flagged_nodes = set()
            self._step_count = 0
            self.graph = None
            self.max_neighbors = 5

        def reset(self, **kwargs):
            self.current_node = kwargs.get("start_node", "node_0")
            self.visited_nodes = {self.current_node}
            self._step_count = 0
            return np.array([1.0, 2.0, 3.0]), {"node_id": self.current_node}

        def step(self, action, **kwargs):
            self._step_count += 1
            self.current_node = f"node_{self._step_count}"
            self.visited_nodes.add(self.current_node)
            reward = 1.0 if action == 5 else 0.0
            done = self._step_count >= 5
            return (
                np.array([1.0, 2.0, 3.0]),
                reward,
                done,
                False,
                {"node_id": self.current_node},
            )

    mock_env = MockEnv()
    collector = EpisodeCollector()
    adapter = EnvironmentAdapter(env=mock_env, collector=collector, enable_collection=True)
    print("  OK EnvironmentAdapter wrapping mock env")

    # Test reset
    obs, info = adapter.reset(start_node="start_node")
    assert obs is not None
    print("  OK Reset works")

    # Test step
    obs, reward, term, trunc, info = adapter.step(action=5, confidence=0.9)
    assert reward == 1.0
    print("  OK Step works")

    # Run full episode
    done = False
    while not done:
        obs, reward, term, trunc, info = adapter.step(action=0, confidence=0.5)
        done = term or trunc

    # Get episode data
    episode_data = adapter.get_episode_data()
    assert episode_data is not None
    assert len(episode_data.steps) > 0
    print(f"  OK Episode collected: {len(episode_data.steps)} steps")

    # Test disable/enable collection (use correct attribute name)
    adapter.disable_collection()
    assert not adapter._enable_collection
    adapter.enable_collection()
    assert adapter._enable_collection
    print("  OK Collection toggle works")

    print("OK EnvironmentAdapter tests passed")
    return True


def test_embedding_cache():
    """Test EmbeddingCache LRU behavior."""
    print("\n[TEST] EmbeddingCache...")

    from rl_money_laundering.multiagent.integration import EmbeddingCache

    cache = EmbeddingCache(max_size=3)
    print("  OK EmbeddingCache created with max_size=3")

    # Add items
    cache.put("key1", np.array([1, 2, 3]))
    cache.put("key2", np.array([4, 5, 6]))
    cache.put("key3", np.array([7, 8, 9]))
    assert len(cache._cache) == 3
    print("  OK Added 3 items")

    # Get item (should move to end)
    result = cache.get("key1")
    assert result is not None
    print("  OK Cache hit for key1")

    # Add new item (should evict key2, not key1)
    cache.put("key4", np.array([10, 11, 12]))
    assert cache.get("key2") is None  # Evicted
    assert cache.get("key1") is not None  # Still there (was accessed)
    print("  OK LRU eviction works (key2 evicted, key1 retained)")

    # Check stats
    stats = cache.get_stats()
    assert stats["size"] == 3
    assert stats["hits"] > 0
    assert stats["hit_rate"] > 0
    print(f"  OK Stats: size={stats['size']}, hits={stats['hits']}, hit_rate={stats['hit_rate']:.2f}")

    print("OK EmbeddingCache tests passed")
    return True


def test_encoder_output():
    """Test EncoderOutput dataclass."""
    print("\n[TEST] EncoderOutput...")

    from rl_money_laundering.multiagent.integration.encoder_bridge import EncoderOutput

    output = EncoderOutput(
        embedding=np.array([1, 2, 3, 4]),
        history_features=np.array([5, 6]),
        judge_features=np.array([7, 8, 9]),
        auxiliary_outputs={"attention": np.array([0.1, 0.2])},
    )

    assert output.embedding.shape == (4,)
    assert output.history_features.shape == (2,)
    print("  OK EncoderOutput created")

    # Test full_state concatenation
    full = output.full_state
    assert full.shape == (9,)  # 4 + 2 + 3
    assert np.array_equal(full[:4], output.embedding)
    assert np.array_equal(full[4:6], output.history_features)
    assert np.array_equal(full[6:], output.judge_features)
    print(f"  OK full_state: {output.embedding.shape} + {output.history_features.shape} + {output.judge_features.shape} = {full.shape}")

    # Test without judge features
    output_no_judge = EncoderOutput(
        embedding=np.array([1, 2, 3]),
        history_features=np.array([4, 5]),
    )
    full_no_judge = output_no_judge.full_state
    assert full_no_judge.shape == (5,)  # 3 + 2
    print("  OK full_state without judge features works")

    print("OK EncoderOutput tests passed")
    return True


def test_episode_text_builder():
    """Test EpisodeTextBuilder for judge input."""
    print("\n[TEST] EpisodeTextBuilder...")

    from rl_money_laundering.multiagent.integration.encoder_bridge import EpisodeTextBuilder
    from rl_money_laundering.multiagent.integration.environment_adapter import EpisodeData, StepRecord

    builder = EpisodeTextBuilder(max_tokens=500, include_features=True, include_actions=True)
    print("  OK EpisodeTextBuilder created")

    # Create step records
    steps = [
        StepRecord(
            step_number=0,
            node_id="tx_001",
            action=5,
            action_type="FLAG",
            reward=1.0,
            confidence=0.95,
            is_fraud=True,
            is_flagged=True,
            node_features={"amount": 5000, "degree": 5},
            neighbors=["tx_002"],
            cumulative_reward=1.0,
        ),
        StepRecord(
            step_number=1,
            node_id="tx_002",
            action=0,
            action_type="MOVE",
            reward=0.0,
            confidence=0.2,
            is_fraud=False,
            is_flagged=False,
            node_features={},
            neighbors=[],
            cumulative_reward=1.0,
        ),
    ]

    episode = EpisodeData(
        episode_id="ep_test_001",
        steps=steps,
        flagged_nodes=["tx_001"],
        visited_nodes=["tx_001", "tx_002"],
        fraud_nodes_encountered=["tx_001"],
        true_positives=1,
        false_positives=0,
        total_reward=1.0,
        graph_stats={"num_nodes": 50, "num_edges": 100, "density": 0.04},
    )

    # Build text
    text = builder.build(episode)
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"  OK Built text: {len(text)} chars")

    # Check content
    assert "[EPISODE" in text
    assert "precision=" in text.lower() or "METRICS" in text
    print(f"  Preview:\n{text[:400]}...")

    print("OK EpisodeTextBuilder tests passed")
    return True


def test_reward_metrics():
    """Test RewardMetrics dataclass."""
    print("\n[TEST] RewardMetrics...")

    from rl_money_laundering.multiagent.integration import RewardMetrics

    metrics = RewardMetrics(
        original_reward=10.0,
        judge_reward=0.8,
        judge_uncertainty=0.2,
        hard_metric_reward=0.7,
        combined_reward=0.75,
        judge_weight=0.2,
        was_clipped=False,
        uncertainty_penalty_applied=0.1,
        precision=0.9,
        recall=0.8,
        f1=0.85,
    )

    assert metrics.combined_reward == 0.75
    assert metrics.f1 == 0.85
    print("  OK RewardMetrics created")

    # Test to_dict
    d = metrics.to_dict()
    assert "combined_reward" in d
    assert "judge_weight" in d
    assert d["precision"] == 0.9
    print("  OK RewardMetrics serializes to dict")

    print("OK RewardMetrics tests passed")
    return True


def test_reward_history():
    """Test RewardHistory for drift detection."""
    print("\n[TEST] RewardHistory...")

    from rl_money_laundering.multiagent.integration.reward_integrator import RewardHistory
    from rl_money_laundering.multiagent.integration import RewardMetrics

    history = RewardHistory(max_size=10)
    print("  OK RewardHistory created with max_size=10")

    # Add metrics
    for i in range(15):
        metrics = RewardMetrics(
            original_reward=float(i),
            judge_reward=float(i) * 0.1,
            judge_uncertainty=0.2,
            hard_metric_reward=float(i) * 0.05,
            combined_reward=float(i) * 0.15,
            judge_weight=0.2,
            was_clipped=False,
            uncertainty_penalty_applied=0.04,
            f1=0.5 + i * 0.03,
        )
        history.add(metrics)

    # Check max size
    assert len(history.original_rewards) == 10
    print("  OK History limited to max size")

    # Check correlation
    corr = history.get_correlation(window=10)
    assert -1.0 <= corr <= 1.0
    print(f"  OK Correlation computed: {corr:.3f}")

    # Get stats
    stats = history.get_stats()
    assert "count" in stats
    assert "original_mean" in stats
    assert "correlation" in stats
    print(f"  OK Stats: count={stats['count']}, original_mean={stats['original_mean']:.2f}")

    print("OK RewardHistory tests passed")
    return True


def test_reward_integrator():
    """Test RewardIntegrator reward combination."""
    print("\n[TEST] RewardIntegrator...")

    from rl_money_laundering.multiagent.integration import (
        IntegrationConfig,
        RewardIntegrator,
    )
    from rl_money_laundering.multiagent.integration.environment_adapter import EpisodeData

    config = IntegrationConfig.default_conservative()
    integrator = RewardIntegrator(config=config, judge_model=None)
    print("  OK RewardIntegrator created (no judge)")

    # Create episode data
    episode = EpisodeData(
        episode_id="ep_test",
        flagged_nodes=["tx_1"],
        visited_nodes=["tx_1"],
        fraud_nodes_encountered=["tx_1"],
        true_positives=1,
        false_positives=0,
        false_negatives=0,
        total_reward=1.0,
    )

    # Compute step reward
    step_reward = integrator.compute_step_reward(original_reward=1.0, step_info={})
    assert step_reward == 1.0  # Unchanged without judge
    print(f"  OK Step reward: {step_reward}")

    # Compute episode reward
    combined, metrics = integrator.compute_episode_reward(
        original_episode_reward=5.0,
        episode_data=episode,
    )

    assert isinstance(combined, float)
    assert metrics.f1 == 1.0  # Perfect detection
    print(f"  OK Episode reward: combined={combined:.3f}, F1={metrics.f1:.2f}")

    # Get stats
    stats = integrator.get_stats()
    assert "current_episode" in stats
    assert stats["judge_available"] is False
    print(f"  OK Stats: episode={stats['current_episode']}, judge_available={stats['judge_available']}")

    # Test should_stop_training
    should_stop, reason = integrator.should_stop_training()
    assert not should_stop  # No safety violations yet
    print("  OK Safety check passed")

    print("OK RewardIntegrator tests passed")
    return True


def test_training_state():
    """Test TrainingState tracking."""
    print("\n[TEST] TrainingState...")

    from rl_money_laundering.multiagent.integration import TrainingState

    state = TrainingState()
    assert state.episode == 0
    assert state.best_f1 == 0.0
    print("  OK TrainingState created")

    # Update with results
    state.update(reward=10.0, length=50, f1=0.7, judge_reward=0.5)
    assert state.episode == 1
    assert state.best_f1 == 0.7
    assert state.total_steps == 50
    print("  OK State updated after episode 1")

    state.update(reward=12.0, length=45, f1=0.8, judge_reward=0.6)
    assert state.episode == 2
    assert state.best_f1 == 0.8
    assert state.best_episode == 2
    print("  OK State updated after episode 2 (new best)")

    state.update(reward=8.0, length=40, f1=0.6, judge_reward=0.4)
    assert state.best_f1 == 0.8  # Still the best
    assert state.best_episode == 2
    print("  OK Best F1 retained after worse episode")

    # Get recent stats
    stats = state.get_recent_stats(window=10)
    assert "mean_reward" in stats
    assert "mean_f1" in stats
    print(f"  OK Recent stats: mean_reward={stats['mean_reward']:.2f}, mean_f1={stats['mean_f1']:.2f}")

    # To dict
    d = state.to_dict()
    assert "episode" in d
    assert "best_f1" in d
    print("  OK State serializes to dict")

    print("OK TrainingState tests passed")
    return True


def test_module_imports():
    """Test that all integration exports are accessible."""
    print("\n[TEST] Integration module imports...")

    from rl_money_laundering.multiagent.integration import (
        IntegrationConfig,
        JudgeIntegrationConfig,
        AdversaryConfig,
        EnvironmentAdapter,
        EpisodeCollector,
        EncoderBridge,
        EmbeddingCache,
        RewardIntegrator,
        RewardMetrics,
        TrainingOrchestrator,
        TrainingState,
    )

    # All imports should work
    assert IntegrationConfig is not None
    assert JudgeIntegrationConfig is not None
    assert AdversaryConfig is not None
    assert EnvironmentAdapter is not None
    assert EpisodeCollector is not None
    assert EncoderBridge is not None
    assert EmbeddingCache is not None
    assert RewardIntegrator is not None
    assert RewardMetrics is not None
    assert TrainingOrchestrator is not None
    assert TrainingState is not None
    print("  OK All 11 exports imported successfully")

    print("OK Module imports test passed")
    return True


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("MULTIAGENT INTEGRATION LAYER TESTS")
    print("=" * 70)
    print("\nTesting integration components...")

    tests = [
        ("Module Imports", test_module_imports),
        ("IntegrationConfig", test_integration_config),
        ("JudgeIntegrationConfig", test_judge_integration_config),
        ("StepRecord", test_step_record),
        ("EpisodeData", test_episode_data),
        ("EpisodeCollector", test_episode_collector),
        ("EpisodeCollector Anonymization", test_episode_collector_anonymization),
        ("EnvironmentAdapter", test_environment_adapter),
        ("EmbeddingCache", test_embedding_cache),
        ("EncoderOutput", test_encoder_output),
        ("EpisodeTextBuilder", test_episode_text_builder),
        ("RewardMetrics", test_reward_metrics),
        ("RewardHistory", test_reward_history),
        ("RewardIntegrator", test_reward_integrator),
        ("TrainingState", test_training_state),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\nFAIL {test_name} FAILED:")
            print(f"  {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failed == 0:
        print("\nALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"\n{failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
