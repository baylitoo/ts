"""
Standalone tests for multiagent module components.

Tests the core multiagent components:
- JudgeModel and JudgeConfig
- EpisodeSerializer and GraphTextFusion
- ConservativeRewardComputer and RewardConfig
- MultiAgentAMLEnvironment
- Training callbacks

Run directly: python scripts/test_multiagent_components.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np


def test_judge_config():
    """Test JudgeConfig initialization and validation."""
    print("\n[TEST] JudgeConfig initialization...")

    from rl_money_laundering.multiagent.judge_model import JudgeConfig

    # Default config
    config = JudgeConfig()
    assert config.model_name == "answerdotai/ModernBERT-base"
    assert config.max_length == 512  # Default is 512
    assert config.hidden_dim == 256  # Default is 256
    print("  OK Default config created")
    print(f"     model_name={config.model_name}")
    print(f"     max_length={config.max_length}")
    print(f"     hidden_dim={config.hidden_dim}")

    # Custom config
    custom = JudgeConfig(
        model_name="bert-base-uncased",
        max_length=1024,
        hidden_dim=512,
    )
    assert custom.model_name == "bert-base-uncased"
    assert custom.max_length == 1024
    print("  OK Custom config created")

    # Check fraud type labels
    assert len(config.fraud_type_labels) == 5
    assert "layering" in config.fraud_type_labels
    print(f"  OK Fraud type labels: {config.fraud_type_labels}")

    print("OK JudgeConfig tests passed")
    return True


def test_episode_serializer():
    """Test EpisodeSerializer for episode-to-text conversion."""
    print("\n[TEST] EpisodeSerializer...")

    from rl_money_laundering.multiagent.judge_model import EpisodeSerializer

    # Create serializer with default params
    serializer = EpisodeSerializer(
        quantize_precision=2,
        max_nodes=50,
        max_edges=100,
        include_temporal=True,
        anonymize_ids=True,
    )
    print("  OK EpisodeSerializer created")

    # Create mock episode data matching expected format
    episode_data = {
        "metadata": {
            "episode_length": 10,
            "unique_nodes": 5,
            "flags_used": 2,
        },
        "visited_nodes": ["node_1", "node_2", "node_3"],
        "visited_edges": [("node_1", "node_2"), ("node_2", "node_3")],
        "actions": [0, 0, 1, 0, 1],
        "node_features": {
            "node_1": {"account_type": "individual", "risk_score": 0.3, "balance": 1000},
            "node_2": {"account_type": "business", "risk_score": 0.7, "balance": 50000},
            "node_3": {"account_type": "shell_company", "risk_score": 0.9, "balance": 100000},
        },
        "edge_features": {
            ("node_1", "node_2"): {"amount": 5000, "timestamp": 100},
            ("node_2", "node_3"): {"amount": 45000, "timestamp": 101},
        },
    }

    # Serialize
    text = serializer.serialize(episode_data)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "[META]" in text
    assert "[NODES]" in text
    assert "[EDGES]" in text
    assert "[ACTIONS]" in text
    print(f"  OK Serialized episode to {len(text)} chars")
    print(f"  Preview: {text[:300]}...")

    # Test with anonymize_ids=False
    non_anon_serializer = EpisodeSerializer(anonymize_ids=False)
    non_anon_text = non_anon_serializer.serialize(episode_data)
    assert "node_1" in non_anon_text or "n:node_1" in non_anon_text
    print("  OK Non-anonymized serializer works")

    print("OK EpisodeSerializer tests passed")
    return True


def test_graph_text_fusion():
    """Test GraphTextFusion for combining graph features with text."""
    print("\n[TEST] GraphTextFusion...")

    from rl_money_laundering.multiagent.judge_model import GraphTextFusion

    # Create fusion module with correct parameter names
    fusion = GraphTextFusion(
        graph_dim=64,  # Not graph_feature_dim
        text_dim=768,  # Not text_feature_dim
        fused_dim=256,  # Not output_dim
        num_heads=4,
        dropout=0.1,
    )
    print("  OK GraphTextFusion created")

    # Test forward pass with mock data
    import torch

    batch_size = 4
    graph_features = torch.randn(batch_size, 64)
    text_features = torch.randn(batch_size, 768)

    output = fusion(graph_features, text_features)
    assert output.shape == (batch_size, 256)
    print(f"  OK Forward pass: ({batch_size}, 64) + ({batch_size}, 768) -> {output.shape}")

    # Test with different sizes
    fusion2 = GraphTextFusion(
        graph_dim=128,
        text_dim=512,
        fused_dim=128,
        num_heads=8,
    )
    g = torch.randn(2, 128)
    t = torch.randn(2, 512)
    out2 = fusion2(g, t)
    assert out2.shape == (2, 128)
    print("  OK Different dimensions work")

    # Test with 3D graph input (batch, num_nodes, dim)
    g3d = torch.randn(2, 5, 128)  # 2 batch, 5 nodes, 128 dim
    t2 = torch.randn(2, 512)
    out3 = fusion2(g3d, t2)
    assert out3.shape == (2, 128)
    print("  OK 3D graph input works")

    print("OK GraphTextFusion tests passed")
    return True


def test_reward_config():
    """Test RewardConfig initialization."""
    print("\n[TEST] RewardConfig...")

    from rl_money_laundering.multiagent.reward_computation import RewardConfig

    # Default config
    config = RewardConfig()
    assert config.precision_weight > 0
    assert config.recall_weight > 0
    assert config.f1_weight > 0
    assert config.hard_metric_floor > 0
    print("  OK Default config created")
    print(f"     precision_weight={config.precision_weight}")
    print(f"     recall_weight={config.recall_weight}")
    print(f"     f1_weight={config.f1_weight}")

    # Test weight computation (get_judge_weight method)
    weight_0 = config.get_judge_weight(episode=0)
    weight_500 = config.get_judge_weight(episode=500)
    weight_1000 = config.get_judge_weight(episode=1000)
    weight_2000 = config.get_judge_weight(episode=2000)

    assert weight_0 == config.judge_weight_start
    assert weight_1000 == config.judge_weight_end  # At warmup end
    assert weight_2000 == config.judge_weight_end  # After warmup
    print(f"  OK Weight annealing: ep0={weight_0:.3f}, ep500={weight_500:.3f}, ep1000={weight_1000:.3f}")

    print("OK RewardConfig tests passed")
    return True


def test_conservative_reward_computer():
    """Test ConservativeRewardComputer reward combination."""
    print("\n[TEST] ConservativeRewardComputer...")

    from rl_money_laundering.multiagent.reward_computation import (
        ConservativeRewardComputer,
        RewardConfig,
    )

    # Use config with flag_cooldown_steps=0 to allow immediate flagging in test
    config = RewardConfig(flag_cooldown_steps=0)
    computer = ConservativeRewardComputer(config, judge_fn=None)
    print("  OK ConservativeRewardComputer created (no judge)")

    # Reset episode state
    computer.reset_episode()
    print("  OK Episode reset")

    # Test step reward computation (move action)
    step_reward, step_meta = computer.compute_step_reward(
        action=0,
        is_flag_action=False,
        current_node="node_1",
        is_true_positive=None,
        node_features={"risk_score": 0.5},
        visited_nodes=set(),
    )
    assert isinstance(step_reward, float)
    print(f"  OK Step reward (move): {step_reward:.3f}")

    # Test flag action (true positive)
    flag_reward, flag_meta = computer.compute_step_reward(
        action=5,  # FLAG action
        is_flag_action=True,
        current_node="node_2",
        is_true_positive=True,
        node_features={},
        visited_nodes={"node_1"},
    )
    assert flag_reward > 0, f"True positive should give positive reward, got {flag_reward}"
    print(f"  OK Step reward (flag TP): {flag_reward:.3f}")

    # Test episode reward computation
    episode_data = {
        "flagged_nodes": ["node_2", "node_3"],
        "visited_nodes": ["node_1", "node_2", "node_3", "node_4"],
        "actions": [0, 0, 1, 0, 1],
    }
    true_labels = {
        "node_1": False,
        "node_2": True,
        "node_3": False,
        "node_4": True,
    }

    total_reward, metrics = computer.compute_episode_reward(
        episode_data=episode_data,
        true_labels=true_labels,
    )
    assert isinstance(total_reward, float)
    assert "components" in metrics
    print(f"  OK Episode reward: {total_reward:.3f}")
    print(f"     Hard metric components: {metrics['components'].get('hard_metric', {})}")

    # Test drift alarm
    alarm = computer.check_drift_alarm()
    assert isinstance(alarm, bool)
    print(f"  OK Drift alarm check: {alarm}")

    # Test statistics
    stats = computer.get_statistics()
    assert "episode_count" in stats
    print(f"  OK Statistics: episode_count={stats['episode_count']}")

    print("OK ConservativeRewardComputer tests passed")
    return True


def test_detector_adversary_config():
    """Test DetectorAdversaryConfig for multi-agent setup."""
    print("\n[TEST] DetectorAdversaryConfig...")

    from rl_money_laundering.multiagent.multi_agent_env import DetectorAdversaryConfig

    config = DetectorAdversaryConfig()
    # Check correct attribute names
    assert config.detector_fraud_reward > 0
    assert config.detector_false_alarm_penalty < 0
    assert config.adversary_budget > 0
    print("  OK Default config created")
    print(f"     detector_fraud_reward={config.detector_fraud_reward}")
    print(f"     detector_false_alarm_penalty={config.detector_false_alarm_penalty}")
    print(f"     adversary_budget={config.adversary_budget}")

    # Custom config
    custom = DetectorAdversaryConfig(
        max_steps=100,
        max_neighbors=10,
        adversary_budget=10,
        enable_adversary=True,
    )
    assert custom.adversary_budget == 10
    assert custom.enable_adversary
    print("  OK Custom config created")

    print("OK DetectorAdversaryConfig tests passed")
    return True


def test_multi_agent_env_creation():
    """Test MultiAgentAMLEnvironment creation and basic methods."""
    print("\n[TEST] MultiAgentAMLEnvironment structure...")

    from rl_money_laundering.multiagent.multi_agent_env import (
        MultiAgentAMLEnvironment,
        DetectorAdversaryConfig,
    )

    # Test that the class has expected methods
    assert hasattr(MultiAgentAMLEnvironment, "reset")
    assert hasattr(MultiAgentAMLEnvironment, "step")
    assert hasattr(MultiAgentAMLEnvironment, "get_episode_data")
    assert hasattr(MultiAgentAMLEnvironment, "get_true_labels")
    print("  OK MultiAgentAMLEnvironment has required methods")

    # Create a simple test graph
    import networkx as nx

    G = nx.DiGraph()
    # Add nodes with features
    for i in range(10):
        G.add_node(
            f"node_{i}",
            account_type="individual",
            balance=1000.0 * i,
            risk_score=0.1 * i,
        )
    # Add edges with features
    for i in range(9):
        G.add_edge(
            f"node_{i}",
            f"node_{i+1}",
            amount=500.0 * (i + 1),
            step=i,
            is_fraud=(i % 3 == 0),  # Some fraud edges
        )

    # Create environment
    config = DetectorAdversaryConfig(max_steps=20, max_neighbors=5)
    env = MultiAgentAMLEnvironment(graph=G, config=config)
    print("  OK Environment created with test graph")

    # Test reset
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    print(f"  OK Reset works, obs shape: {obs.shape}")

    # Test step
    action = 0 if len(env.current_neighbors) > 0 else env.FLAG_ACTION
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    print(f"  OK Step works, reward: {reward:.3f}")

    # Test episode data
    episode_data = env.get_episode_data()
    assert "visited_nodes" in episode_data
    assert "actions" in episode_data
    print(f"  OK Episode data has {len(episode_data['visited_nodes'])} visited nodes")

    # Test true labels
    true_labels = env.get_true_labels()
    assert isinstance(true_labels, dict)
    print(f"  OK True labels retrieved: {len(true_labels)} nodes")

    print("OK MultiAgentAMLEnvironment tests passed")
    return True


def test_two_timescale_callback():
    """Test TwoTimescaleJudgeCallback structure."""
    print("\n[TEST] TwoTimescaleJudgeCallback...")

    from rl_money_laundering.multiagent.training_callbacks import (
        TwoTimescaleJudgeCallback,
        TwoTimescaleConfig,
    )

    # Test config
    config = TwoTimescaleConfig()
    assert config.judge_update_frequency > 0
    assert config.judge_warmup_episodes > 0
    print("  OK TwoTimescaleConfig created")
    print(f"     update_frequency={config.judge_update_frequency}")
    print(f"     warmup_episodes={config.judge_warmup_episodes}")

    # Create callback without judge (dry run mode)
    callback = TwoTimescaleJudgeCallback(
        judge_model=None,
        config=config,
        episode_buffer_capacity=100,
    )
    assert callback.config.judge_update_frequency == config.judge_update_frequency
    print("  OK Callback created in dry-run mode")

    # Test _should_update_judge logic
    # Before warmup - should not update
    should_update_early = callback._should_update_judge(episode=50)
    assert not should_update_early
    print("  OK Should not update before warmup")

    # After warmup at update frequency - should update (if buffer has enough)
    # Add some dummy episodes to buffer
    for i in range(60):
        callback._add_to_buffer(
            episode_data={"visited_nodes": ["a", "b"], "flagged_nodes": [], "actions": [0, 1]},
            true_labels={"a": False, "b": True},
        )

    should_update_at_freq = callback._should_update_judge(
        episode=config.judge_warmup_episodes + config.judge_update_frequency
    )
    # Note: still False because judge_model is None
    print(f"  OK Update check at frequency: {should_update_at_freq}")

    # Test statistics
    stats = callback.get_statistics()
    assert "total_episodes" in stats
    assert "buffer_size" in stats
    print(f"  OK Statistics: buffer_size={stats['buffer_size']}")

    print("OK TwoTimescaleJudgeCallback tests passed")
    return True


def test_safety_monitor_callback():
    """Test SafetyMonitorCallback for drift detection."""
    print("\n[TEST] SafetyMonitorCallback...")

    from rl_money_laundering.multiagent.training_callbacks import (
        SafetyMonitorCallback,
        SafetyConfig,
    )

    # Test config
    config = SafetyConfig(
        hard_metric_floor=0.1,
        drift_correlation_threshold=-0.3,
        max_consecutive_alarms=3,
    )
    print("  OK SafetyConfig created")

    # Create callback
    callback = SafetyMonitorCallback(config=config)
    assert callback.config.hard_metric_floor == 0.1
    print("  OK Callback created with thresholds")

    # Test on_episode_end
    result = callback.on_episode_end(
        episode=1,
        hard_metric=0.5,
        judge_score=0.4,
        combined_reward=0.45,
    )
    assert "alarms" in result
    assert not result["should_stop"]
    print("  OK Episode end check passed (no alarms)")

    # Test with floor violation
    result_low = callback.on_episode_end(
        episode=2,
        hard_metric=0.05,  # Below floor
        judge_score=0.4,
        combined_reward=0.25,
    )
    assert len(result_low["alarms"]) > 0
    print(f"  OK Floor violation detected: {result_low['alarms'][0]['type']}")

    # Test statistics
    stats = callback.get_statistics()
    assert "total_alarms" in stats
    print(f"  OK Statistics: total_alarms={stats['total_alarms']}")

    print("OK SafetyMonitorCallback tests passed")
    return True


def test_judge_metrics_callback():
    """Test JudgeMetricsCallback for logging."""
    print("\n[TEST] JudgeMetricsCallback...")

    from rl_money_laundering.multiagent.training_callbacks import JudgeMetricsCallback

    # Create callback
    callback = JudgeMetricsCallback(log_frequency=10, detailed_logging=False)
    assert callback.log_frequency == 10
    print("  OK Callback created")

    # Test on_judge_evaluation
    callback.on_judge_evaluation(
        episode=1,
        reward=0.5,
        metadata={"uncertainty": 0.2, "fraud_type": "layering"},
        latency_ms=15.0,
    )
    callback.on_judge_evaluation(
        episode=2,
        reward=0.7,
        metadata={"uncertainty": 0.1, "fraud_type": "structuring"},
        latency_ms=12.0,
    )
    print("  OK Judge evaluations recorded")

    # Test statistics
    stats = callback.get_statistics()
    assert "reward_mean" in stats
    assert "fraud_type_distribution" in stats
    print(f"  OK Statistics: reward_mean={stats['reward_mean']:.3f}")
    print(f"     Fraud type distribution: {stats['fraud_type_distribution']}")

    # Test report
    report = callback.get_report()
    assert "JUDGE METRICS REPORT" in report
    print("  OK Report generated")

    print("OK JudgeMetricsCallback tests passed")
    return True


def test_module_imports():
    """Test that all module exports are accessible."""
    print("\n[TEST] Module imports...")

    from rl_money_laundering.multiagent import (
        JudgeModel,
        JudgeConfig,
        EpisodeSerializer,
        GraphTextFusion,
        ConservativeRewardComputer,
        RewardConfig,
        MultiAgentAMLEnvironment,
        DetectorAdversaryConfig,
        TwoTimescaleJudgeCallback,
        SafetyMonitorCallback,
        JudgeMetricsCallback,
    )

    # All imports should work
    assert JudgeModel is not None
    assert JudgeConfig is not None
    assert EpisodeSerializer is not None
    assert GraphTextFusion is not None
    assert ConservativeRewardComputer is not None
    assert RewardConfig is not None
    assert MultiAgentAMLEnvironment is not None
    assert DetectorAdversaryConfig is not None
    assert TwoTimescaleJudgeCallback is not None
    assert SafetyMonitorCallback is not None
    assert JudgeMetricsCallback is not None
    print("  OK All 11 exports imported successfully")

    print("OK Module imports test passed")
    return True


def main():
    """Run all multiagent component tests."""
    print("=" * 70)
    print("MULTIAGENT COMPONENT TESTS")
    print("=" * 70)
    print("\nTesting core multiagent module components...")

    tests = [
        ("Module Imports", test_module_imports),
        ("JudgeConfig", test_judge_config),
        ("EpisodeSerializer", test_episode_serializer),
        ("GraphTextFusion", test_graph_text_fusion),
        ("RewardConfig", test_reward_config),
        ("ConservativeRewardComputer", test_conservative_reward_computer),
        ("DetectorAdversaryConfig", test_detector_adversary_config),
        ("MultiAgentAMLEnvironment", test_multi_agent_env_creation),
        ("TwoTimescaleJudgeCallback", test_two_timescale_callback),
        ("SafetyMonitorCallback", test_safety_monitor_callback),
        ("JudgeMetricsCallback", test_judge_metrics_callback),
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
        print("\nALL MULTIAGENT COMPONENT TESTS PASSED!")
        return 0
    else:
        print(f"\n{failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
