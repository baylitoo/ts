"""
Quick test script for Phase 1 RLlib integration.

Tests that all components work together before full training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import networkx as nx
from rl_money_laundering.rllib_integration import GraphSpace, create_rllib_env, register_aml_env


def test_graph_space():
    """Test GraphSpace sampling and validation."""
    print("Testing GraphSpace...")

    space = GraphSpace(
        max_nodes=10,
        max_edges=30,
        node_feature_dim=8,
        context_feature_dim=4
    )

    # Sample random observation
    obs = space.sample()

    print(f"  Node features shape: {obs['node_features'].shape}")
    print(f"  Edge index shape: {obs['edge_index'].shape}")
    print(f"  Num nodes: {obs['num_nodes']}")
    print(f"  Num edges: {obs['num_edges']}")
    print(f"  Context features shape: {obs['context_features'].shape}")

    # Validate
    assert space.contains(obs), "Sampled observation should be valid!"

    print("  ✓ GraphSpace working correctly\n")


def test_env_wrapper():
    """Test environment wrapper with dummy data."""
    print("Testing Environment Wrapper...")

    # Create dummy graph
    G = nx.DiGraph()
    nodes = ["A", "B", "C", "D", "E"]
    for i, node in enumerate(nodes):
        G.add_node(node, feature_vector=np.random.randn(8), is_fraud=i % 2 == 0)

    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
    for u, v in edges:
        G.add_edge(u, v, amount=np.random.rand() * 1000, is_fraud=False)

    print(f"  Created dummy graph: {len(nodes)} nodes, {len(edges)} edges")

    # Create dummy feature extractor
    class DummyFeatureExtractor:
        node_feature_dim = 8

        def extract(self, node_data):
            return node_data.get("feature_vector", np.zeros(8))

    # Create dummy GNN encoder
    class DummyGNNEncoder:
        embedding_dim = 4
        history_dim = 4

    # Configure environment
    env_config = {
        "graph": G,
        "start_node": "A",
        "node_feature_extractor": DummyFeatureExtractor(),
        "gnn_encoder": DummyGNNEncoder(),
        "max_steps": 10,
        "max_nodes": 10,
        "max_edges": 30,
    }

    # Create environment
    try:
        env = create_rllib_env(env_config)
        print(f"  ✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}\n")

        # Reset and check observation
        obs, info = env.reset()
        print(f"  ✓ Reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Observation['num_nodes']: {obs['num_nodes']}")

        # Take a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Step successful (reward={reward:.2f})")
        print()

    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def test_environment_registration():
    """Test registering environment with RLlib."""
    print("Testing Environment Registration...")

    register_aml_env("test_aml_env")
    print("  ✓ Environment registered with RLlib")
    print("  You can now use .environment('test_aml_env') in DQNConfig\n")


def main():
    print("=" * 60)
    print("Phase 1 RLlib Integration Test")
    print("=" * 60)
    print()

    try:
        test_graph_space()
        test_env_wrapper()
        test_environment_registration()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Install Ray: pip install 'ray[rllib]>=2.40.0'")
        print("2. Run training: python scripts/train_rllib.py --node-features ... --edges ... --classes ...")
        print()

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
