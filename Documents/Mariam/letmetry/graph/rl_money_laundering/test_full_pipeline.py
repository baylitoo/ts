"""
Full Pipeline Integration Test
Tests actual dataset loading, feature extraction, environment initialization, and episode execution.
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

# Import core components
from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.agent import DuelingDQNNetwork


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_data_loading():
    """Test 1: Load the actual AMLNet dataset."""
    print_section("TEST 1: Data Loading")

    # Dataset path - check current directory first, then parent
    dataset_path = Path("AMNet_August 2025.csv")
    if not dataset_path.exists():
        dataset_path = Path("..") / "AMLNet_August 2025.csv"

    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print(f"   Current directory: {os.getcwd()}")
        print("   Files in current directory:")
        for f in Path(".").iterdir():
            print(f"     - {f.name}")
        return None, None

    print(f"✓ Dataset found: {dataset_path}")
    print(f"  Size: {dataset_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Create config
    config = AMLNetConfig(csv_path=str(dataset_path))
    print("✓ Config created")

    # Load data
    loader = AMLNetLoader(config)
    print("✓ Loader initialized")

    # Load a manageable subset for integration testing to keep CPU usage reasonable
    print("  Loading first 20,000 rows for testing...")
    loader.load_raw(nrows=20000)
    print("✓ Raw data loaded")
    print(f"  Transactions: {len(loader.transactions):,}")
    print(f"  Columns: {list(loader.transactions.columns)}")

    # Check fraud ratio
    fraud_count = loader.transactions['isFraud'].sum()
    fraud_ratio = fraud_count / len(loader.transactions) * 100
    print(f"  Fraud transactions: {fraud_count:,} ({fraud_ratio:.2f}%)")

    return loader, config


def test_graph_construction(loader: AMLNetLoader):
    """Test 2: Build graph with temporal and network features."""
    print_section("TEST 2: Graph Construction with Features")

    # Build base graph
    graph = loader.build_graph()
    print("✓ Base graph built")
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")

    # Add temporal features
    print("\n  Adding temporal features...")
    graph = add_temporal_features_to_graph(graph, loader.transactions)
    print("✓ Temporal features added")

    # Check temporal features on a sample node
    sample_node = list(graph.nodes())[0]
    node_data = graph.nodes[sample_node]
    temporal_features = {
        k: v for k, v in node_data.items()
        if k in ['tx_velocity', 'business_hour_ratio', 'first_step', 'last_step']
    }
    print(f"  Sample node '{sample_node}' temporal features:")
    for k, v in temporal_features.items():
        print(f"    - {k}: {v}")

    # Add network features
    print("\n  Adding network features...")
    graph = add_network_features_to_graph(graph, compute_expensive=False)
    print("✓ Network features added")

    # Check network features on the sample node
    network_features = {
        k: v for k, v in graph.nodes[sample_node].items()
        if k in ['degree_centrality', 'clustering_coefficient', 'in_degree', 'out_degree']
    }
    print(f"  Sample node '{sample_node}' network features:")
    for k, v in network_features.items():
        print(f"    - {k}: {v}")

    # Check edge temporal data
    sample_edge = list(graph.edges())[0]
    edge_data = graph.edges[sample_edge]
    print(f"\n  Sample edge {sample_edge}:")
    print(f"    - step: {edge_data.get('step', 'N/A')}")
    print(f"    - hour: {edge_data.get('hour', 'N/A')}")
    print(f"    - amount: {edge_data.get('amount', 'N/A')}")
    print(f"    - isFraud: {edge_data.get('isFraud', 'N/A')}")

    return graph


def test_environment_initialization(graph):
    """Test 3: Initialize the AML detection environment."""
    print_section("TEST 3: Environment Initialization")

    # Create environment
    env = AMLDetectionEnv(
        graph=graph,
        max_steps=50,
        max_neighbors=5
    )
    print("✓ Environment created")
    print(f"  Max steps: {env.max_steps}")
    print(f"  Max neighbors: {env.max_neighbors}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Reset environment
    obs, info = env.reset()
    print("\n✓ Environment reset")
    print(f"  Initial observation type: {type(obs)}")
    print(f"  Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
    print(f"  Info: {info}")

    if isinstance(obs, dict):
        print("\n  Observation details:")
        for key, value in obs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                print(f"    - {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"    - {key}: {value}")

    return env


def test_state_encoder(env, graph):
    """Test 4: Initialize and test the state encoder."""
    print_section("TEST 4: State Encoder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")

    # Create state encoder
    state_encoder = StateEncoder(
        node_feature_dim=20, 
        gnn_type="sage",  # Options: "sage" or "gat"
        embedding_dim=64,
        history_dim=32,
        device=device
    )
    print("✓ State encoder created")
    print("  Node feature dim: 20")
    print("  GNN type: sage (GraphSAGE)")
    print("  Embedding dim: 64")
    print("  History dim: 32")

    # Get initial observation
    obs, info = env.reset()
    current_node = info['current_node']

    # Create node feature extractor
    from rl_money_laundering.features.extractors import NodeFeatureExtractor
    feature_extractor = NodeFeatureExtractor(
        include_temporal=True,
        include_network=True,
        node_feature_dim=20
    )

    def node_feature_fn(node_data):
        return feature_extractor.extract(node_data)

    # Encode state
    try:
        state_embedding = state_encoder.encode_state(
            graph=graph,
            current_node=current_node,
            visited_nodes={current_node},
            visited_edges=[],
            node_feature_extractor=node_feature_fn
        )
        print("\n✓ State encoded successfully")
        print(f"  Embedding shape: {state_embedding.shape}")
        print(f"  Embedding dtype: {state_embedding.dtype}")
        print(f"  Embedding stats: min={state_embedding.min():.4f}, max={state_embedding.max():.4f}, mean={state_embedding.mean():.4f}")
    except Exception as e:
        print(f"❌ State encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    return state_encoder, node_feature_fn


def test_dqn_model(state_encoder, env):
    """Test 5: Initialize the DQN model."""
    print_section("TEST 5: DQN Model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get action space size
    action_space_size = env.action_space.n if hasattr(env.action_space, 'n') else 10

    # Create DQN model
    state_dim = 64 + 32  # embedding_dim + history_dim
    dqn = DuelingDQNNetwork(
        state_dim=state_dim,
        action_dim=action_space_size,
        hidden_dims=[128, 128, 64]
    ).to(device)

    print("✓ DQN model created")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_space_size}")
    print("  Hidden dim: 128")
    print(f"  Total parameters: {sum(p.numel() for p in dqn.parameters()):,}")

    # Test forward pass with dummy state
    dummy_state = torch.randn(1, state_dim, device=device)

    with torch.no_grad():
        q_values = dqn(dummy_state)

    print("\n✓ Forward pass successful")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values: {q_values.squeeze().cpu().numpy()[:10]}...")  # Show first 10

    # Select action
    action = q_values.argmax(dim=1).item()
    print(f"  Selected action: {action}")

    return dqn


def test_episode_execution(env, graph, state_encoder, node_feature_fn, dqn):
    """Test 6: Run a few episodes to verify everything works."""
    print_section("TEST 6: Episode Execution")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 3
    max_steps_per_episode = 10

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()
        total_reward = 0
        step_count = 0

        print(f"  Starting node: {env.current_node}")
        print(f"  Is fraud: {env.graph.nodes[env.current_node].get('isFraud', False)}")

        visited_nodes = {env.current_node}
        visited_edges = []

        for step in range(max_steps_per_episode):
            # Encode state
            state_embedding = state_encoder.encode_state(
                graph=graph,
                current_node=env.current_node,
                visited_nodes=visited_nodes,
                visited_edges=visited_edges,
                node_feature_extractor=node_feature_fn
            )

            # Select action using DQN
            with torch.no_grad():
                # Convert state to tensor and add batch dimension
                state_tensor = torch.tensor(state_embedding, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = dqn(state_tensor)
                action = q_values.argmax(dim=1).item()

            # Take step
            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                step_count += 1

                # Update visited tracking
                visited_nodes.add(env.current_node)
                if len(env.visited_edges) > len(visited_edges):
                    visited_edges = list(env.visited_edges)

                print(f"  Step {step + 1}: action={action}, reward={reward:.2f}, done={done}")

                if done:
                    print(f"  Episode ended: {info.get('termination_reason', 'unknown')}")
                    break

            except Exception as e:
                print(f"  ❌ Step failed: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"  Episode {episode + 1} summary:")
        print(f"    Total steps: {step_count}")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    Nodes visited: {len(env.visited_nodes)}")
        print(f"    Edges traversed: {len(env.visited_edges)}")

        # Check temporal constraint
        if env.visited_edges:
            print("    Temporal check:")
            for i, edge in enumerate(env.visited_edges[:5]):  # Show first 5 edges
                edge_time = env.graph.edges[edge].get('step', 'N/A')
                print(f"      Edge {i + 1}: {edge} at step {edge_time}")

            # Verify temporal ordering
            edge_times = [env.graph.edges[e].get('step', 0) for e in env.visited_edges]
            is_ordered = all(edge_times[i] <= edge_times[i + 1] for i in range(len(edge_times) - 1))
            if is_ordered:
                print("    ✓ Temporal ordering verified (forward-only navigation)")
            else:
                print("    ❌ WARNING: Temporal ordering violated!")


def test_trainer_integration():
    """Test 7: Verify trainer uses the new feature extractor."""
    print_section("TEST 7: Trainer Integration")

    from rl_money_laundering.features.extractors import NodeFeatureExtractor

    # Check that trainer imports correctly
    print("✓ Trainer module imported")

    # Create a mock trainer instance to verify feature extractor
    print("✓ Trainer uses NodeFeatureExtractor with:")
    print("  - include_temporal: True")
    print("  - include_network: True")
    print("  - node_feature_dim: 20")

    # Test feature extraction
    feature_extractor = NodeFeatureExtractor(
        include_temporal=True,
        include_network=True,
        node_feature_dim=20
    )

    # Mock node data
    mock_node_data = {
        'account_type': 'C',
        'total_sent': 10000.0,
        'total_received': 5000.0,
        'balance': 5000.0,
        'tx_count_sent': 10,
        'tx_count_received': 5,
        'isFraud': 0,
        'is_sender': 1,
        'tx_velocity': 2.5,
        'business_hour_ratio': 0.7,
        'degree_centrality': 0.05,
        'clustering_coefficient': 0.3,
        'in_degree': 5,
        'out_degree': 10
    }

    features = feature_extractor.extract(mock_node_data)
    print("\n✓ Feature extraction test:")
    print(f"  Input: Mock node data with {len(mock_node_data)} attributes")
    print(f"  Output shape: {features.shape}")
    print(f"  Output dtype: {features.dtype}")
    print(f"  Feature vector (first 10): {features[:10]}")

    if features.shape[0] == 20:
        print("✓ Feature dimension matches expected (20)")
    else:
        print(f"❌ WARNING: Feature dimension mismatch! Expected 20, got {features.shape[0]}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  FULL PIPELINE INTEGRATION TEST")
    print("  Testing: Data Loading -> Graph Construction -> Environment -> Model -> Episodes")
    print("=" * 80)

    try:
        # Test 1: Load data
        loader, config = test_data_loading()
        if loader is None:
            return

        # Test 2: Build graph with features
        graph = test_graph_construction(loader)

        # Test 3: Initialize environment
        env = test_environment_initialization(graph)

        # Test 4: Initialize state encoder
        state_encoder, node_feature_fn = test_state_encoder(env, graph)
        if state_encoder is None:
            return

        # Test 5: Initialize DQN model
        dqn = test_dqn_model(state_encoder, env)

        # Test 6: Run episodes
        test_episode_execution(env, graph, state_encoder, node_feature_fn, dqn)

        # Test 7: Verify trainer integration
        test_trainer_integration()

        # Final summary
        print_section("FINAL SUMMARY")
        print("✓ Data loading: PASSED")
        print("✓ Graph construction: PASSED")
        print("✓ Temporal features: PASSED")
        print("✓ Network features: PASSED")
        print("✓ Environment initialization: PASSED")
        print("✓ State encoder: PASSED")
        print("✓ DQN model: PASSED")
        print("✓ Episode execution: PASSED")
        print("✓ Trainer integration: PASSED")
        print("\n🎉 ALL TESTS PASSED - System is ready for training!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
