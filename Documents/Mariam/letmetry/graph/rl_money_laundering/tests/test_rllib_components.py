"""
Comprehensive unit tests for RLlib integration components.

Tests each component in isolation before integration testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from rl_money_laundering.rllib_integration.graph_space import GraphSpace
from rl_money_laundering.rllib_integration.batch_utils import batch_graph_observations, GraphBatchCollator
from rl_money_laundering.rllib_integration.env_wrapper import RLlibAMLEnv
from rl_money_laundering.environment import AMLDetectionEnv


class TestGraphSpace:
    """Test GraphSpace observation space."""

    def test_initialization(self):
        """Test GraphSpace can be created with valid parameters."""
        space = GraphSpace(
            max_nodes=50,
            max_edges=200,
            node_feature_dim=10,
            context_feature_dim=5
        )

        assert space.max_nodes == 50
        assert space.max_edges == 200
        assert space.node_feature_dim == 10
        assert space.context_feature_dim == 5

        # Check nested spaces
        assert "node_features" in space.spaces
        assert "edge_index" in space.spaces
        assert "num_nodes" in space.spaces
        assert "num_edges" in space.spaces
        assert "context_features" in space.spaces

    def test_sample(self):
        """Test sampling from GraphSpace produces valid observations."""
        space = GraphSpace(max_nodes=30, max_edges=100, node_feature_dim=8, context_feature_dim=4)

        sample = space.sample()

        # Check all keys present
        assert "node_features" in sample
        assert "edge_index" in sample
        assert "num_nodes" in sample
        assert "num_edges" in sample
        assert "context_features" in sample

        # Check shapes
        assert sample["node_features"].shape == (30, 8)
        assert sample["edge_index"].shape == (2, 100)
        assert sample["num_nodes"].shape == ()
        assert sample["num_edges"].shape == ()
        assert sample["context_features"].shape == (4,)

        # Check types
        assert sample["node_features"].dtype == np.float32
        assert sample["edge_index"].dtype == np.int64
        assert sample["num_nodes"].dtype == np.int64
        assert sample["num_edges"].dtype == np.int64
        assert sample["context_features"].dtype == np.float32

    def test_contains(self):
        """Test membership checking in GraphSpace."""
        space = GraphSpace(max_nodes=20, max_edges=50, node_feature_dim=5, context_feature_dim=3)

        # Valid sample
        valid_sample = {
            "node_features": np.random.randn(20, 5).astype(np.float32),
            "edge_index": np.random.randint(0, 20, size=(2, 50)).astype(np.int64),
            "num_nodes": np.array(15, dtype=np.int64),
            "num_edges": np.array(30, dtype=np.int64),
            "context_features": np.random.randn(3).astype(np.float32),
        }
        assert space.contains(valid_sample)

        # Invalid sample - wrong shape
        invalid_sample = valid_sample.copy()
        invalid_sample["node_features"] = np.random.randn(10, 5).astype(np.float32)
        assert not space.contains(invalid_sample)

    def test_padding(self):
        """Test that smaller graphs can be padded to fit space."""
        space = GraphSpace(max_nodes=50, max_edges=200, node_feature_dim=10, context_feature_dim=5)

        # Create small graph
        small_graph = {
            "node_features": np.random.randn(10, 10).astype(np.float32),
            "edge_index": np.random.randint(0, 10, size=(2, 20)).astype(np.int64),
            "num_nodes": np.array(10, dtype=np.int64),
            "num_edges": np.array(20, dtype=np.int64),
            "context_features": np.random.randn(5).astype(np.float32),
        }

        # Pad to max size
        padded = {
            "node_features": np.pad(small_graph["node_features"],
                                   ((0, 40), (0, 0)), mode='constant'),
            "edge_index": np.pad(small_graph["edge_index"],
                                ((0, 0), (0, 180)), mode='constant'),
            "num_nodes": small_graph["num_nodes"],
            "num_edges": small_graph["num_edges"],
            "context_features": small_graph["context_features"],
        }

        assert space.contains(padded)


class TestBatchUtils:
    """Test PyG batch processing utilities."""

    def test_batch_graph_observations(self):
        """Test batching multiple graph observations."""
        batch_size = 4
        max_nodes = 20
        max_edges = 50
        node_feature_dim = 8
        context_feature_dim = 4

        # Create batch of graph observations (as would come from environment)
        graph_obs_batch = {
            "node_features": torch.randn(batch_size, max_nodes, node_feature_dim),
            "edge_index": torch.randint(0, max_nodes, size=(batch_size, 2, max_edges)),
            "num_nodes": torch.tensor([10, 15, 8, 12], dtype=torch.int64),
            "num_edges": torch.tensor([25, 40, 18, 30], dtype=torch.int64),
            "context_features": torch.randn(batch_size, context_feature_dim),
        }

        # Batch into PyG Batch
        pyg_batch = batch_graph_observations(graph_obs_batch)

        # Check it's a valid Batch
        assert isinstance(pyg_batch, Batch)

        # Check total nodes and edges
        expected_total_nodes = sum([10, 15, 8, 12])
        expected_total_edges = sum([25, 40, 18, 30])

        assert pyg_batch.num_nodes == expected_total_nodes
        assert pyg_batch.edge_index.shape[1] == expected_total_edges

        # Check batch assignment
        assert pyg_batch.batch.shape[0] == expected_total_nodes
        assert pyg_batch.batch.min() == 0
        assert pyg_batch.batch.max() == batch_size - 1

        # Check context features preserved
        assert pyg_batch.context.shape == (batch_size, context_feature_dim)

    def test_extract_first_node_embeddings(self):
        """Test extracting first node embedding per graph in batch."""
        batch_size = 3
        embedding_dim = 16

        # Create dummy node embeddings
        node_embeddings = torch.randn(30, embedding_dim)  # 30 total nodes

        # Create batch assignment (first graph has 10 nodes, second 15, third 5)
        batch_assignment = torch.tensor(
            [0]*10 + [1]*15 + [2]*5,
            dtype=torch.long
        )

        # Create PyG batch
        pyg_batch = Batch(
            x=node_embeddings,
            batch=batch_assignment,
            num_graphs=batch_size
        )

        # Extract first nodes
        collator = GraphBatchCollator()
        first_nodes = collator.extract_first_nodes(node_embeddings, pyg_batch)

        # Check shape
        assert first_nodes.shape == (batch_size, embedding_dim)

        # Check values (should be embeddings at indices 0, 10, 25)
        assert torch.allclose(first_nodes[0], node_embeddings[0])
        assert torch.allclose(first_nodes[1], node_embeddings[10])
        assert torch.allclose(first_nodes[2], node_embeddings[25])

    def test_collator_end_to_end(self):
        """Test GraphBatchCollator end-to-end."""
        collator = GraphBatchCollator()

        batch_size = 2
        graph_obs_batch = {
            "node_features": torch.randn(batch_size, 10, 5),
            "edge_index": torch.randint(0, 10, size=(batch_size, 2, 20)),
            "num_nodes": torch.tensor([8, 9], dtype=torch.int64),
            "num_edges": torch.tensor([15, 18], dtype=torch.int64),
            "context_features": torch.randn(batch_size, 3),
        }

        # Batch
        pyg_batch = collator(graph_obs_batch)

        assert isinstance(pyg_batch, Batch)
        assert hasattr(pyg_batch, 'context')
        assert pyg_batch.num_graphs == batch_size


class TestEnvironmentWrapper:
    """Test RLlib environment wrapper."""

    @pytest.fixture
    def sample_graph_data(self, tmp_path):
        """Create minimal sample data for testing."""
        # Create minimal CSV files
        node_features_path = tmp_path / "nodes.csv"
        edges_path = tmp_path / "edges.csv"
        classes_path = tmp_path / "classes.csv"

        # Minimal node features (10 nodes, 2 features)
        with open(node_features_path, 'w') as f:
            f.write("txId,feature1,feature2\n")
            for i in range(10):
                f.write(f"{i},{np.random.rand()},{np.random.rand()}\n")

        # Minimal edges (15 edges)
        with open(edges_path, 'w') as f:
            f.write("txId1,txId2\n")
            edges = [(0,1), (1,2), (2,3), (3,4), (4,5),
                    (5,6), (6,7), (7,8), (8,9), (0,5),
                    (1,6), (2,7), (3,8), (4,9), (0,9)]
            for src, dst in edges:
                f.write(f"{src},{dst}\n")

        # Minimal classes (5 fraud, 5 unknown)
        with open(classes_path, 'w') as f:
            f.write("txId,class\n")
            for i in range(5):
                f.write(f"{i},1\n")  # Fraud
            for i in range(5, 10):
                f.write(f"{i},unknown\n")  # Unknown

        return {
            "node_features": str(node_features_path),
            "edges": str(edges_path),
            "classes": str(classes_path)
        }

    def test_environment_creation(self, sample_graph_data):
        """Test RLlib environment can be created."""
        env = RLlibAMLEnv({
            "node_features_path": sample_graph_data["node_features"],
            "edges_path": sample_graph_data["edges"],
            "classes_path": sample_graph_data["classes"],
            "max_nodes": 20,
            "max_edges": 50,
        })

        assert isinstance(env, RLlibAMLEnv)
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')

    def test_reset(self, sample_graph_data):
        """Test environment reset returns valid observation."""
        env = RLlibAMLEnv({
            "node_features_path": sample_graph_data["node_features"],
            "edges_path": sample_graph_data["edges"],
            "classes_path": sample_graph_data["classes"],
            "max_nodes": 20,
            "max_edges": 50,
        })

        obs, info = env.reset()

        # Check observation is in observation space
        assert env.observation_space.contains(obs)

        # Check all required keys
        assert "node_features" in obs
        assert "edge_index" in obs
        assert "num_nodes" in obs
        assert "num_edges" in obs
        assert "context_features" in obs

        # Check info dict
        assert isinstance(info, dict)

    def test_step(self, sample_graph_data):
        """Test environment step with action."""
        env = RLlibAMLEnv({
            "node_features_path": sample_graph_data["node_features"],
            "edges_path": sample_graph_data["edges"],
            "classes_path": sample_graph_data["classes"],
            "max_nodes": 20,
            "max_edges": 50,
        })

        obs, info = env.reset()

        # Take valid action (choose edge index)
        action = 0

        obs, reward, terminated, truncated, info = env.step(action)

        # Check observation
        assert env.observation_space.contains(obs)

        # Check reward is float
        assert isinstance(reward, (float, np.floating))

        # Check termination flags are bool
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

        # Check info
        assert isinstance(info, dict)

    def test_episode_completion(self, sample_graph_data):
        """Test completing a full episode."""
        env = RLlibAMLEnv({
            "node_features_path": sample_graph_data["node_features"],
            "edges_path": sample_graph_data["edges"],
            "classes_path": sample_graph_data["classes"],
            "max_nodes": 20,
            "max_edges": 50,
        })

        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        max_steps = 100

        while not (terminated or truncated) and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert steps < max_steps, "Episode should terminate within max_steps"


class TestGNNRLModule:
    """Test GNN RLModule components."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for RLModule."""
        from ray.rllib.core.rl_module.rl_module import RLModuleConfig
        from gymnasium.spaces import Box, Discrete

        observation_space = GraphSpace(
            max_nodes=30,
            max_edges=100,
            node_feature_dim=8,
            context_feature_dim=4
        )
        action_space = Discrete(10)

        return RLModuleConfig(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict={
                "gnn_type": "rmganets",
                "node_feature_dim": 8,
                "embedding_dim": 32,
                "history_dim": 16,
                "context_feature_dim": 4,
                "hidden_dims": [64, 64],
            }
        )

    def test_rlmodule_creation(self, mock_config):
        """Test GNN RLModule can be instantiated."""
        from rl_money_laundering.rllib_integration.gnn_rl_module import GNNDQNModule

        module = GNNDQNModule(mock_config)

        assert module is not None
        assert hasattr(module, '_forward_inference')
        assert hasattr(module, '_forward_exploration')
        assert hasattr(module, '_forward_train')

    def test_forward_inference(self, mock_config):
        """Test forward pass in inference mode."""
        from rl_money_laundering.rllib_integration.gnn_rl_module import GNNDQNModule
        from ray.rllib.core.columns import Columns

        module = GNNDQNModule(mock_config)
        module.eval()

        # Create batch of observations
        batch_size = 2
        batch = {
            Columns.OBS: {
                "node_features": torch.randn(batch_size, 30, 8),
                "edge_index": torch.randint(0, 30, size=(batch_size, 2, 100)),
                "num_nodes": torch.tensor([20, 25], dtype=torch.int64),
                "num_edges": torch.tensor([60, 80], dtype=torch.int64),
                "context_features": torch.randn(batch_size, 4),
            }
        }

        with torch.no_grad():
            output = module._forward_inference(batch)

        # Check output has action dist inputs (Q-values)
        assert Columns.ACTION_DIST_INPUTS in output

        # Check Q-values shape (batch_size x num_actions)
        q_values = output[Columns.ACTION_DIST_INPUTS]
        assert q_values.shape == (batch_size, 10)

    def test_forward_train(self, mock_config):
        """Test forward pass in training mode."""
        from rl_money_laundering.rllib_integration.gnn_rl_module import GNNDQNModule
        from ray.rllib.core.columns import Columns

        module = GNNDQNModule(mock_config)
        module.train()

        batch_size = 4
        batch = {
            Columns.OBS: {
                "node_features": torch.randn(batch_size, 30, 8),
                "edge_index": torch.randint(0, 30, size=(batch_size, 2, 100)),
                "num_nodes": torch.tensor([15, 20, 25, 18], dtype=torch.int64),
                "num_edges": torch.tensor([40, 60, 80, 50], dtype=torch.int64),
                "context_features": torch.randn(batch_size, 4),
            }
        }

        output = module._forward_train(batch)

        # Check output
        assert Columns.ACTION_DIST_INPUTS in output
        q_values = output[Columns.ACTION_DIST_INPUTS]
        assert q_values.shape == (batch_size, 10)

        # Check gradients can flow
        assert q_values.requires_grad


def run_all_tests():
    """Run all tests and report results."""
    print("="*70)
    print("RLLIB INTEGRATION TEST SUITE")
    print("="*70)
    print()

    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]

    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print()
        print("="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
    else:
        print()
        print("="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70)

    return exit_code


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
