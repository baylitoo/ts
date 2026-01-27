"""
Standalone tests for RLlib integration components that don't require Ray to be installed.

Tests GraphSpace and batch utilities which only depend on gymnasium, numpy, and PyTorch.
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


class TestGraphSpace:
    """Test GraphSpace observation space (gymnasium-based, no Ray needed)."""

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

    def test_contains_valid(self):
        """Test membership checking with valid sample."""
        space = GraphSpace(max_nodes=20, max_edges=50, node_feature_dim=5, context_feature_dim=3)

        # Valid sample
        valid_sample = {
            "node_features": np.random.randn(20, 5).astype(np.float32),
            "edge_index": np.random.randint(0, 15, size=(2, 50)).astype(np.int64),
            "num_nodes": np.array(15, dtype=np.int64),
            "num_edges": np.array(30, dtype=np.int64),
            "context_features": np.random.randn(3).astype(np.float32),
        }
        assert space.contains(valid_sample)

    def test_contains_invalid_shape(self):
        """Test membership checking rejects wrong shapes."""
        space = GraphSpace(max_nodes=20, max_edges=50, node_feature_dim=5, context_feature_dim=3)

        # Invalid sample - wrong node feature shape
        invalid_sample = {
            "node_features": np.random.randn(10, 5).astype(np.float32),  # Wrong: should be (20, 5)
            "edge_index": np.random.randint(0, 20, size=(2, 50)).astype(np.int64),
            "num_nodes": np.array(10, dtype=np.int64),
            "num_edges": np.array(30, dtype=np.int64),
            "context_features": np.random.randn(3).astype(np.float32),
        }
        assert not space.contains(invalid_sample)

    def test_contains_invalid_counts(self):
        """Test membership checking rejects invalid node/edge counts."""
        space = GraphSpace(max_nodes=20, max_edges=50, node_feature_dim=5, context_feature_dim=3)

        # Invalid sample - num_nodes exceeds max_nodes
        invalid_sample = {
            "node_features": np.random.randn(20, 5).astype(np.float32),
            "edge_index": np.random.randint(0, 20, size=(2, 50)).astype(np.int64),
            "num_nodes": np.array(25, dtype=np.int64),  # Invalid: > max_nodes
            "num_edges": np.array(30, dtype=np.int64),
            "context_features": np.random.randn(3).astype(np.float32),
        }
        assert not space.contains(invalid_sample)

    def test_contains_invalid_edge_indices(self):
        """Test membership checking rejects out-of-bounds edge indices."""
        space = GraphSpace(max_nodes=20, max_edges=50, node_feature_dim=5, context_feature_dim=3)

        # Invalid sample - edge indices reference non-existent nodes
        invalid_sample = {
            "node_features": np.random.randn(20, 5).astype(np.float32),
            "edge_index": np.array([[0, 18], [1, 2]], dtype=np.int64),  # 18 >= 10 (num_nodes)
            "num_nodes": np.array(10, dtype=np.int64),
            "num_edges": np.array(2, dtype=np.int64),
            "context_features": np.random.randn(3).astype(np.float32),
        }
        # Pad edge_index to max_edges
        padded_edge_index = np.zeros((2, 50), dtype=np.int64)
        padded_edge_index[:, :2] = invalid_sample["edge_index"]
        invalid_sample["edge_index"] = padded_edge_index

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

    def test_repr(self):
        """Test string representation."""
        space = GraphSpace(max_nodes=30, max_edges=100, node_feature_dim=8, context_feature_dim=4)
        repr_str = repr(space)

        assert "GraphSpace" in repr_str
        assert "30" in repr_str  # max_nodes
        assert "100" in repr_str  # max_edges
        assert "8" in repr_str  # node_feature_dim
        assert "4" in repr_str  # context_feature_dim


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

    def test_empty_graphs(self):
        """Test handling graphs with zero edges."""
        batch_size = 2
        graph_obs_batch = {
            "node_features": torch.randn(batch_size, 10, 5),
            "edge_index": torch.randint(0, 10, size=(batch_size, 2, 20)),
            "num_nodes": torch.tensor([5, 3], dtype=torch.int64),
            "num_edges": torch.tensor([0, 0], dtype=torch.int64),  # No edges
            "context_features": torch.randn(batch_size, 3),
        }

        collator = GraphBatchCollator()
        pyg_batch = collator(graph_obs_batch)

        # Should still work with 0 edges
        assert isinstance(pyg_batch, Batch)
        assert pyg_batch.edge_index.shape[1] == 0

    def test_single_graph_batch(self):
        """Test batching a single graph."""
        batch_size = 1
        graph_obs_batch = {
            "node_features": torch.randn(batch_size, 10, 5),
            "edge_index": torch.randint(0, 10, size=(batch_size, 2, 20)),
            "num_nodes": torch.tensor([7], dtype=torch.int64),
            "num_edges": torch.tensor([12], dtype=torch.int64),
            "context_features": torch.randn(batch_size, 3),
        }

        collator = GraphBatchCollator()
        pyg_batch = collator(graph_obs_batch)

        assert isinstance(pyg_batch, Batch)
        assert pyg_batch.num_graphs == 1
        assert pyg_batch.num_nodes == 7


def run_all_tests():
    """Run all standalone tests."""
    print("="*70)
    print("RLLIB STANDALONE TESTS (No Ray Required)")
    print("="*70)
    print()

    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
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
