"""
Standalone component tests for RLlib integration.

Run this directly to test GraphSpace and batch utilities without needing Ray installed.
"""

import sys
from pathlib import Path
import importlib.util

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import torch
from torch_geometric.data import Batch

# Import modules directly to avoid Ray dependencies in __init__.py
def import_module_from_file(module_name, file_path):
    """Import a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import graph_space directly
graph_space_path = src_path / "rl_money_laundering" / "rllib_integration" / "graph_space.py"
graph_space_module = import_module_from_file("graph_space", graph_space_path)
GraphSpace = graph_space_module.GraphSpace

# Import batch_utils directly
batch_utils_path = src_path / "rl_money_laundering" / "rllib_integration" / "batch_utils.py"
batch_utils_module = import_module_from_file("batch_utils", batch_utils_path)
batch_graph_observations = batch_utils_module.batch_graph_observations
GraphBatchCollator = batch_utils_module.GraphBatchCollator


def test_graph_space_basic():
    """Test GraphSpace initialization and basic operations."""
    print("\n[TEST] GraphSpace initialization...")

    space = GraphSpace(
        max_nodes=50,
        max_edges=200,
        node_feature_dim=10,
        context_feature_dim=5
    )

    assert space.max_nodes == 50
    assert space.max_edges == 200
    print("OK GraphSpace initialized correctly")

    # Test sampling
    print("\n[TEST] GraphSpace sampling...")
    sample = space.sample()

    assert sample["node_features"].shape == (50, 10)
    assert sample["edge_index"].shape == (2, 200)
    assert sample["context_features"].shape == (5,)
    print(f"OK Sample shapes correct")
    print(f"  - num_nodes: {sample['num_nodes']}")
    print(f"  - num_edges: {sample['num_edges']}")

    return True


def test_graph_space_contains():
    """Test GraphSpace membership validation."""
    print("\n[TEST] GraphSpace.contains()...")

    space = GraphSpace(max_nodes=20, max_edges=50, node_feature_dim=5, context_feature_dim=3)

    # Valid sample - use numpy 2.x compatible API
    rng = np.random.default_rng(42)
    valid_sample = {
        "node_features": rng.standard_normal((20, 5)).astype(np.float32),
        "edge_index": rng.integers(0, 15, size=(2, 50)).astype(np.int64),
        "num_nodes": np.array([15], dtype=np.int64),  # shape (1,) for vectorization
        "num_edges": np.array([30], dtype=np.int64),  # shape (1,) for vectorization
        "context_features": rng.standard_normal(3).astype(np.float32),
    }

    # Debug output
    print(f"  Sample shapes:")
    print(f"    node_features: {valid_sample['node_features'].shape} dtype={valid_sample['node_features'].dtype}")
    print(f"    edge_index: {valid_sample['edge_index'].shape} dtype={valid_sample['edge_index'].dtype}")
    print(f"    num_nodes: {valid_sample['num_nodes']} shape={valid_sample['num_nodes'].shape}")
    print(f"    num_edges: {valid_sample['num_edges']} shape={valid_sample['num_edges'].shape}")
    print(f"    context_features: {valid_sample['context_features'].shape} dtype={valid_sample['context_features'].dtype}")

    result = space.contains(valid_sample)
    if not result:
        print(f"  FAIL VALIDATION FAILED - sample rejected")
        # Try to debug which check failed
        print(f"  Checking individual validations:")
        print(f"    Is dict: {isinstance(valid_sample, dict)}")
        print(f"    Has all keys: {set(['node_features', 'edge_index', 'num_nodes', 'num_edges', 'context_features']).issubset(valid_sample.keys())}")
        print(f"    Node features shape match: {valid_sample['node_features'].shape == (20, 5)}")
        print(f"    Edge index shape match: {valid_sample['edge_index'].shape == (2, 50)}")
        print(f"    Context features shape match: {valid_sample['context_features'].shape == (3,)}")
        print(f"    Num nodes shape: {valid_sample['num_nodes'].shape == (1,)}")
        print(f"    Num edges shape: {valid_sample['num_edges'].shape == (1,)}")
        print(f"    Num nodes bounds: 1 <= {valid_sample['num_nodes'][0]} <= 20 = {1 <= valid_sample['num_nodes'][0] <= 20}")
        print(f"    Num edges bounds: 0 <= {valid_sample['num_edges'][0]} <= 50 = {0 <= valid_sample['num_edges'][0] <= 50}")

        # Check edge indices
        num_edges_val = int(valid_sample['num_edges'][0])
        num_nodes_val = int(valid_sample['num_nodes'][0])
        if num_edges_val > 0:
            active_edges = valid_sample['edge_index'][:, :num_edges_val]
            print(f"    Active edges range: min={active_edges.min()}, max={active_edges.max()}, num_nodes={num_nodes_val}")
            print(f"    Edge indices valid: {np.all(active_edges >= 0) and np.all(active_edges < num_nodes_val)}")

    assert result, "Valid sample rejected"
    print("OK Valid sample accepted")

    # Invalid sample - wrong shape
    invalid_sample = valid_sample.copy()
    invalid_sample["node_features"] = rng.standard_normal((10, 5)).astype(np.float32)

    assert not space.contains(invalid_sample), "Invalid sample accepted"
    print("OK Invalid sample rejected")

    return True


def test_batch_processing():
    """Test PyG batch processing utilities."""
    print("\n[TEST] Batch processing...")

    batch_size = 4
    max_nodes = 20
    max_edges = 50
    node_feature_dim = 8
    context_feature_dim = 4

    # Create batch of graph observations
    graph_obs_batch = {
        "node_features": torch.randn(batch_size, max_nodes, node_feature_dim),
        "edge_index": torch.randint(0, max_nodes, size=(batch_size, 2, max_edges)),
        "num_nodes": torch.tensor([10, 15, 8, 12], dtype=torch.int64),
        "num_edges": torch.tensor([25, 40, 18, 30], dtype=torch.int64),
        "context_features": torch.randn(batch_size, context_feature_dim),
    }

    # Batch into PyG Batch
    pyg_batch = batch_graph_observations(graph_obs_batch)

    assert isinstance(pyg_batch, Batch)
    print(f"OK Created PyG Batch")

    expected_total_nodes = sum([10, 15, 8, 12])
    expected_total_edges = sum([25, 40, 18, 30])

    assert pyg_batch.num_nodes == expected_total_nodes
    assert pyg_batch.edge_index.shape[1] == expected_total_edges
    print(f"OK Batch has correct total nodes ({expected_total_nodes}) and edges ({expected_total_edges})")

    assert pyg_batch.context.shape == (batch_size, context_feature_dim)
    print(f"OK Context features preserved")

    return True


def test_extract_first_nodes():
    """Test extracting first node per graph from batch."""
    print("\n[TEST] Extract first node embeddings...")

    batch_size = 3
    embedding_dim = 16

    # Create dummy node embeddings (30 total nodes)
    node_embeddings = torch.randn(30, embedding_dim)

    # Create batch assignment (first graph: 10 nodes, second: 15, third: 5)
    batch_assignment = torch.tensor(
        [0]*10 + [1]*15 + [2]*5,
        dtype=torch.long
    )

    pyg_batch = Batch(
        x=node_embeddings,
        batch=batch_assignment,
        num_graphs=batch_size
    )

    # Extract first nodes
    collator = GraphBatchCollator()
    first_nodes = collator.extract_first_nodes(node_embeddings, pyg_batch)

    assert first_nodes.shape == (batch_size, embedding_dim)
    print(f"OK Extracted {batch_size} first nodes with dim {embedding_dim}")

    # Verify correct nodes extracted
    assert torch.allclose(first_nodes[0], node_embeddings[0])
    assert torch.allclose(first_nodes[1], node_embeddings[10])
    assert torch.allclose(first_nodes[2], node_embeddings[25])
    print(f"OK Correct nodes extracted (indices 0, 10, 25)")

    return True


def test_collator_end_to_end():
    """Test GraphBatchCollator full workflow."""
    print("\n[TEST] GraphBatchCollator end-to-end...")

    collator = GraphBatchCollator()
    batch_size = 2

    graph_obs_batch = {
        "node_features": torch.randn(batch_size, 10, 5),
        "edge_index": torch.randint(0, 10, size=(batch_size, 2, 20)),
        "num_nodes": torch.tensor([8, 9], dtype=torch.int64),
        "num_edges": torch.tensor([15, 18], dtype=torch.int64),
        "context_features": torch.randn(batch_size, 3),
    }

    pyg_batch = collator(graph_obs_batch)

    assert isinstance(pyg_batch, Batch)
    assert hasattr(pyg_batch, 'context')
    assert pyg_batch.num_graphs == batch_size
    print(f"OK Collator produced valid batch with {batch_size} graphs")

    return True


def test_edge_cases():
    """Test edge cases like empty graphs."""
    print("\n[TEST] Edge cases...")

    # Zero edges
    batch_size = 2
    graph_obs_batch = {
        "node_features": torch.randn(batch_size, 10, 5),
        "edge_index": torch.zeros(batch_size, 2, 20, dtype=torch.int64),
        "num_nodes": torch.tensor([5, 3], dtype=torch.int64),
        "num_edges": torch.tensor([0, 0], dtype=torch.int64),
        "context_features": torch.randn(batch_size, 3),
    }

    collator = GraphBatchCollator()
    pyg_batch = collator(graph_obs_batch)

    assert pyg_batch.edge_index.shape[1] == 0
    print("OK Handled graphs with zero edges")

    # Single graph
    graph_obs_batch = {
        "node_features": torch.randn(1, 10, 5),
        "edge_index": torch.randint(0, 10, size=(1, 2, 20)),
        "num_nodes": torch.tensor([7], dtype=torch.int64),
        "num_edges": torch.tensor([12], dtype=torch.int64),
        "context_features": torch.randn(1, 3),
    }

    pyg_batch = collator(graph_obs_batch)
    assert pyg_batch.num_graphs == 1
    print("OK Handled single graph batch")

    return True


def main():
    """Run all tests."""
    print("="*70)
    print("RLLIB COMPONENT TESTS (Standalone)")
    print("="*70)
    print("\nTesting components without requiring Ray installation...")

    tests = [
        ("GraphSpace Basic", test_graph_space_basic),
        ("GraphSpace Contains", test_graph_space_contains),
        ("Batch Processing", test_batch_processing),
        ("Extract First Nodes", test_extract_first_nodes),
        ("Collator End-to-End", test_collator_end_to_end),
        ("Edge Cases", test_edge_cases),
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

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)

    if failed == 0:
        print("\nOK ALL TESTS PASSED!")
        return 0
    else:
        print(f"\nFAIL {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
