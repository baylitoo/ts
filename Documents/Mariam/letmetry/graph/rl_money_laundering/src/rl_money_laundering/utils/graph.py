"""
Graph utility functions.

Provides reusable operations for graph manipulation, subgraph extraction,
and conversion between NetworkX and PyTorch Geometric formats.
"""

from __future__ import annotations

from typing import Any, Callable, List, Set

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def extract_k_hop_subgraph(
    graph: nx.DiGraph,
    center_node: str,
    k: int = 2,
    max_nodes: int = 100
) -> nx.DiGraph:
    """
    Extract k-hop subgraph around a center node.

    Args:
        graph: Full transaction graph
        center_node: Node to center subgraph around
        k: Number of hops to traverse
        max_nodes: Maximum nodes to include (for memory efficiency)

    Returns:
        Subgraph containing k-hop neighborhood
    """
    if center_node not in graph:
        return nx.DiGraph()

    # BFS to find k-hop neighbors
    visited: Set[str] = {center_node}
    current_level = {center_node}

    for _ in range(k):
        next_level: Set[str] = set()

        for node in current_level:
            # Add predecessors and successors
            next_level.update(graph.predecessors(node))
            next_level.update(graph.successors(node))

        # Remove already visited
        next_level -= visited
        visited.update(next_level)
        current_level = next_level

        # Stop if we exceed max_nodes
        if len(visited) > max_nodes:
            break

    # Extract subgraph
    subgraph_nodes = list(visited)[:max_nodes]
    subgraph = graph.subgraph(subgraph_nodes).copy()

    return subgraph


def networkx_to_pyg(
    graph: nx.DiGraph,
    node_feature_extractor: Callable[[dict], np.ndarray],
    node_list: List[str] | None = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        graph: NetworkX directed graph
        node_feature_extractor: Function to extract features from node data
        node_list: Optional list of nodes (for consistent ordering)

    Returns:
        PyTorch Geometric Data object
    """
    if node_list is None:
        node_list = list(graph.nodes())

    # Create node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Extract node features
    node_features = []
    for node in node_list:
        node_data = graph.nodes.get(node, {})
        features = node_feature_extractor(node_data)
        node_features.append(features)

    x = torch.tensor(np.array(node_features), dtype=torch.float32)

    # Extract edges
    edge_index = []
    edge_attr = []

    for src, dst, edge_data in graph.edges(data=True):
        if src in node_to_idx and dst in node_to_idx:
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]

            edge_index.append([src_idx, dst_idx])

            # Extract edge features (amount, timestamp, etc.)
            edge_features = [
                edge_data.get('amount', 0.0),
                edge_data.get('timestamp', 0.0),
            ]
            edge_attr.append(edge_features)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_neighbors(
    graph: nx.DiGraph,
    node: str,
    max_neighbors: int = 5,
    direction: str = "both"
) -> List[str]:
    """
    Get neighbors of a node with optional filtering.

    Args:
        graph: Transaction graph
        node: Node to get neighbors for
        max_neighbors: Maximum neighbors to return
        direction: "in", "out", or "both"

    Returns:
        List of neighbor node IDs
    """
    if node not in graph:
        return []

    neighbors: Set[str] = set()

    if direction in ("out", "both"):
        neighbors.update(graph.successors(node))

    if direction in ("in", "both"):
        neighbors.update(graph.predecessors(node))

    # Convert to list and limit
    neighbor_list = list(neighbors)[:max_neighbors]

    return neighbor_list


def compute_graph_statistics(graph: nx.DiGraph) -> dict[str, Any]:
    """
    Compute graph-level statistics.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "is_connected": nx.is_weakly_connected(graph),
    }

    # Degree statistics
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]

    stats.update({
        "avg_in_degree": np.mean(in_degrees) if in_degrees else 0.0,
        "avg_out_degree": np.mean(out_degrees) if out_degrees else 0.0,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
    })

    return stats


def find_fraud_subgraphs(
    graph: nx.DiGraph,
    fraud_nodes: List[str],
    k_hop: int = 2
) -> List[nx.DiGraph]:
    """
    Find k-hop subgraphs around fraud nodes.

    Useful for curriculum learning and targeted training.

    Args:
        graph: Full transaction graph
        fraud_nodes: List of known fraud nodes
        k_hop: Number of hops around each fraud node

    Returns:
        List of subgraphs centered on fraud
    """
    subgraphs = []

    for fraud_node in fraud_nodes:
        if fraud_node in graph:
            subgraph = extract_k_hop_subgraph(graph, fraud_node, k=k_hop)
            if subgraph.number_of_nodes() > 1:  # Skip isolated nodes
                subgraphs.append(subgraph)

    return subgraphs


def entity_disjoint_split(
    graph: nx.DiGraph,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int | None = 42,
    min_edges_per_split: int = 10
) -> tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph]:
    """
    Split graph into train/val/test with NO account overlap (entity-disjoint).

    This prevents data leakage where the model learns patterns from accounts
    that appear in both train and test sets.

    Method:
    1. Split all accounts (nodes) into disjoint sets
    2. Assign edges to splits based on BOTH endpoints being in that split
    3. Discard edges that cross split boundaries

    Args:
        graph: Transaction graph to split
        test_size: Fraction of nodes for test (0.0-1.0)
        val_size: Fraction of nodes for validation (0.0-1.0)
        random_state: Random seed for reproducibility
        min_edges_per_split: Minimum edges required per split

    Returns:
        (train_graph, val_graph, test_graph)

    Example:
        Accounts: [C1, C2, C3, C4, C5, C6]
        Train: [C1, C2, C3]
        Val: [C4]
        Test: [C5, C6]

        Edge C1->C2: TRAIN (both in train)
        Edge C1->C5: DISCARD (crosses train/test boundary)
        Edge C5->C6: TEST (both in test)

    Raises:
        ValueError: If splits would have too few edges
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Get all accounts
    all_accounts = list(graph.nodes())
    n_accounts = len(all_accounts)

    # Shuffle accounts
    shuffled_accounts = np.random.permutation(all_accounts)

    # Calculate split sizes
    n_test = int(n_accounts * test_size)
    n_val = int(n_accounts * val_size)

    # Split accounts into disjoint sets
    test_accounts = set(shuffled_accounts[:n_test])
    val_accounts = set(shuffled_accounts[n_test:n_test + n_val])
    train_accounts = set(shuffled_accounts[n_test + n_val:])

    # Verify disjoint
    assert len(train_accounts & test_accounts) == 0, "Train/test overlap!"
    assert len(train_accounts & val_accounts) == 0, "Train/val overlap!"
    assert len(test_accounts & val_accounts) == 0, "Test/val overlap!"
    assert len(train_accounts) + len(val_accounts) + len(test_accounts) == n_accounts, (
        "Split sizes do not sum to total accounts"
    )

    # Assign edges to splits (both endpoints must be in same split)
    train_edges = []
    val_edges = []
    test_edges = []
    discarded_edges = 0

    for src, dst in graph.edges():
        if src in train_accounts and dst in train_accounts:
            train_edges.append((src, dst))
        elif src in val_accounts and dst in val_accounts:
            val_edges.append((src, dst))
        elif src in test_accounts and dst in test_accounts:
            test_edges.append((src, dst))
        else:
            # Edge crosses split boundary - discard
            discarded_edges += 1

    # Check minimum edges
    if len(train_edges) < min_edges_per_split:
        raise ValueError(
            f"Train split has only {len(train_edges)} edges (min {min_edges_per_split}). "
            f"Try larger test_size or smaller min_edges_per_split."
        )
    if len(test_edges) < min_edges_per_split:
        raise ValueError(
            f"Test split has only {len(test_edges)} edges (min {min_edges_per_split}). "
            f"Try smaller test_size or smaller min_edges_per_split."
        )

    # Create subgraphs
    train_graph = graph.edge_subgraph(train_edges).copy()
    val_graph = graph.edge_subgraph(val_edges).copy() if val_edges else nx.DiGraph()
    test_graph = graph.edge_subgraph(test_edges).copy()

    # Log statistics
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"Entity-disjoint split: train={len(train_accounts)} accounts / {len(train_edges)} edges, "
        f"val={len(val_accounts)} accounts / {len(val_edges)} edges, "
        f"test={len(test_accounts)} accounts / {len(test_edges)} edges, "
        f"discarded={discarded_edges} edges"
    )

    return train_graph, val_graph, test_graph


def temporal_split(
    graph: nx.DiGraph,
    test_size: float = 0.2,
    val_size: float = 0.1,
    time_column: str = "step"
) -> tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph]:
    """
    Split graph by time: train on early data, test on recent data.

    This simulates real-world deployment where you train on historical
    data and evaluate on future data.

    Args:
        graph: Transaction graph
        test_size: Fraction of edges for test (by time)
        val_size: Fraction of edges for validation (by time)
        time_column: Edge attribute containing timestamp

    Returns:
        (train_graph, val_graph, test_graph)

    Example:
        Edges sorted by time:
        [t=0, t=1, t=2, t=3, t=4, t=5, t=6, t=7, t=8, t=9]

        test_size=0.2 (20%), val_size=0.1 (10%):
        Train: t=0..6 (70%)
        Val: t=7 (10%)
        Test: t=8..9 (20%)
    """
    # Get all edges with timestamps
    edges_with_time = []
    for src, dst, data in graph.edges(data=True):
        timestamp = data.get(time_column, 0)
        edges_with_time.append((src, dst, timestamp))

    # Sort by time
    edges_with_time.sort(key=lambda x: x[2])

    # Calculate split points
    n_edges = len(edges_with_time)
    n_test = int(n_edges * test_size)
    n_val = int(n_edges * val_size)
    n_train = n_edges - n_test - n_val

    # Split edges
    train_edges = [(src, dst) for src, dst, _ in edges_with_time[:n_train]]
    val_edges = [(src, dst) for src, dst, _ in edges_with_time[n_train:n_train + n_val]]
    test_edges = [(src, dst) for src, dst, _ in edges_with_time[n_train + n_val:]]

    # Create subgraphs
    train_graph = graph.edge_subgraph(train_edges).copy()
    val_graph = graph.edge_subgraph(val_edges).copy() if val_edges else nx.DiGraph()
    test_graph = graph.edge_subgraph(test_edges).copy()

    # Log statistics
    import logging
    logger = logging.getLogger(__name__)

    train_time_range = (
        min(edges_with_time[:n_train], key=lambda x: x[2])[2],
        max(edges_with_time[:n_train], key=lambda x: x[2])[2]
    ) if train_edges else (0, 0)

    test_time_range = (
        min(edges_with_time[n_train + n_val:], key=lambda x: x[2])[2],
        max(edges_with_time[n_train + n_val:], key=lambda x: x[2])[2]
    ) if test_edges else (0, 0)

    logger.info(
        f"Temporal split: train={len(train_edges)} edges (t={train_time_range[0]}-{train_time_range[1]}), "
        f"val={len(val_edges)} edges, "
        f"test={len(test_edges)} edges (t={test_time_range[0]}-{test_time_range[1]})"
    )

    return train_graph, val_graph, test_graph


def check_split_leakage(
    train_graph: nx.DiGraph,
    test_graph: nx.DiGraph,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Check for data leakage between train and test splits.

    Args:
        train_graph: Training graph
        test_graph: Test graph
        verbose: Print detailed report

    Returns:
        Dictionary with leakage statistics
    """
    train_accounts = set(train_graph.nodes())
    test_accounts = set(test_graph.nodes())

    # Check account overlap
    account_overlap = train_accounts & test_accounts
    overlap_ratio = len(account_overlap) / len(test_accounts) if test_accounts else 0.0

    # Check if any test edges have both endpoints in train
    test_edges_in_train = 0
    for src, dst in test_graph.edges():
        if src in train_accounts and dst in train_accounts:
            test_edges_in_train += 1

    leakage_stats = {
        "train_accounts": len(train_accounts),
        "test_accounts": len(test_accounts),
        "overlap_accounts": len(account_overlap),
        "overlap_ratio": overlap_ratio,
        "test_edges_fully_in_train": test_edges_in_train,
        "is_entity_disjoint": (len(account_overlap) == 0),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("SPLIT LEAKAGE CHECK")
        print("=" * 70)
        print(f"  Train accounts: {leakage_stats['train_accounts']:,}")
        print(f"  Test accounts: {leakage_stats['test_accounts']:,}")
        print(f"  Account overlap: {leakage_stats['overlap_accounts']:,} "
              f"({leakage_stats['overlap_ratio']:.1%})")
        print(f"  Test edges with both endpoints in train: {test_edges_in_train}")

        if leakage_stats['is_entity_disjoint']:
            print("\n  [OK] Entity-disjoint split - NO LEAKAGE")
        else:
            print(f"\n  [WARNING] Leakage detected! "
                  f"{leakage_stats['overlap_ratio']:.1%} of test accounts in train")
        print("=" * 70)

    return leakage_stats
