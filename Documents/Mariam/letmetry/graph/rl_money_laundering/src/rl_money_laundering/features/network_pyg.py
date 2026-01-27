"""
PyG-Based Network Feature Extraction

Network feature computation using PyTorch Geometric.
"""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import torch
from torch_geometric.utils import degree  # type: ignore[import-untyped]


def networkx_to_edge_index(graph: nx.DiGraph) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Convert NetworkX graph to PyG edge_index format.

    Args:
        graph: NetworkX directed graph

    Returns:
        edge_index: Tensor of shape (2, num_edges)
        node_mapping: Dict mapping node IDs to indices
    """
    # Create node index mapping
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Convert edges to index pairs
    edge_list = []
    for src, dst in graph.edges():
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        edge_list.append([src_idx, dst_idx])

    if not edge_list:
        # Empty graph
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index, node_to_idx


def compute_network_features_pyg(
    graph: nx.DiGraph,
    device: str = 'cpu'
) -> dict[str, np.ndarray]:
    """
    Compute network features using PyG.

    Features computed:
    - degree_centrality: Normalized degree
    - clustering_coefficient: Local clustering
    - in_degree: Number of incoming edges
    - out_degree: Number of outgoing edges

    Args:
        graph: NetworkX directed graph
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary mapping feature names to numpy arrays
    """
    num_nodes = len(graph.nodes())

    if num_nodes == 0:
        return {
            'degree_centrality': np.array([]),
            'clustering_coefficient': np.array([]),
            'in_degree': np.array([]),
            'out_degree': np.array([]),
        }

    # Convert to PyG format
    edge_index, _ = networkx_to_edge_index(graph)
    edge_index = edge_index.to(device)

    # Compute degrees (O(m) where m = number of edges)
    row, col = edge_index
    in_deg = degree(col, num_nodes=num_nodes, dtype=torch.float)
    out_deg = degree(row, num_nodes=num_nodes, dtype=torch.float)

    # Degree centrality: normalized degree (in + out) / (2 * (n - 1))
    if num_nodes > 1:
        deg_centrality = (in_deg + out_deg) / (2 * (num_nodes - 1))
    else:
        deg_centrality = torch.zeros(num_nodes)

    # Clustering coefficient: use NetworkX for now (PyG doesn't have this in torch_geometric.nn)
    # For directed graphs, convert to undirected first
    undirected_graph = graph.to_undirected()
    clustering_dict = nx.clustering(undirected_graph)

    # Convert to tensor in same order as nodes
    node_list = list(graph.nodes())
    clustering_values = [clustering_dict.get(node, 0.0) for node in node_list]
    clustering = torch.tensor(clustering_values, dtype=torch.float)

    # Convert to numpy
    return {
        'degree_centrality': deg_centrality.cpu().numpy(),
        'clustering_coefficient': clustering.cpu().numpy(),
        'in_degree': in_deg.cpu().numpy(),
        'out_degree': out_deg.cpu().numpy(),
    }


def add_network_features_pyg(
    graph: nx.DiGraph,
    device: str = 'cpu'
) -> nx.DiGraph:
    """
    Add network features to graph nodes using PyG.

    Args:
        graph: NetworkX directed graph
        device: 'cpu' or 'cuda'

    Returns:
        Graph with network features added to nodes
    """
    # Compute features with PyG
    features = compute_network_features_pyg(graph, device=device)

    # Add features to NetworkX nodes
    node_list = list(graph.nodes())
    for idx, node in enumerate(node_list):
        graph.nodes[node]['degree_centrality'] = float(features['degree_centrality'][idx])
        graph.nodes[node]['clustering_coefficient'] = float(features['clustering_coefficient'][idx])
        graph.nodes[node]['in_degree'] = float(features['in_degree'][idx])
        graph.nodes[node]['out_degree'] = float(features['out_degree'][idx])

    return graph
