"""
Feature extraction utilities.

Provides consistent node and edge feature extraction across different
dataset types and components.
"""

from __future__ import annotations


import networkx as nx
import numpy as np

# Canonical implementations live in the features package; re-export for
# backward compatibility so that `from rl_money_laundering.utils.features
# import NodeFeatureExtractor` continues to work.
from ..features import NodeFeatureExtractor  # noqa: F401

__all__ = [
    "NodeFeatureExtractor",
    "extract_edge_features",
    "compute_graph_features",
    "normalize_features",
    "log_transform",
]


def extract_edge_features(edge_data: dict) -> np.ndarray:
    """
    Extract features from edge data.

    Args:
        edge_data: Dictionary of edge attributes

    Returns:
        NumPy array of edge features
    """
    features = [
        edge_data.get("amount", 0.0),
        edge_data.get("timestamp", 0.0),
        edge_data.get("transaction_type", 0.0),
    ]

    return np.array(features, dtype=np.float32)


def compute_graph_features(graph: nx.DiGraph, node: str) -> dict[str, float]:
    """
    Compute graph-based features for a node.

    Args:
        graph: NetworkX graph
        node: Node ID

    Returns:
        Dictionary of computed features
    """
    features = {}

    # Basic degree features
    features["in_degree"] = graph.in_degree(node)
    features["out_degree"] = graph.out_degree(node)
    features["total_degree"] = features["in_degree"] + features["out_degree"]

    # Neighbor statistics
    in_neighbors = list(graph.predecessors(node))
    out_neighbors = list(graph.successors(node))

    # Average neighbor degree
    if in_neighbors:
        avg_in_neighbor_degree = np.mean([graph.in_degree(n) for n in in_neighbors])
        features["avg_in_neighbor_degree"] = avg_in_neighbor_degree
    else:
        features["avg_in_neighbor_degree"] = 0.0

    if out_neighbors:
        avg_out_neighbor_degree = np.mean([graph.out_degree(n) for n in out_neighbors])
        features["avg_out_neighbor_degree"] = avg_out_neighbor_degree
    else:
        features["avg_out_neighbor_degree"] = 0.0

    return features


def normalize_features(features: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.

    Args:
        features: Feature array (n_samples, n_features)
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    # Avoid division by zero
    std = np.where(std < epsilon, 1.0, std)

    normalized = (features - mean) / std

    return normalized


def log_transform(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply log transformation to handle skewed distributions.

    Args:
        values: Input values
        epsilon: Small value to avoid log(0)

    Returns:
        Log-transformed values
    """
    return np.log(values + epsilon)
