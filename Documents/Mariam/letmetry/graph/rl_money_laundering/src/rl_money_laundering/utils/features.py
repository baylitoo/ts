"""
Feature extraction utilities.

Provides consistent node and edge feature extraction across different
dataset types and components.
"""

from __future__ import annotations


import networkx as nx
import numpy as np


class NodeFeatureExtractor:
    """
    Consistent node feature extraction for different datasets.

    Handles feature extraction from NetworkX node data with fallbacks
    for missing attributes.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        default_value: float = 0.0
    ):
        """
        Initialize feature extractor.

        Args:
            feature_names: List of feature names to extract
            default_value: Default value for missing features
        """
        self.feature_names = feature_names or self._default_features()
        self.default_value = default_value

    @staticmethod
    def _default_features() -> list[str]:
        """Default feature set for AMLNet dataset."""
        return [
            "total_sent",
            "total_received",
            "num_transactions_sent",
            "num_transactions_received",
            "avg_transaction_amount",
            "max_transaction_amount",
            "min_transaction_amount",
            "transaction_velocity",
            "in_degree",
            "out_degree",
            "pagerank",
            "risk_score",
        ]

    def extract(self, node_data: dict) -> np.ndarray:
        """
        Extract features from node data.

        Args:
            node_data: Dictionary of node attributes

        Returns:
            NumPy array of features
        """
        features = []

        for feature_name in self.feature_names:
            value = node_data.get(feature_name, self.default_value)
            features.append(float(value))

        return np.array(features, dtype=np.float32)

    def extract_from_graph(
        self,
        graph: nx.DiGraph,
        node: str
    ) -> np.ndarray:
        """
        Extract features directly from graph node.

        Args:
            graph: NetworkX graph
            node: Node ID

        Returns:
            NumPy array of features
        """
        if node not in graph:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        node_data = graph.nodes[node]
        return self.extract(node_data)

    def get_feature_dim(self) -> int:
        """Get dimensionality of feature vector."""
        return len(self.feature_names)


class AMLNetFeatureExtractor(NodeFeatureExtractor):
    """Feature extractor specialized for AMLNet dataset."""

    def __init__(self):
        super().__init__(
            feature_names=[
                "total_sent",
                "total_received",
                "num_transactions_sent",
                "num_transactions_received",
                "avg_transaction_amount",
                "max_transaction_amount",
                "min_transaction_amount",
                "transaction_velocity",
                "in_degree",
                "out_degree",
                "pagerank",
                "risk_score",
            ]
        )


class EllipticFeatureExtractor(NodeFeatureExtractor):
    """Feature extractor specialized for Elliptic dataset."""

    def __init__(self, num_features: int = 166):
        """
        Initialize Elliptic feature extractor.

        Args:
            num_features: Number of features (166 for Elliptic)
        """
        # Elliptic uses numeric feature indices
        feature_names = [f"feature_{i}" for i in range(num_features)]
        super().__init__(feature_names=feature_names)


def create_feature_extractor(dataset_type: str) -> NodeFeatureExtractor:
    """
    Factory function to create appropriate feature extractor.

    Args:
        dataset_type: "amlnet" or "elliptic"

    Returns:
        Feature extractor instance
    """
    if dataset_type == "amlnet":
        return AMLNetFeatureExtractor()
    elif dataset_type == "elliptic":
        return EllipticFeatureExtractor()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


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
