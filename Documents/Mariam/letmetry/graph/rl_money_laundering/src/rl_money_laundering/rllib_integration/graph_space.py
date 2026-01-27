"""
Graph Space for RLlib Integration

Defines a custom gymnasium space for graph-structured observations,
based on the tf-gnn-example-for-rllib approach but adapted for PyTorch Geometric.

Reference: https://github.com/kk-55/tf-gnn-example-for-rllib
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace


class GraphSpace(DictSpace):
    """
    Graph observation space for GNN-based RL agents.

    Represents variable-size graphs as nested dictionaries with:
    - node_features: (max_nodes, node_feature_dim) node feature matrix
    - edge_index: (2, max_edges) edge connectivity (COO format)
    - edge_time: (max_edges,) edge timestamps aligned with edge_index
    - num_nodes: Scalar indicating actual number of nodes
    - num_edges: Scalar indicating actual number of edges
    - context_features: Fixed-size global/history features

    This design allows RLlib to handle variable-size graphs while maintaining
    compatibility with standard gym.spaces infrastructure.
    """

    def __init__(
        self,
        max_nodes: int,
        max_edges: int,
        node_feature_dim: int,
        context_feature_dim: int,
        seed: Optional[int] = None
    ):
        """
        Initialize Graph space.

        Args:
            max_nodes: Maximum number of nodes in subgraph
            max_edges: Maximum number of edges in subgraph
            node_feature_dim: Dimension of node feature vectors
            context_feature_dim: Dimension of context/history features
            seed: Random seed for sampling
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_feature_dim = node_feature_dim
        self.context_feature_dim = context_feature_dim

        # Define component spaces
        spaces = {
            # Node features: (max_nodes, node_feature_dim)
            # Unused nodes are zero-padded
            "node_features": Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_nodes, node_feature_dim),
                dtype=np.float32
            ),

            # Edge connectivity: (2, max_edges) in COO format
            # Each column is [source_idx, target_idx]
            "edge_index": Box(
                low=0,
                high=max_nodes - 1,
                shape=(2, max_edges),
                dtype=np.int64
            ),
            # Edge timestamps aligned with edge_index columns (padded)
            "edge_time": Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_edges,),
                dtype=np.float32
            ),

            # Actual counts (for masking padding)
            # Note: shape=(1,) instead of () for vectorization compatibility
            "num_nodes": Box(low=1, high=max_nodes, shape=(1,), dtype=np.int64),
            "num_edges": Box(low=0, high=max_edges, shape=(1,), dtype=np.int64),

            # Fixed-size context features (path history, global stats)
            "context_features": Box(
                low=-np.inf,
                high=np.inf,
                shape=(context_feature_dim,),
                dtype=np.float32
            ),
        }

        super().__init__(spaces=spaces, seed=seed)

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Sample a random graph observation.

        Returns:
            Dictionary with node_features, edge_index, num_nodes, num_edges, context_features
        """
        # Sample actual sizes
        num_nodes = self.np_random.integers(1, self.max_nodes + 1)
        max_possible_edges = num_nodes * (num_nodes - 1)  # Directed graph
        num_edges = self.np_random.integers(0, min(max_possible_edges, self.max_edges) + 1)

        # Sample node features (active nodes)
        node_features = np.zeros((self.max_nodes, self.node_feature_dim), dtype=np.float32)
        node_features[:num_nodes] = self.np_random.standard_normal((num_nodes, self.node_feature_dim)).astype(np.float32)

        # Sample edge indices
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_time = np.zeros((self.max_edges,), dtype=np.float32)
        if num_edges > 0:
            # Generate random edges within active nodes
            sources = self.np_random.integers(0, num_nodes, size=num_edges)
            targets = self.np_random.integers(0, num_nodes, size=num_edges)
            edge_index[0, :num_edges] = sources
            edge_index[1, :num_edges] = targets
            edge_time[:num_edges] = self.np_random.random(num_edges).astype(np.float32)

        # Sample context features
        context_features = self.np_random.standard_normal(self.context_feature_dim).astype(np.float32)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_time": edge_time,
            "num_nodes": np.array([num_nodes], dtype=np.int64),  # shape (1,) for vectorization
            "num_edges": np.array([num_edges], dtype=np.int64),  # shape (1,) for vectorization
            "context_features": context_features,
        }

    def contains(self, x: Any) -> bool:
        """
        Check if x is a valid graph observation.

        Args:
            x: Observation to validate

        Returns:
            True if x matches the space specification
        """
        # Use parent Dict space's contains() for proper validation of shapes and dtypes
        if not super().contains(x):
            return False

        # Additional semantic validation: edge indices must be within node range
        try:
            num_edges_val = int(x["num_edges"][0])
            num_nodes_val = int(x["num_nodes"][0])
            if num_edges_val > 0:
                active_edges = x["edge_index"][:, :num_edges_val]
                if np.any(active_edges < 0) or np.any(active_edges >= num_nodes_val):
                    return False

            return True
        except (KeyError, AttributeError, TypeError, IndexError):
            return False

    def __repr__(self) -> str:
        return (
            f"GraphSpace(max_nodes={self.max_nodes}, "
            f"max_edges={self.max_edges}, "
            f"node_feature_dim={self.node_feature_dim}, "
            f"context_feature_dim={self.context_feature_dim})"
        )
