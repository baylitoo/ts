"""
Base classes for feature extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import networkx as nx
import numpy as np


class BaseNodeFeatureExtractor(ABC):
    """
    Abstract base class for node feature extraction.

    All feature extractors must implement the extract() method
    to convert graph node attributes into numpy feature vectors.
    """

    def __init__(self, node_feature_dim: int):
        """
        Args:
            node_feature_dim: Expected output feature dimension
        """
        self.node_feature_dim = node_feature_dim

    @abstractmethod
    def extract(
        self,
        node_data: Dict[str, Any],
        node_id: str | None = None,
        graph: nx.DiGraph | None = None
    ) -> np.ndarray:
        """
        Extract features from a graph node.

        Args:
            node_data: Node attributes dictionary from graph.nodes[node_id]
            node_id: Node identifier (optional, needed for network features)
            graph: Full transaction graph (optional, needed for network features)

        Returns:
            Feature vector of shape (node_feature_dim,)
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return list of feature names in extraction order.

        Returns:
            List of feature names corresponding to the output vector
        """
        pass

    def __call__(
        self,
        node_data: Dict[str, Any],
        node_id: str | None = None,
        graph: nx.DiGraph | None = None
    ) -> np.ndarray:
        """Allow extractor to be called as a function."""
        return self.extract(node_data, node_id, graph)


__all__ = ["BaseNodeFeatureExtractor"]
