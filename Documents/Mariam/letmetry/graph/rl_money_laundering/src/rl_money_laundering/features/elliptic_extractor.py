"""
Elliptic Dataset Feature Extractor

Extracts features from Elliptic dataset nodes which have 166 raw features
plus 6 engineered features (temporal + statistics).
"""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np

from .base import BaseNodeFeatureExtractor


class EllipticNodeFeatureExtractor(BaseNodeFeatureExtractor):
    """
    Extract node features from Elliptic dataset.

    Elliptic nodes have:
    - 166 original features (feature_1 ... feature_166)
    - 6 engineered features: time_tx_density, time_illicit_ratio,
                             local_mean, local_std, agg_mean, agg_std
    Total: 172 features
    """

    def __init__(self, node_feature_dim: int = 172):
        """
        Args:
            node_feature_dim: Expected output feature dimension (default 172)
        """
        super().__init__(node_feature_dim=node_feature_dim)

        # Define feature order
        self.feature_keys = [f'feature_{i}' for i in range(1, 167)]  # 166 features
        self.engineered_keys = [
            'time_tx_density',
            'time_illicit_ratio',
            'local_mean',
            'local_std',
            'agg_mean',
            'agg_std'
        ]

    def extract(
        self,
        node_data: Dict[str, Any],
        node_id: str | None = None,
        graph: nx.DiGraph | None = None
    ) -> np.ndarray:
        """
        Extract features from Elliptic node.

        Args:
            node_data: Node attributes from graph (must contain feature_1...feature_166)
            node_id: Node identifier (unused for Elliptic)
            graph: Full transaction graph (unused for Elliptic)

        Returns:
            Feature vector of shape (node_feature_dim,) = 172 features
        """
        features = []

        # Extract 166 original features
        for key in self.feature_keys:
            features.append(float(node_data.get(key, 0.0)))

        # Extract 6 engineered features
        for key in self.engineered_keys:
            features.append(float(node_data.get(key, 0.0)))

        # Convert to numpy array
        feature_array = np.array(features, dtype=np.float32)

        # Verify dimension
        if len(feature_array) != self.node_feature_dim:
            raise ValueError(
                f"Expected {self.node_feature_dim} features, "
                f"got {len(feature_array)}"
            )

        return feature_array

    def get_feature_names(self) -> list[str]:
        """Return list of feature names in extraction order."""
        return self.feature_keys + self.engineered_keys


__all__ = ["EllipticNodeFeatureExtractor"]
