"""
Unified Node Feature Extraction

Combines amount, temporal, and network features following AMLNet Algorithm 2.
"""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np

from .base import BaseNodeFeatureExtractor
from .temporal import TemporalFeatureExtractor
from .network import NetworkFeatureExtractor


class NodeFeatureExtractor(BaseNodeFeatureExtractor):
    """
    Extract comprehensive node features for RL agent.

    Combines three feature categories from AMLNet Algorithm 2:
    1. Amount features (transaction amounts, balances)
    2. Temporal features (velocity, periodicity)
    3. Network features (centrality, clustering)
    """

    def __init__(
        self,
        include_temporal: bool = True,
        include_network: bool = True,
        node_feature_dim: int = 12
    ):
        """
        Args:
            include_temporal: Whether to include temporal features
            include_network: Whether to include network features
            node_feature_dim: Expected output feature dimension
        """
        super().__init__(node_feature_dim=node_feature_dim)
        self.include_temporal = include_temporal
        self.include_network = include_network

        self.temporal_extractor = TemporalFeatureExtractor() if include_temporal else None
        self.network_extractor = NetworkFeatureExtractor() if include_network else None

    def extract(
        self,
        node_data: Dict[str, Any],
        node_id: str | None = None,
        graph: nx.DiGraph | None = None
    ) -> np.ndarray:
        """
        Extract features from node for GNN encoding.

        Args:
            node_data: Node attributes from graph
            node_id: Node identifier (needed for network features)
            graph: Full transaction graph (needed for network features)

        Returns:
            Feature vector of shape (node_feature_dim,)
        """
        features = []

        # 1. Account type encoding (5 categories)
        account_types = ['customer', 'merchant', 'shell_company', 'foreign', 'crypto_exchange']
        node_type = node_data.get('node_type', node_data.get('type', 'customer'))
        type_encoding = [1.0 if node_type == t else 0.0 for t in account_types]
        features.extend(type_encoding)

        # 2. Amount features (log-scaled)
        features.append(np.log1p(node_data.get('total_sent', 0.0)))
        features.append(np.log1p(node_data.get('total_received', 0.0)))
        features.append(np.log1p(abs(node_data.get('balance', 0.0))))

        # 3. Transaction counts
        features.append(float(node_data.get('num_transactions_sent', 0)))
        features.append(float(node_data.get('num_transactions_received', 0)))

        # 4. Risk indicators
        features.append(node_data.get('risk_score', 0.0))
        features.append(float(node_data.get('is_suspicious', False)))

        # 5. Temporal features (from AMLNet Algorithm 2)
        if self.include_temporal:
            features.append(node_data.get('tx_velocity', 0.0))
            features.append(node_data.get('business_hour_ratio', 0.5))
            features.append(node_data.get('periodicity_score', 0.0))

        # 6. Network features (from AMLNet Algorithm 2)
        if self.include_network:
            features.append(node_data.get('degree_centrality', 0.0))
            features.append(node_data.get('clustering_coefficient', 0.0))
            features.append(node_data.get('in_degree', 0.0) / 100.0)  # Normalized
            features.append(node_data.get('out_degree', 0.0) / 100.0)  # Normalized

        # Convert to numpy array
        feature_array = np.array(features, dtype=np.float32)

        # Pad or truncate to expected dimension
        if len(feature_array) < self.node_feature_dim:
            # Pad with zeros
            feature_array = np.pad(
                feature_array,
                (0, self.node_feature_dim - len(feature_array)),
                mode='constant'
            )
        elif len(feature_array) > self.node_feature_dim:
            # Truncate
            feature_array = feature_array[:self.node_feature_dim]

        return feature_array

    def get_feature_names(self) -> list[str]:
        """
        Get names of features in order.

        Returns:
            List of feature names
        """
        names = [
            # Account type (5)
            'type_customer', 'type_merchant', 'type_shell', 'type_foreign', 'type_crypto',
            # Amount features (3)
            'log_total_sent', 'log_total_received', 'log_balance',
            # Transaction counts (2)
            'num_tx_sent', 'num_tx_received',
            # Risk (2)
            'risk_score', 'is_suspicious',
        ]

        if self.include_temporal:
            names.extend([
                'tx_velocity',
                'business_hour_ratio',
                'periodicity_score'
            ])

        if self.include_network:
            names.extend([
                'degree_centrality',
                'clustering_coefficient',
                'in_degree_norm',
                'out_degree_norm'
            ])

        return names[:self.node_feature_dim]

