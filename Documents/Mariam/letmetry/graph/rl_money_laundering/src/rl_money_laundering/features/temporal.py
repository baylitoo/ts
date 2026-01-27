"""
Temporal Feature Extraction for AML Detection

Implements temporal features from AMLNet Algorithm 2:
- Transaction velocity (transactions per time window)
- Periodicity detection (business hours, daily/weekly patterns)
- Temporal deviations from normal patterns
"""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class TemporalFeatureExtractor:
    """
    Extract temporal features for AML detection as per AMLNet Algorithm 2.

    Features include:
    - Transaction velocity (rate over time)
    - Business hour patterns (9-17h adherence)
    - Periodicity scores (weekly/monthly rhythms)
    - Temporal deviations from baseline
    """

    def __init__(
        self,
        business_hour_start: int = 9,
        business_hour_end: int = 17,
        time_window_days: int = 30
    ):
        """
        Args:
            business_hour_start: Start of business hours (default 9)
            business_hour_end: End of business hours (default 17)
            time_window_days: Time window for velocity calculation
        """
        self.business_hour_start = business_hour_start
        self.business_hour_end = business_hour_end
        self.time_window_days = time_window_days

    def extract_node_temporal_features(
        self,
        node_id: str,
        graph: nx.DiGraph,
        transactions_df: pd.DataFrame | None = None
    ) -> Dict[str, float]:
        """
        Extract temporal features for a single node.

        Args:
            node_id: Node identifier
            graph: Transaction graph
            transactions_df: Optional DataFrame with detailed transaction data

        Returns:
            Dictionary of temporal features
        """
        features = {}

        # Get node data
        if node_id not in graph:
            return self._empty_features()

        node_data = graph.nodes[node_id]

        # 1. Transaction velocity
        features['tx_velocity'] = node_data.get('tx_velocity', 0.0)

        # 2. Business hour adherence
        features['business_hour_ratio'] = node_data.get('business_hour_ratio', 0.5)

        # 3. Temporal spread (first to last transaction)
        first_step = node_data.get('first_step', 0)
        last_step = node_data.get('last_step', 0)
        features['temporal_span'] = float(last_step - first_step)

        # 4. Periodicity indicator (if we have hourly data)
        features['periodicity_score'] = self._calculate_periodicity(node_id, graph)

        return features

    def extract_edge_temporal_features(
        self,
        edge: Tuple[str, str],
        graph: nx.DiGraph,
        current_time: int | None = None
    ) -> Dict[str, float]:
        """
        Extract temporal features for an edge (transaction).

        Args:
            edge: (source, target) edge tuple
            graph: Transaction graph
            current_time: Current time step (for relative timing)

        Returns:
            Dictionary of temporal features
        """
        if edge not in graph.edges:
            return {}

        edge_data = graph.edges[edge]
        features = {}

        # Edge timestamp
        step = edge_data.get('step', 0)
        features['edge_timestamp'] = float(step)

        # Time since current (if navigating graph)
        if current_time is not None:
            features['time_delta'] = float(step - current_time)

        # Business hour indicator
        hour = edge_data.get('hour', 12)
        features['in_business_hours'] = float(
            self.business_hour_start <= hour < self.business_hour_end
        )

        # Cyclical hour encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week if available
        if 'day_of_week' in edge_data:
            dow = edge_data['day_of_week']
            features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        return features

    def calculate_transaction_velocity(
        self,
        transactions: pd.DataFrame,
        group_by: str = 'nameOrig'
    ) -> pd.Series:
        """
        Calculate transaction velocity (tx/time) for each entity.

        Args:
            transactions: DataFrame with 'step' column
            group_by: Column to group by (e.g., 'nameOrig', 'nameDest')

        Returns:
            Series with velocity values
        """
        if 'step' not in transactions.columns:
            raise ValueError("transactions must have 'step' column")

        def _velocity(group: pd.DataFrame) -> float:
            if len(group) <= 1:
                return 0.0
            time_span = float(group['step'].max() - group['step'].min())
            if time_span == 0:
                return float(len(group))
            return float(len(group) / time_span)

        return transactions.groupby(group_by, group_keys=False).apply(_velocity, include_groups=False)

    def detect_business_hour_pattern(
        self,
        transactions: pd.DataFrame,
        group_by: str = 'nameOrig'
    ) -> pd.Series:
        """
        Calculate ratio of transactions during business hours.

        Args:
            transactions: DataFrame with 'hour' column
            group_by: Column to group by

        Returns:
            Series with business hour ratios
        """
        if 'hour' not in transactions.columns:
            return pd.Series(dtype=float)

        transactions['in_business_hours'] = transactions['hour'].between(
            self.business_hour_start, self.business_hour_end - 1
        ).astype(int)

        return transactions.groupby(group_by)['in_business_hours'].mean()

    def _calculate_periodicity(
        self,
        node_id: str,
        graph: nx.DiGraph
    ) -> float:
        """
        Calculate periodicity score for a node's transactions.

        Simple heuristic based on variance in transaction timing.
        Low variance = high periodicity.

        Args:
            node_id: Node identifier
            graph: Transaction graph

        Returns:
            Periodicity score (0-1, higher = more periodic)
        """
        # Get all edges involving this node
        out_edges = [(node_id, v) for v in graph.successors(node_id)]
        in_edges = [(u, node_id) for u in graph.predecessors(node_id)]
        all_edges = out_edges + in_edges

        if len(all_edges) < 2:
            return 0.0

        # Get timestamps
        timestamps = [
            graph.edges[edge].get('step', 0)
            for edge in all_edges
        ]

        # Calculate coefficient of variation (inverse of periodicity)
        if len(timestamps) < 2:
            return 0.0

        mean_time = np.mean(timestamps)
        std_time = np.std(timestamps)

        if mean_time == 0:
            return 0.0

        cv = std_time / mean_time

        # Convert to periodicity score (lower CV = higher periodicity)
        # Normalize to 0-1 range
        periodicity = 1.0 / (1.0 + cv)

        return float(periodicity)

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict."""
        return {
            'tx_velocity': 0.0,
            'business_hour_ratio': 0.5,
            'temporal_span': 0.0,
            'periodicity_score': 0.0,
        }


def add_temporal_features_to_graph(
    graph: nx.DiGraph,
    transactions: pd.DataFrame
) -> nx.DiGraph:
    """
    Add temporal features to graph nodes.

    Adds velocity, business hour patterns, and temporal spans as node attributes.

    Args:
        graph: Transaction graph
        transactions: DataFrame with transaction data

    Returns:
        Graph with enhanced node attributes
    """
    extractor = TemporalFeatureExtractor()

    # Calculate velocities
    if 'step' in transactions.columns and 'nameOrig' in transactions.columns:
        velocities = extractor.calculate_transaction_velocity(
            transactions, 'nameOrig'
        )

        for node in graph.nodes():
            if node in velocities.index:
                graph.nodes[node]['tx_velocity'] = float(velocities[node])

    # Calculate business hour patterns
    if 'hour' in transactions.columns and 'nameOrig' in transactions.columns:
        bh_ratios = extractor.detect_business_hour_pattern(
            transactions, 'nameOrig'
        )

        for node in graph.nodes():
            if node in bh_ratios.index:
                graph.nodes[node]['business_hour_ratio'] = float(bh_ratios[node])

    # Add temporal spans
    if 'step' in transactions.columns:
        for node in graph.nodes():
            node_txs = transactions[
                (transactions['nameOrig'] == node) |
                (transactions['nameDest'] == node)
            ]

            if len(node_txs) > 0:
                graph.nodes[node]['first_step'] = int(node_txs['step'].min())
                graph.nodes[node]['last_step'] = int(node_txs['step'].max())

    return graph
