"""
Network Feature Extraction for AML Detection

Implements network/graph features from AMLNet Algorithm 2:
- Degree centrality
- Betweenness centrality
- Clustering coefficient
- Community detection features
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np


class NetworkFeatureExtractor:
    """
    Extract network/graph features for AML detection as per AMLNet Algorithm 2.

    Features include:
    - Centrality measures (degree, betweenness, closeness)
    - Clustering coefficient
    - Community structure features
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Args:
            cache_enabled: Whether to cache expensive computations
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Dict[str, float]] = {}

    def extract_node_network_features(
        self,
        node_id: str,
        graph: nx.DiGraph,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Extract network features for a single node.

        Args:
            node_id: Node identifier
            graph: Transaction graph
            use_cache: Whether to use cached centrality values

        Returns:
            Dictionary of network features
        """
        if node_id not in graph:
            return self._empty_features()

        features = {}

        # 1. Degree features
        features['in_degree'] = float(graph.in_degree(node_id))
        features['out_degree'] = float(graph.out_degree(node_id))
        features['total_degree'] = float(graph.in_degree(node_id) + graph.out_degree(node_id))

        # 2. Degree centrality (normalized)
        if use_cache and 'degree_centrality' in self._cache:
            degree_cent = self._cache['degree_centrality']
        else:
            degree_cent = nx.degree_centrality(graph)
            if self.cache_enabled:
                self._cache['degree_centrality'] = degree_cent

        features['degree_centrality'] = degree_cent.get(node_id, 0.0)

        # 3. Clustering coefficient
        try:
            # For directed graphs, use the underlying undirected graph
            undirected = graph.to_undirected()
            features['clustering_coefficient'] = nx.clustering(undirected, node_id)
        except (nx.NetworkXError, ZeroDivisionError):
            features['clustering_coefficient'] = 0.0

        # 4. Betweenness centrality (expensive, use sparingly)
        # Only compute for smaller graphs or when explicitly requested
        if len(graph.nodes()) < 1000:
            if use_cache and 'betweenness_centrality' in self._cache:
                between_cent = self._cache['betweenness_centrality']
            else:
                between_cent = nx.betweenness_centrality(graph)
                if self.cache_enabled:
                    self._cache['betweenness_centrality'] = between_cent

            features['betweenness_centrality'] = between_cent.get(node_id, 0.0)
        else:
            features['betweenness_centrality'] = 0.0

        return features

    def extract_local_network_features(
        self,
        node_id: str,
        graph: nx.DiGraph,
        k_hops: int = 2
    ) -> Dict[str, float]:
        """
        Extract features from local k-hop neighborhood.

        Args:
            node_id: Node identifier
            graph: Transaction graph
            k_hops: Number of hops for local neighborhood

        Returns:
            Dictionary of local network features
        """
        if node_id not in graph:
            return {}

        features = {}

        # Get k-hop subgraph
        neighbors = self._get_k_hop_neighbors(node_id, graph, k_hops)
        subgraph = graph.subgraph(neighbors)

        # Local graph statistics
        features['local_num_nodes'] = float(len(subgraph.nodes()))
        features['local_num_edges'] = float(len(subgraph.edges()))

        if len(subgraph.nodes()) > 1:
            features['local_density'] = nx.density(subgraph)
        else:
            features['local_density'] = 0.0

        # Neighbor diversity (out-degree distribution)
        out_degrees = [graph.out_degree(n) for n in graph.successors(node_id)]
        if out_degrees:
            features['neighbor_out_degree_mean'] = float(np.mean(out_degrees))
            features['neighbor_out_degree_std'] = float(np.std(out_degrees))
        else:
            features['neighbor_out_degree_mean'] = 0.0
            features['neighbor_out_degree_std'] = 0.0

        return features

    def _get_k_hop_neighbors(
        self,
        node_id: str,
        graph: nx.DiGraph,
        k: int
    ) -> set[str]:
        """
        Get all nodes within k hops of node_id.

        Args:
            node_id: Starting node
            graph: Transaction graph
            k: Number of hops

        Returns:
            Set of neighbor nodes
        """
        neighbors = {node_id}

        for _ in range(k):
            new_neighbors = set()
            for node in neighbors:
                if node in graph:
                    new_neighbors.update(graph.predecessors(node))
                    new_neighbors.update(graph.successors(node))
            neighbors.update(new_neighbors)

        return neighbors

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict."""
        return {
            'in_degree': 0.0,
            'out_degree': 0.0,
            'total_degree': 0.0,
            'degree_centrality': 0.0,
            'clustering_coefficient': 0.0,
            'betweenness_centrality': 0.0,
        }

    def clear_cache(self) -> None:
        """Clear cached centrality computations."""
        self._cache.clear()


def add_network_features_to_graph(
    graph: nx.DiGraph,
    compute_expensive: bool = False
) -> nx.DiGraph:
    """
    Add network features to graph nodes.

    Args:
        graph: Transaction graph
        compute_expensive: Whether to compute expensive features (betweenness)

    Returns:
        Graph with enhanced node attributes
    """
    # Compute once for all nodes
    degree_centrality = nx.degree_centrality(graph)

    if compute_expensive and len(graph.nodes()) < 5000:
        betweenness_centrality = nx.betweenness_centrality(graph)
    else:
        betweenness_centrality = {node: 0.0 for node in graph.nodes()}

    undirected = graph.to_undirected()
    clustering_coeffs = nx.clustering(undirected)

    # Add to nodes
    for node in graph.nodes():
        graph.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0.0)
        graph.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0.0)
        graph.nodes[node]['clustering_coefficient'] = clustering_coeffs.get(node, 0.0)
        graph.nodes[node]['in_degree'] = float(graph.in_degree(node))
        graph.nodes[node]['out_degree'] = float(graph.out_degree(node))

    return graph
