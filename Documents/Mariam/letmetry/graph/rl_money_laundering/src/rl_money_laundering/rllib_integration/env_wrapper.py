"""
RLlib Environment Wrapper

Adapts AMLDetectionEnv for RLlib by:
1. Adding config-based initialization
2. Converting observations to GraphSpace format
3. Handling environment registration
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from ray.tune.registry import register_env

from ..environment import AMLDetectionEnv
from ..features import NodeFeatureExtractor
from .graph_space import GraphSpace


_FEATURE_EXTRACTOR_CACHE: dict[str, NodeFeatureExtractor] = {}


class RLlibAMLEnv(gym.Wrapper):
    """
    RLlib-compatible wrapper for AMLDetectionEnv.

    Converts graph observations to GraphSpace format and handles
    RLlib's config-based initialization pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize wrapped environment.

        Args:
            config: RLlib environment config containing:
                - graph: NetworkX graph
                - start_node: Starting node ID
                - node_feature_extractor: Feature extraction function
                - gnn_encoder: StateEncoder instance
                - max_steps: Maximum episode length
                - max_nodes: Maximum subgraph nodes (for GraphSpace)
                - max_edges: Maximum subgraph edges (for GraphSpace)
        """
        if config is None:
            raise ValueError("RLlibAMLEnv requires a config dict")

        # Extract configuration
        graph = config["graph"]
        if "node_feature_extractor" in config:
            self.node_feature_extractor = config["node_feature_extractor"]
        else:
            cache_key = str(config.get("feature_extractor_cache_key", "default"))
            extractor = _FEATURE_EXTRACTOR_CACHE.get(cache_key)
            if extractor is None:
                extractor = NodeFeatureExtractor(
                    include_temporal=True,
                    include_network=True,
                    node_feature_dim=self.gnn_encoder.node_feature_dim,
                )
                _FEATURE_EXTRACTOR_CACHE[cache_key] = extractor
            self.node_feature_extractor = extractor
        self.gnn_encoder = config["gnn_encoder"]
        max_steps = config.get("max_steps", 50)
        max_neighbors = config.get("max_neighbors", 5)

        # Create base environment (it handles start_node selection internally)
        base_env = AMLDetectionEnv(
            graph=graph,
            max_steps=max_steps,
            max_neighbors=max_neighbors
        )

        super().__init__(base_env)

        # Graph space parameters
        self.max_nodes = config.get("max_nodes", 30)
        self.max_edges = config.get("max_edges", 200)
        node_feature_dim = self.node_feature_extractor.node_feature_dim
        context_feature_dim = self.gnn_encoder.history_dim

        # Override observation space with GraphSpace
        self.observation_space = GraphSpace(
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            node_feature_dim=node_feature_dim,
            context_feature_dim=context_feature_dim
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment and return GraphSpace observation.

        Returns:
            observation: Graph-structured observation dict
            info: Additional information
        """
        # Reset base environment
        state, info = self.env.reset(seed=seed, options=options)

        # Convert state to GraphSpace format
        graph_obs = self._state_to_graph_obs(state, info)

        return graph_obs, info

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take action and return GraphSpace observation.

        Args:
            action: Action index

        Returns:
            observation: Graph-structured observation
            reward: Reward value
            terminated: Episode terminated flag
            truncated: Episode truncated flag
            info: Additional information
        """
        # Step base environment
        state, reward, terminated, truncated, info = self.env.step(action)

        # Convert state to GraphSpace format
        graph_obs = self._state_to_graph_obs(state, info)

        return graph_obs, reward, terminated, truncated, info

    def _state_to_graph_obs(
        self,
        state: np.ndarray,
        info: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Convert flat state vector to GraphSpace observation.

        The base environment returns: [raw_node_features (10), history_features (6+)]
        NOT [GNN_embedding, history] as previously assumed.

        Edge times are extracted from graph edge attributes (time/timestamp/step)
        and padded to max_edges for temporal GNN compatibility.

        Args:
            state: Flat state vector (raw node features + history features)
            info: Info dict potentially containing graph structure

        Returns:
            Graph observation dict compatible with GraphSpace
        """
        # Extract subgraph structure from environment internals
        env = self.env.unwrapped  # Access original AMLDetectionEnv
        current_node = env.current_node
        graph = env.graph

        # Get local subgraph (same k-hop logic as StateEncoder)
        subgraph_nodes = self._get_local_subgraph(graph, current_node, k=2)

        # Cap subgraph size to prevent overflow
        if len(subgraph_nodes) > self.max_nodes:
            # Keep current_node + closest neighbors
            sorted_nodes = sorted(subgraph_nodes, key=lambda n: (n != current_node, n))
            subgraph_nodes = set(sorted_nodes[:self.max_nodes])

        subgraph = graph.subgraph(subgraph_nodes)

        # CRITICAL FIX: Ensure current_node is ALWAYS first (not random from set)
        # This ensures batch_utils.extract_first_node_embeddings() gets the right node
        node_list = [current_node] + sorted([n for n in subgraph_nodes if n != current_node])
        num_nodes = len(node_list)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Node features matrix (padded to max_nodes)
        node_features = np.zeros((self.max_nodes, self.observation_space.node_feature_dim), dtype=np.float32)
        for idx, node in enumerate(node_list):
            if idx >= self.max_nodes:
                break
            node_data = graph.nodes[node]
            features = self.node_feature_extractor.extract(node_data)
            node_features[idx] = features

        # Edge index matrix (padded to max_edges)
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_time = np.zeros((self.max_edges,), dtype=np.float32)
        edge_list = []
        edge_times = []
        for u, v in subgraph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
                edge_data = graph.get_edge_data(u, v, default={})
                edge_times.append(
                    float(
                        edge_data.get(
                            "time",
                            edge_data.get("timestamp", edge_data.get("step", 0.0))
                        )
                    )
                )

        num_edges = min(len(edge_list), self.max_edges)
        if num_edges > 0:
            edge_index[:, :num_edges] = np.array(edge_list[:num_edges], dtype=np.int64).T
            edge_time[:num_edges] = np.array(edge_times[:num_edges], dtype=np.float32)

        # Context features are history features from the state vector
        # Base env returns: [10 node features, 6+ history features]
        # History features: [num_visited, max_fraud_score, avg_amount, neighbor_ratio, step_ratio, is_flagged]
        num_node_features = 10  # balance, risk, suspicious, degrees (2), account_type (5)
        history_dim = self.gnn_encoder.history_dim

        # Extract history features from correct position
        if len(state) >= num_node_features + history_dim:
            context_features = state[num_node_features:num_node_features + history_dim].astype(np.float32)
        else:
            # Fallback if state too short
            context_features = np.zeros(history_dim, dtype=np.float32)

        obs = {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_time": edge_time,
            "num_nodes": np.array([num_nodes], dtype=np.int64),  # shape (1,) for vectorization
            "num_edges": np.array([num_edges], dtype=np.int64),  # shape (1,) for vectorization
            "context_features": context_features,
        }

        # Debug: Validate observation before returning
        is_valid = self.observation_space.contains(obs)
        if not is_valid:
            print(f"\n[DEBUG] Invalid observation generated!")
            print(f"  num_nodes: {num_nodes}, num_edges: {num_edges}")
            if num_edges > 0:
                active_edges = edge_index[:, :num_edges]
                print(f"  Edge index range: [{active_edges.min()}, {active_edges.max()}]")
                print(f"  Edge indices: {active_edges.T[:5]}")  # Show first 5 edges
                invalid_mask = (active_edges < 0) | (active_edges >= num_nodes)
                if invalid_mask.any():
                    print(f"  INVALID EDGES FOUND: {active_edges[:, invalid_mask.any(axis=0)]}")

            # Check each field individually
            for key in ['node_features', 'edge_index', 'edge_time', 'num_nodes', 'num_edges', 'context_features']:
                field_valid = self.observation_space.spaces[key].contains(obs[key])
                print(f"  {key}: valid={field_valid}, shape={obs[key].shape}, dtype={obs[key].dtype}")

        return obs

    def _get_local_subgraph(self, graph, center_node: str, k: int = 2) -> set:
        """Get k-hop neighborhood (same as StateEncoder)"""
        neighbors = {center_node}
        for _ in range(k):
            new_neighbors = set()
            for node in neighbors:
                if node in graph:
                    new_neighbors.update(graph.predecessors(node))
                    new_neighbors.update(graph.successors(node))
            neighbors.update(new_neighbors)
        return neighbors


def create_rllib_env(config: Dict[str, Any]) -> RLlibAMLEnv:
    """
    Environment creator function for RLlib registration.

    Args:
        config: Environment configuration dict

    Returns:
        Wrapped AML detection environment
    """
    return RLlibAMLEnv(config)


def register_aml_env(env_name: str = "aml_detection") -> None:
    """
    Register AML detection environment with RLlib.

    Args:
        env_name: Name to register environment under

    Example:
        >>> register_aml_env("aml_detection")
        >>> # Now can use .environment("aml_detection") in config
    """
    register_env(env_name, create_rllib_env)
