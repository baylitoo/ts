"""
Temporal Graph Guardrail Environment for Reinforcement Learning

A Gymnasium-compatible environment where an RL agent traverses a transaction
or entity graph, applies guardrail-aware actions, and surfaces high-risk flows.
The AML case study instantiates this environment, but the interfaces are built
to support any temporal heterogeneous graph subject to compliance directives.
"""

from typing import Any, Dict, List, Tuple, cast
from collections import defaultdict

import gymnasium as gym
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Space


class AMLDetectionEnv(gym.Env[np.ndarray, int]):
    """
    Temporal Graph Governance Environment

    The agent starts at a seed account/entity and navigates through the transaction
    graph by following edges (transactions). The goal is to surface risky subgraphs
    while minimizing false positives and respecting guardrail directives.

    MDP Formalization:
    ------------------
    State: (current_node, visited_history, graph_context)
        - current_node: ID of current account/entity
        - visited_history: Set of previously visited nodes
        - graph_context: Local subgraph features

    Actions:
        - MOVE_TO_NEIGHBOR_i: Navigate to i-th neighbor
        - FLAG: Flag current subgraph as suspicious and end episode

    Rewards:
        - Terminal reward: Positive when guardrail criteria confirm risk, negative when rejected
        - Step penalty: -0.01 to encourage efficiency
        - Intrinsic reward: Small bonus for visiting high-risk or novel nodes
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        graph: nx.DiGraph,
        max_steps: int = 20,
        max_neighbors: int = 5,
        intrinsic_reward_weight: float = 0.1,
        fraud_reward_base: float = 10.0,
        false_alarm_penalty_base: float = -10.0,
        render_mode: str | None = None
    ):
        """
        Args:
            graph: Transaction graph (nodes=accounts, edges=transactions)
            max_steps: Maximum steps per episode before auto-termination
            max_neighbors: Maximum number of neighbors to consider per state
            intrinsic_reward_weight: Weight for intrinsic curiosity reward
            fraud_reward_base: Maximum terminal reward granted for a correct flag (confidence-scaled)
            false_alarm_penalty_base: Maximum penalty applied for an incorrect flag (confidence-scaled)
            render_mode: Rendering mode
        """
        super().__init__()

        self.graph = graph
        self.max_steps = max_steps
        self.max_neighbors = max_neighbors
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.fraud_reward_base = float(fraud_reward_base)
        self.false_alarm_penalty_base = float(false_alarm_penalty_base)
        self.render_mode = render_mode

        # Get all nodes that have outgoing edges (potential starting points)
        self.valid_start_nodes = [
            n for n in graph.nodes()
            if graph.out_degree(n) > 0
        ]

        if not self.valid_start_nodes:
            raise ValueError("Graph has no nodes with outgoing edges")

        # State space: node features + visited history encoding
        # For simplicity, we use a fixed-size feature vector
        self.node_feature_dim = 10  # account features
        self.history_dim = 6  # encoding of visited nodes (increased from 5 to 6 for is_current_flagged)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.node_feature_dim + self.history_dim,),
            dtype=np.float32
        )

        # Action space: move to neighbor (0 to max_neighbors-1) or FLAG (max_neighbors)
        discrete_action_space = spaces.Discrete(self.max_neighbors + 1)
        self.action_space = cast(Space[int], discrete_action_space)
        self._discrete_action_space = discrete_action_space
        self.FLAG_ACTION = self.max_neighbors

        # Episode state
        self.current_node: str | None = None
        self.start_node: str | None = None
        self.visited_nodes: set[str] = set()
        self.visited_edges: List[Tuple[str, str]] = []
        self.flagged_nodes: set[str] = set()  # Track nodes flagged as suspicious
        self.step_count: int = 0
        self.current_neighbors: List[str] = []
        self.episode_reward: float = 0.0

        # Hybrid reward system components
        self.node_visit_counts: Dict[str, int] = defaultdict(int)  # For curiosity
        self._fraud_nodes: set[str] | None = None  # Cache fraud nodes
        self._fraud_distances: Dict[str, int] | None = None  # Cache distances to frauds

        # Pre-compute fraud nodes and distances for efficiency
        self._precompute_fraud_info()

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        start_node = None
        if options:
            start_node = options.get("start_node")
            if start_node not in self.valid_start_nodes:
                start_node = None

        # Sample starting node (curriculum may supply one)
        if start_node is None:
            self.current_node = self.np_random.choice(self.valid_start_nodes)
        else:
            self.current_node = start_node

        self.start_node = self.current_node

        self.visited_nodes = {self.current_node}
        self.visited_edges = []
        self.flagged_nodes = set()
        self.step_count = 0
        self.episode_reward = 0.0

        # Get current neighbors
        self.current_neighbors = self._get_current_neighbors()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int, confidence: float = 0.5) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with HYBRID REWARD SYSTEM.

        Args:
            action: Action to take (0 to max_neighbors for MOVE, max_neighbors for FLAG)
            confidence: Model confidence for this action [0, 1] (default 0.5 for random)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_node is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward = 0.0
        terminated = False
        truncated = False

        # Handle FLAG action - TERMINAL (one flag per episode)
        if action == self.FLAG_ACTION:
            if self.current_node:
                # Flag the current node
                self.flagged_nodes.add(self.current_node)

                # FIXED FLAG REWARD: Large reward/penalty (+5 to +10 for TP, -5 to -10 for FP)
                reward = self._get_confidence_scaled_flag_reward(self.current_node, confidence)

            else:
                # No current node - invalid state
                reward = -5.0

            # FLAG TERMINATES EPISODE (high-stakes decision)
            terminated = True

        # Handle MOVE action
        elif 0 <= action < len(self.current_neighbors):
            # Move to selected neighbor
            next_node = self.current_neighbors[action]
            edge = (self.current_node, next_node)

            # HYBRID REWARD COMPONENT 1: Proximity to frauds
            proximity_reward = self._get_proximity_reward(next_node)

            # HYBRID REWARD COMPONENT 2: Curiosity for rare nodes
            curiosity_reward = self._get_curiosity_reward(next_node)

            # Combine hybrid rewards (scaled by intrinsic_reward_weight)
            reward = self.intrinsic_reward_weight * (proximity_reward + curiosity_reward)

            # Bonus for traversing fraud edges (still keep this for immediate feedback)
            edge_data = self.graph.get_edge_data(*edge)
            if edge_data and self._edge_is_fraud(edge_data):
                reward += self.intrinsic_reward_weight * 2.0

            # Apply per-step penalty to discourage wandering
            reward -= 0.01

            # Update state
            self.current_node = next_node
            self.visited_nodes.add(next_node)
            self.visited_edges.append(edge)
            self.step_count += 1

            # Get new neighbors
            self.current_neighbors = self._get_current_neighbors()

            # Check if stuck (no unvisited neighbors)
            if not self.current_neighbors:
                terminated = True
                reward += self._compute_terminal_reward()

        else:
            # Invalid action (neighbor index out of bounds)
            reward = -0.1  # Penalty for invalid action
            # Stay in same state

        # Check for truncation (max steps)
        if self.step_count >= self.max_steps:
            truncated = True
            reward += self._compute_terminal_reward()

        self.episode_reward += reward

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_current_neighbors(self) -> List[str]:
        """
        Get valid neighbors for current node with temporal constraints.

        Only returns neighbors that can be reached via edges that occur
        AFTER the current time (enforces temporal ordering).

        Returns up to max_neighbors neighbors, prioritizing unvisited nodes.
        """
        if self.current_node is None:
            return []

        # Determine current time from last traversed edge
        if self.visited_edges:
            # Use the timestamp of the last edge traversed
            last_edge = self.visited_edges[-1]
            current_time = self.graph.edges[last_edge].get('step', 0)
        else:
            # First move - use the earliest time from current node
            current_time = self.graph.nodes[self.current_node].get('first_step', 0)

        # Get all successors
        all_neighbors = list(self.graph.successors(self.current_node))

        # TEMPORAL CONSTRAINT: Only allow edges that happen at or after current_time
        temporal_neighbors = []
        for neighbor in all_neighbors:
            edge_time = self.graph.edges[self.current_node, neighbor].get('step', float('inf'))
            if edge_time >= current_time:
                temporal_neighbors.append(neighbor)

        # Prioritize unvisited neighbors
        unvisited = [n for n in temporal_neighbors if n not in self.visited_nodes]
        visited = [n for n in temporal_neighbors if n in self.visited_nodes]

        # Return up to max_neighbors, preferring unvisited
        neighbors = (unvisited + visited)[:self.max_neighbors]

        return neighbors

    def _get_observation(self) -> np.ndarray:
        """
        Construct state observation

        Features:
        - Current node features (account type, balance, risk score, etc.)
        - Visited history encoding (number of visited nodes, max fraud score seen, etc.)
        """
        if self.current_node is None:
            shape = self.observation_space.shape
            if shape is None:
                raise RuntimeError("Observation space shape is undefined.")
            return np.zeros(shape[0], dtype=np.float32)

        node_data = self.graph.nodes[self.current_node]

        # Node features
        account_type_encoding = self._encode_account_type(
            node_data.get("account_type", "individual")
        )
        # FIXED: Use signed_log1p to handle negative balances safely
        balance_raw = node_data.get("balance", 0.0)
        balance = np.sign(balance_raw) * np.log1p(abs(balance_raw))
        risk_score = node_data.get("risk_score", 0.0)
        is_suspicious = float(node_data.get("is_suspicious", False))
        out_degree = self.graph.out_degree(self.current_node)
        in_degree = self.graph.in_degree(self.current_node)

        node_features = np.array([
            balance,
            risk_score,
            is_suspicious,
            out_degree,
            in_degree,
            *account_type_encoding  # 5 features for one-hot encoding
        ], dtype=np.float32)

        # History features
        num_visited = len(self.visited_nodes)
        num_fraud_edges_seen = sum(
            1 for edge in self.visited_edges
            if self._edge_is_fraud(self.graph.get_edge_data(*edge, default={}))
        )
        max_fraud_score = num_fraud_edges_seen / max(len(self.visited_edges), 1)
        avg_amount_raw = np.mean([
            self.graph.get_edge_data(*edge).get("amount", 0.0)
            for edge in self.visited_edges
        ]) if self.visited_edges else 0.0
        # FIXED: Use signed_log1p for amounts
        avg_amount = np.sign(avg_amount_raw) * np.log1p(abs(avg_amount_raw))

        # Check if current node is already flagged (agent shouldn't flag twice)
        is_current_flagged = float(self.current_node in self.flagged_nodes) if self.current_node else 0.0

        history_features = np.array([
            num_visited / self.max_steps,  # Normalized
            max_fraud_score,
            avg_amount,
            len(self.current_neighbors) / self.max_neighbors,  # Normalized
            self.step_count / self.max_steps,  # Progress
            is_current_flagged  # NEW: Whether current node already flagged
        ], dtype=np.float32)

        obs = np.concatenate([node_features, history_features])

        # FEATURE WATCHDOG: Check for NaN/Inf
        if not np.all(np.isfinite(obs)):
            nan_count = np.sum(~np.isfinite(obs))
            print(f"[WARNING] Feature corruption detected: {nan_count} NaN/Inf values")
            print(f"  Node: {self.current_node}, Balance: {balance_raw}, Avg amount: {avg_amount_raw}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def _encode_account_type(self, account_type: str) -> List[float]:
        """One-hot encode account type"""
        types = ["individual", "business", "shell_company", "foreign", "crypto_exchange"]
        encoding = [0.0] * len(types)

        if account_type in types:
            encoding[types.index(account_type)] = 1.0

        return encoding

    def _compute_terminal_reward(self) -> float:
        """
        Compute terminal reward when episode ends.

        NEW DESIGN: Terminal reward is a bonus for good flagging coverage.
        - Flags are now given immediate rewards during episode (+1.0 or -5.0)
        - Terminal reward is small bonus for thoroughness
        """
        if not self.flagged_nodes:
            # No nodes flagged - neutral (already got rewards/penalties during episode)
            return 0.0

        # Count correct flags (nodes involved in fraud)
        correct_flags = sum(1 for node in self.flagged_nodes if self._is_node_fraudulent(node))

        # Bonus for high precision
        if correct_flags > 0:
            precision = correct_flags / len(self.flagged_nodes)
            # Small bonus for good precision (0.0 to 0.5)
            return 0.5 * precision
        else:
            # All flags were wrong - already paid -5.0 each, no additional penalty
            return 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state - ENHANCED EVENT LOGGING"""
        # Count fraud edges encountered
        fraud_edges = [
            edge for edge in self.visited_edges
            if self._edge_is_fraud(self.graph.get_edge_data(*edge, default={}))
        ]
        fraud_edge_encountered = len(fraud_edges) > 0

        # Compute minimum distance to fraud edge during episode
        min_dist_to_fraud = 999
        if self._fraud_distances:
            for node in self.visited_nodes:
                dist = self._fraud_distances.get(node, 999)
                min_dist_to_fraud = min(min_dist_to_fraud, dist)

        # Flag correctness
        flag_used = len(self.flagged_nodes) > 0
        flag_correct = False
        if flag_used:
            flag_correct = any(self._is_node_fraudulent(node) for node in self.flagged_nodes)

        # Step-level fraud signals for RLlib callbacks
        last_edge_is_fraud = False
        last_edge_time = None
        if self.visited_edges:
            last_edge = self.visited_edges[-1]
            last_edge_data = self.graph.get_edge_data(*last_edge, default={})
            last_edge_is_fraud = self._edge_is_fraud(last_edge_data)
            last_edge_time = last_edge_data.get(
                "time",
                last_edge_data.get("timestamp", last_edge_data.get("step"))
            )

        current_node_is_fraud = (
            self._is_node_fraudulent(self.current_node) if self.current_node is not None else False
        )
        if last_edge_time is None and self.current_node is not None:
            node_data = self.graph.nodes.get(self.current_node, {})
            last_edge_time = node_data.get("time", node_data.get("timestamp", node_data.get("step")))

        timestamp_value = float(last_edge_time) if last_edge_time is not None else float(self.step_count)

        episode_contains_fraud = any(self._is_node_fraudulent(node) for node in self.visited_nodes)
        start_was_fraud = self.start_node is not None and self._is_node_fraudulent(self.start_node)

        info = {
            "current_node": self.current_node,
            "visited_nodes": len(self.visited_nodes),
            "step_count": self.step_count,
            "num_neighbors": len(self.current_neighbors),
            "episode_reward": self.episode_reward,
            "flagged_nodes": len(self.flagged_nodes),
            "fraud_edges_found": len(fraud_edges),
            "fraud_rate": len(fraud_edges) / max(len(self.visited_edges), 1) if self.visited_edges else 0.0,
            # PER-EPISODE EVENT LOGGING (for diagnostics)
            "fraud_edge_encountered": fraud_edge_encountered,
            "encountered_fraud_edge": fraud_edge_encountered,  # Backwards compatibility
            "is_fraud_edge": last_edge_is_fraud,
            "is_fraud_node": current_node_is_fraud,
            "fraud_pattern_type": None,
            "timestamp": timestamp_value,
            "min_distance_to_fraud": min_dist_to_fraud,
            "flag_used": flag_used,
            "flag_correct": flag_correct,
            "episode_contains_fraud": episode_contains_fraud,
            "start_was_fraud": start_was_fraud,
        }

        # Add flagging accuracy
        if self.flagged_nodes:
            correct_flags = sum(1 for node in self.flagged_nodes if self._is_node_fraudulent(node))
            info["flag_precision"] = correct_flags / len(self.flagged_nodes)
            info["correct_flags"] = correct_flags
        else:
            info["flag_precision"] = 0.0
            info["correct_flags"] = 0

        return info

    def _is_node_fraudulent(self, node: str) -> bool:
        """
        Check if a node is involved in fraudulent activity.
        Uses pre-computed fraud nodes for efficiency.
        """
        if self._fraud_nodes is None:
            # Fallback if not pre-computed
            for _, _, edge_data in self.graph.out_edges(node, data=True):
                if self._edge_is_fraud(edge_data):
                    return True
            for _, _, edge_data in self.graph.in_edges(node, data=True):
                if self._edge_is_fraud(edge_data):
                    return True
            return False
        return node in self._fraud_nodes

    def _get_proximity_reward(self, node: str) -> float:
        """
        Component 1: Potential-based reward shaping (Ng et al., 1999).
        Gives directional gradient toward fraud nodes while preserving optimal policy.

        Φ(s) = -distance_to_nearest_fraud
        r'(s, a, s') = r(s, a, s') + γ * Φ(s') - Φ(s)

        Returns:
            Potential-based shaping reward (small, typically ≤ 0.05)
        """
        if self._fraud_distances is None:
            return 0.0

        # Get previous and current distances
        prev_node = self.current_node
        prev_distance = self._fraud_distances.get(prev_node, 999) if prev_node is not None else 999
        curr_distance = self._fraud_distances.get(node, 999)

        # Potential function: Φ(s) = -distance
        phi_prev = -prev_distance
        phi_curr = -curr_distance

        # Shaping reward: γ * Φ(s') - Φ(s)
        gamma = 0.99  # Discount factor
        shaping_reward = gamma * phi_curr - phi_prev

        # Scale to keep small but effective for long-range navigation
        # Increased from 0.01 to 0.05 to give stronger gradient across 200-step episodes
        return shaping_reward * 0.05

    def _get_curiosity_reward(self, node: str) -> float:
        """
        Component 2: Curiosity-driven intrinsic rewards.
        Rewards exploring rare/unvisited nodes, especially fraud nodes.
        SCALED DOWN to avoid dominating task reward.

        Returns:
            Small inverse frequency bonus (≤ 0.05)
        """
        visit_count = self.node_visit_counts[node]

        # Inverse frequency bonus (scaled to ~0.01-0.05 range)
        curiosity_bonus = 0.05 / (1.0 + visit_count)

        # Small amplify for fraud nodes (encourage finding new frauds)
        if self._is_node_fraudulent(node):
            curiosity_bonus *= 2.0  # 2x multiplier for fraud nodes

        # Update visit count
        self.node_visit_counts[node] += 1

        return curiosity_bonus

    def _get_confidence_scaled_flag_reward(self, node: str, confidence: float) -> float:
        """
        Component 3: FIXED FLAG rewards (mathematically consistent).

        TRUE POSITIVES: +(0.5-1.0)*fraud_reward_base (much higher than exploration)
        FALSE POSITIVES: -(0.5-1.0)*|false_alarm_penalty_base| (much worse than exploration)

        This makes FLAG a high-stakes decision, not a random spam action.

        Args:
            node: Node being flagged
            confidence: Model confidence in [0, 1] (from Q-values or action probabilities)

        Returns:
            Large reward/penalty that dominates exploration rewards
        """
        is_fraud = self._is_node_fraudulent(node)

        if is_fraud:
            # TRUE POSITIVE: Large reward (5.0-10.0 range)
            # More confident correct flags get higher rewards
            return self.fraud_reward_base * (0.5 + 0.5 * confidence)
        else:
            # FALSE POSITIVE: Large penalty (-5.0 to -10.0 range)
            # More confident wrong flags get harsher penalties
            return self.false_alarm_penalty_base * (0.5 + 0.5 * confidence)

    def _precompute_fraud_info(self) -> None:
        """
        Pre-compute fraud nodes and distances using BFS for efficient proximity rewards.
        This is called once during initialization to avoid repeated computation.
        """
        # Find all fraud nodes (nodes involved in fraud edges)
        self._fraud_nodes = set()
        for u, v, edge_data in self.graph.edges(data=True):
            if self._edge_is_fraud(edge_data):
                self._fraud_nodes.add(u)
                self._fraud_nodes.add(v)

        # Count distance distribution for reporting
        dist_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        print(f"\n{'=' * 80}")
        print("HYBRID REWARD SYSTEM - Initialization")
        print(f"{'=' * 80}")
        print(f"Found {len(self._fraud_nodes)} fraud nodes ({len(self._fraud_nodes)/len(self.graph.nodes)*100:.2f}% of graph)")

        # Pre-compute distances from all nodes to nearest fraud using multi-source BFS
        self._fraud_distances = {}
        if self._fraud_nodes:
            # Multi-source BFS: start from all fraud nodes simultaneously
            queue = [(fraud_node, 0) for fraud_node in self._fraud_nodes]
            visited = set(self._fraud_nodes)

            # Initialize fraud nodes with distance 0
            for fraud_node in self._fraud_nodes:
                self._fraud_distances[fraud_node] = 0
                dist_counts[0] += 1

            # BFS from all fraud nodes
            idx = 0
            while idx < len(queue):
                node, dist = queue[idx]
                idx += 1

                # Explore neighbors (both directions for undirected proximity)
                for neighbor in list(self.graph.predecessors(node)) + list(self.graph.successors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = dist + 1
                        self._fraud_distances[neighbor] = new_dist
                        queue.append((neighbor, new_dist))

                        # Track distance distribution
                        if new_dist <= 3:
                            dist_counts[new_dist] += 1

        print(f"Computed proximity distances for {len(self._fraud_distances)} nodes")
        print(f"  • Distance 0 (fraud nodes): {dist_counts[0]} nodes (+1.0 reward)")
        print(f"  • Distance 1 (adjacent): {dist_counts[1]} nodes (+0.5 reward)")
        print(f"  • Distance 2 (2-hops): {dist_counts[2]} nodes (+0.25 reward)")
        print(f"  • Distance 3+ (far): {dist_counts[3]} nodes (0.0 reward)")
        print(f"{'=' * 80}\n")

    @staticmethod
    def _edge_is_fraud(edge_data: Dict[str, Any]) -> bool:
        """Return True if edge metadata marks the transaction as fraudulent."""
        return bool(
            edge_data.get("is_fraud")
            or edge_data.get("isFraud")
            or edge_data.get("is_money_laundering")
        )

    def render(self) -> None:
        """Render current state (for debugging)"""
        if self.render_mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            print(f"Current Node: {self.current_node}")
            print(f"Visited Nodes: {len(self.visited_nodes)}")
            print(f"Visited Edges: {len(self.visited_edges)}")
            print(f"Available Neighbors: {self.current_neighbors}")
            print(f"Episode Reward: {self.episode_reward:.3f}")

    def close(self) -> None:
        """Clean up resources"""
        pass


class AMLDetectionEnvWrapper:
    """
    Wrapper for managing multiple episodes and collecting statistics
    """

    def __init__(self, env: AMLDetectionEnv):
        self.env = env
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.fraud_detection_rate: List[float] = []

    def run_episode(self, policy: Any) -> Dict[str, Any]:
        """
        Run a complete episode using given policy

        Args:
            policy: Function that maps observation to action

        Returns:
            Episode statistics
        """
        obs, info = self.env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        trajectory = []

        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            trajectory.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            })

            episode_reward += reward
            steps += 1
            obs = next_obs
            done = terminated or truncated

        # Record statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.fraud_detection_rate.append(info.get("fraud_rate", 0.0))

        return {
            "episode_reward": episode_reward,
            "episode_length": steps,
            "fraud_edges_found": info.get("fraud_edges_found", 0),
            "fraud_rate": info.get("fraud_rate", 0.0),
            "trajectory": trajectory
        }

    def get_statistics(self) -> Dict[str, float | int]:
        """Get aggregated statistics across all episodes"""
        if not self.episode_rewards:
            return {}

        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "mean_fraud_detection_rate": float(np.mean(self.fraud_detection_rate)),
            "num_episodes": len(self.episode_rewards)
        }
