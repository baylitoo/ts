"""
Multi-Agent Environment for AML Detection

This module implements a multi-agent environment where:
1. Detector Agent: Navigates the transaction graph to identify fraud
2. Adversary Agent: Modifies the graph to evade detection (planned)

The environment supports both single-agent (detector only) and two-agent
(detector vs adversary) modes for robust training.

Architecture:
- Extends PettingZoo's ParallelEnv for multi-agent compatibility
- Wraps the base AMLDetectionEnv with multi-agent logic
- Supports RLlib's multi-agent training interface

Key Features:
- Turn-based or simultaneous agent actions
- Shared or separate reward structures
- Adversary action space: edge masking, feature perturbation
- Safety constraints on adversary actions (budget limits)

References:
- PettingZoo: https://pettingzoo.farama.org/
- RLlib Multi-Agent: https://docs.ray.io/en/latest/rllib/rllib-env.html
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Space

logger = logging.getLogger(__name__)


@dataclass
class DetectorAdversaryConfig:
    """Configuration for multi-agent detector-adversary environment.

    Args:
        # Environment settings
        max_steps: Maximum steps per episode
        max_neighbors: Maximum neighbors for detector action space
        turn_based: Whether agents take turns (True) or act simultaneously

        # Detector settings
        detector_intrinsic_reward_weight: Weight for detector's intrinsic rewards
        detector_fraud_reward: Reward for correct fraud detection
        detector_false_alarm_penalty: Penalty for false positives

        # Adversary settings (planned)
        enable_adversary: Whether to enable adversary agent
        adversary_budget: Maximum adversary actions per episode
        adversary_action_types: Types of adversary actions allowed
        adversary_perturbation_magnitude: Maximum feature perturbation

        # Reward structure
        zero_sum: Whether rewards are zero-sum (detector_reward = -adversary_reward)
        adversary_detection_penalty: Penalty when adversary's fraud is detected
        adversary_evasion_reward: Reward when adversary evades detection

        # Safety constraints
        adversary_max_edges_modified: Maximum edges adversary can modify
        adversary_max_features_perturbed: Maximum features to perturb
        preserve_graph_connectivity: Prevent adversary from disconnecting graph
    """

    # Environment settings
    max_steps: int = 50
    max_neighbors: int = 5
    turn_based: bool = True

    # Detector settings
    detector_intrinsic_reward_weight: float = 0.1
    detector_fraud_reward: float = 10.0
    detector_false_alarm_penalty: float = -10.0

    # Adversary settings
    enable_adversary: bool = False  # Disabled by default for MVP
    adversary_budget: int = 5
    adversary_action_types: List[str] = field(
        default_factory=lambda: ["mask_edge", "perturb_amount", "add_noise"]
    )
    adversary_perturbation_magnitude: float = 0.1

    # Reward structure
    zero_sum: bool = True
    adversary_detection_penalty: float = -5.0
    adversary_evasion_reward: float = 5.0

    # Safety constraints
    adversary_max_edges_modified: int = 10
    adversary_max_features_perturbed: int = 5
    preserve_graph_connectivity: bool = True


class MultiAgentAMLEnvironment(gym.Env):
    """
    Multi-Agent AML Detection Environment.

    This environment supports both single-agent (detector only) and
    multi-agent (detector + adversary) modes.

    Agent Roles:
    - detector: Navigates graph, flags suspicious nodes/edges
    - adversary: Modifies graph to evade detection (optional)

    Observation Spaces:
    - detector: Current node features + history + graph context
    - adversary: Full graph view + detector's recent actions

    Action Spaces:
    - detector: Move to neighbor (0 to max_neighbors-1) or FLAG (max_neighbors)
    - adversary: (edge_id, action_type, magnitude) or NO_OP

    The environment tracks both agents' rewards separately and provides
    combined episode statistics for training.
    """

    metadata = {"render_modes": ["human"], "name": "multi_agent_aml"}

    DETECTOR_AGENT = "detector"
    ADVERSARY_AGENT = "adversary"

    def __init__(
        self,
        graph: nx.DiGraph,
        config: Optional[DetectorAdversaryConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize multi-agent environment.

        Args:
            graph: Transaction graph (nodes=accounts, edges=transactions)
            config: Environment configuration
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config or DetectorAdversaryConfig()
        self.render_mode = render_mode

        # Store original graph (adversary modifies a copy)
        self._original_graph = graph
        self.graph = graph.copy()

        # Agent list
        self.possible_agents = [self.DETECTOR_AGENT]
        if self.config.enable_adversary:
            self.possible_agents.append(self.ADVERSARY_AGENT)
        self.agents = self.possible_agents.copy()

        # Valid start nodes for detector
        self.valid_start_nodes = [
            n for n in self.graph.nodes() if self.graph.out_degree(n) > 0
        ]
        if not self.valid_start_nodes:
            raise ValueError("Graph has no nodes with outgoing edges")

        # Pre-compute fraud information
        self._fraud_nodes: Set[str] = set()
        self._fraud_edges: Set[Tuple[str, str]] = set()
        self._precompute_fraud_info()

        # Build observation and action spaces
        self._build_spaces()

        # Episode state
        self._reset_episode_state()

        # Random number generator
        self._np_random: Optional[np.random.Generator] = None

    def _precompute_fraud_info(self) -> None:
        """Pre-compute fraud nodes and edges."""
        self._fraud_nodes = set()
        self._fraud_edges = set()

        for u, v, data in self.graph.edges(data=True):
            if self._edge_is_fraud(data):
                self._fraud_nodes.add(u)
                self._fraud_nodes.add(v)
                self._fraud_edges.add((u, v))

        logger.info(
            f"Graph has {len(self._fraud_nodes)} fraud nodes, "
            f"{len(self._fraud_edges)} fraud edges"
        )

    def _build_spaces(self) -> None:
        """Build observation and action spaces for each agent."""
        # Detector spaces
        self.node_feature_dim = 10
        self.history_dim = 6
        detector_obs_dim = self.node_feature_dim + self.history_dim

        detector_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(detector_obs_dim,),
            dtype=np.float32,
        )

        # Detector action: move to neighbor or FLAG
        detector_action_space = spaces.Discrete(self.config.max_neighbors + 1)
        self.FLAG_ACTION = self.config.max_neighbors

        # Store detector spaces
        self._detector_obs_space = detector_obs_space
        self._detector_action_space = detector_action_space

        # Adversary spaces (if enabled)
        if self.config.enable_adversary:
            # Adversary observes: graph summary + detector state
            adversary_obs_dim = 50  # Graph features + detector history
            adversary_obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(adversary_obs_dim,),
                dtype=np.float32,
            )

            # Adversary action: (edge_index, action_type, magnitude) or NO_OP
            # Simplified: discrete actions for MVP
            num_adversary_actions = len(self.config.adversary_action_types) + 1  # +1 for NO_OP
            adversary_action_space = spaces.Discrete(num_adversary_actions)

            self._adversary_obs_space = adversary_obs_space
            self._adversary_action_space = adversary_action_space
        else:
            self._adversary_obs_space = None
            self._adversary_action_space = None

        # Combined spaces for gym.Env interface (detector-centric)
        self.observation_space = detector_obs_space
        self.action_space = detector_action_space

    def _reset_episode_state(self) -> None:
        """Reset all episode-specific state."""
        # Detector state
        self.current_node: Optional[str] = None
        self.start_node: Optional[str] = None
        self.visited_nodes: Set[str] = set()
        self.visited_edges: List[Tuple[str, str]] = []
        self.flagged_nodes: Set[str] = set()
        self.flagged_edges: Set[Tuple[str, str]] = set()
        self.step_count: int = 0
        self.current_neighbors: List[str] = []

        # Adversary state
        self.adversary_actions_taken: int = 0
        self.modified_edges: Set[Tuple[str, str]] = set()
        self.perturbed_features: Dict[str, Dict[str, float]] = {}

        # Rewards
        self.detector_episode_reward: float = 0.0
        self.adversary_episode_reward: float = 0.0

        # Episode data for judge
        self.episode_data: Dict[str, Any] = {
            "visited_nodes": [],
            "visited_edges": [],
            "flagged_nodes": [],
            "actions": [],
            "node_features": {},
            "edge_features": {},
            "metadata": {},
        }

    @property
    def np_random(self) -> np.random.Generator:
        """Get random number generator."""
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and return initial observation.

        Args:
            seed: Random seed
            options: Reset options (e.g., start_node)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Reset graph to original (undo adversary modifications)
        self.graph = self._original_graph.copy()

        # Reset episode state
        self._reset_episode_state()

        # Determine start node
        start_node = None
        if options:
            start_node = options.get("start_node")
            if start_node not in self.valid_start_nodes:
                start_node = None

        if start_node is None:
            self.current_node = self.np_random.choice(self.valid_start_nodes)
        else:
            self.current_node = start_node

        self.start_node = self.current_node
        self.visited_nodes.add(self.current_node)
        self.episode_data["visited_nodes"].append(self.current_node)

        # Get initial neighbors
        self.current_neighbors = self._get_current_neighbors()

        # Store node features
        self._store_node_features(self.current_node)

        # Build observation
        obs = self._get_detector_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
        confidence: float = 0.5,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        For single-agent mode (detector only), this is the standard step.
        For multi-agent mode, this handles the detector's action.

        Args:
            action: Detector action
            confidence: Action confidence (for reward scaling)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.current_node is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward = 0.0
        terminated = False
        truncated = False

        # Record action
        self.episode_data["actions"].append(action)

        # Handle FLAG action
        if action == self.FLAG_ACTION:
            reward, flag_info = self._handle_flag_action(confidence)
            terminated = True  # FLAG terminates episode

        # Handle MOVE action
        elif 0 <= action < len(self.current_neighbors):
            reward, move_info = self._handle_move_action(action)

            # Check if stuck
            if not self.current_neighbors:
                terminated = True
                reward += self._compute_terminal_reward()

        else:
            # Invalid action
            reward = -0.1

        # Increment step counter
        self.step_count += 1

        # Check truncation
        if self.step_count >= self.config.max_steps:
            truncated = True
            reward += self._compute_terminal_reward()

        # Update episode reward
        self.detector_episode_reward += reward

        # Build observation and info
        obs = self._get_detector_observation()
        info = self._get_info()

        # Finalize episode data if done
        if terminated or truncated:
            self._finalize_episode_data()

        return obs, reward, terminated, truncated, info

    def _handle_flag_action(self, confidence: float) -> Tuple[float, Dict[str, Any]]:
        """Handle FLAG action from detector."""
        info: Dict[str, Any] = {"action_type": "flag"}

        if self.current_node is None:
            return -5.0, {"error": "no_current_node"}

        # Flag the current node
        self.flagged_nodes.add(self.current_node)
        self.episode_data["flagged_nodes"].append(self.current_node)

        # Check if correct
        is_fraud = self.current_node in self._fraud_nodes

        if is_fraud:
            # True positive
            reward = self.config.detector_fraud_reward * (0.5 + 0.5 * confidence)
            info["flag_result"] = "true_positive"

            # Adversary penalty (if enabled and zero-sum)
            if self.config.enable_adversary and self.config.zero_sum:
                self.adversary_episode_reward += self.config.adversary_detection_penalty
        else:
            # False positive
            reward = self.config.detector_false_alarm_penalty * (0.5 + 0.5 * confidence)
            info["flag_result"] = "false_positive"

            # Adversary reward (if enabled)
            if self.config.enable_adversary:
                self.adversary_episode_reward += abs(reward) * 0.5

        info["is_fraud"] = is_fraud
        return reward, info

    def _handle_move_action(self, action: int) -> Tuple[float, Dict[str, Any]]:
        """Handle MOVE action from detector."""
        info: Dict[str, Any] = {"action_type": "move"}

        next_node = self.current_neighbors[action]
        edge = (self.current_node, next_node)

        # Compute intrinsic rewards
        reward = 0.0

        # Proximity reward
        proximity_reward = self._get_proximity_reward(next_node)
        reward += self.config.detector_intrinsic_reward_weight * proximity_reward

        # Curiosity reward
        curiosity_reward = self._get_curiosity_reward(next_node)
        reward += self.config.detector_intrinsic_reward_weight * curiosity_reward

        # Bonus for traversing fraud edges
        edge_data = self.graph.get_edge_data(*edge, default={})
        if self._edge_is_fraud(edge_data):
            reward += self.config.detector_intrinsic_reward_weight * 2.0
            info["traversed_fraud_edge"] = True

        # Step penalty
        reward -= 0.01

        # Update state
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        self.visited_edges.append(edge)
        self.episode_data["visited_nodes"].append(next_node)
        self.episode_data["visited_edges"].append(edge)

        # Store features
        self._store_node_features(next_node)
        self._store_edge_features(edge)

        # Get new neighbors
        self.current_neighbors = self._get_current_neighbors()

        info["next_node"] = next_node
        info["reward_components"] = {
            "proximity": proximity_reward,
            "curiosity": curiosity_reward,
            "step_penalty": -0.01,
        }

        return reward, info

    def _get_current_neighbors(self) -> List[str]:
        """Get valid neighbors for current node with temporal constraints."""
        if self.current_node is None:
            return []

        # Get current time
        if self.visited_edges:
            last_edge = self.visited_edges[-1]
            current_time = self.graph.edges[last_edge].get("step", 0)
        else:
            current_time = self.graph.nodes[self.current_node].get("first_step", 0)

        # Get successors
        all_neighbors = list(self.graph.successors(self.current_node))

        # Filter by temporal constraint
        temporal_neighbors = []
        for neighbor in all_neighbors:
            edge_time = self.graph.edges[self.current_node, neighbor].get(
                "step", float("inf")
            )
            if edge_time >= current_time:
                temporal_neighbors.append(neighbor)

        # Prioritize unvisited
        unvisited = [n for n in temporal_neighbors if n not in self.visited_nodes]
        visited = [n for n in temporal_neighbors if n in self.visited_nodes]

        return (unvisited + visited)[: self.config.max_neighbors]

    def _get_detector_observation(self) -> np.ndarray:
        """Build observation for detector agent."""
        if self.current_node is None:
            return np.zeros(self.node_feature_dim + self.history_dim, dtype=np.float32)

        node_data = self.graph.nodes[self.current_node]

        # Node features
        account_type_encoding = self._encode_account_type(
            node_data.get("account_type", "individual")
        )
        balance_raw = node_data.get("balance", 0.0)
        balance = np.sign(balance_raw) * np.log1p(abs(balance_raw))
        risk_score = node_data.get("risk_score", 0.0)
        is_suspicious = float(node_data.get("is_suspicious", False))
        out_degree = self.graph.out_degree(self.current_node)
        in_degree = self.graph.in_degree(self.current_node)

        node_features = np.array(
            [balance, risk_score, is_suspicious, out_degree, in_degree]
            + account_type_encoding,
            dtype=np.float32,
        )

        # History features
        num_visited = len(self.visited_nodes)
        num_fraud_edges_seen = sum(
            1
            for edge in self.visited_edges
            if self._edge_is_fraud(self.graph.get_edge_data(*edge, default={}))
        )
        max_fraud_score = (
            num_fraud_edges_seen / max(len(self.visited_edges), 1)
            if self.visited_edges
            else 0.0
        )

        avg_amount_raw = (
            np.mean(
                [
                    self.graph.get_edge_data(*edge).get("amount", 0.0)
                    for edge in self.visited_edges
                ]
            )
            if self.visited_edges
            else 0.0
        )
        avg_amount = np.sign(avg_amount_raw) * np.log1p(abs(avg_amount_raw))

        is_current_flagged = float(self.current_node in self.flagged_nodes)

        history_features = np.array(
            [
                num_visited / self.config.max_steps,
                max_fraud_score,
                avg_amount,
                len(self.current_neighbors) / self.config.max_neighbors,
                self.step_count / self.config.max_steps,
                is_current_flagged,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([node_features, history_features])

        # Sanitize
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def _encode_account_type(self, account_type: str) -> List[float]:
        """One-hot encode account type."""
        types = ["individual", "business", "shell_company", "foreign", "crypto_exchange"]
        encoding = [0.0] * len(types)
        if account_type in types:
            encoding[types.index(account_type)] = 1.0
        return encoding

    def _get_proximity_reward(self, node: str) -> float:
        """Compute proximity-based reward shaping."""
        # Simple version: reward for being close to fraud
        if node in self._fraud_nodes:
            return 0.5

        # Check if any neighbor is fraud
        neighbors = list(self.graph.successors(node)) + list(
            self.graph.predecessors(node)
        )
        for neighbor in neighbors:
            if neighbor in self._fraud_nodes:
                return 0.2

        return 0.0

    def _get_curiosity_reward(self, node: str) -> float:
        """Compute curiosity-based reward for exploration."""
        if node not in self.visited_nodes:
            # Bonus for new node
            base_bonus = 0.05
            # Extra bonus for fraud-adjacent nodes
            if node in self._fraud_nodes:
                return base_bonus * 2
            return base_bonus
        return 0.0

    def _compute_terminal_reward(self) -> float:
        """Compute terminal reward based on episode performance."""
        if not self.flagged_nodes:
            return 0.0

        correct_flags = sum(1 for n in self.flagged_nodes if n in self._fraud_nodes)
        if correct_flags > 0:
            precision = correct_flags / len(self.flagged_nodes)
            return 0.5 * precision

        return 0.0

    def _store_node_features(self, node: str) -> None:
        """Store node features for episode data."""
        if node not in self.episode_data["node_features"]:
            data = self.graph.nodes.get(node, {})
            # Exclude ground truth fields
            safe_data = {
                k: v
                for k, v in data.items()
                if k not in ("is_fraud", "label", "is_money_laundering")
            }
            self.episode_data["node_features"][node] = safe_data

    def _store_edge_features(self, edge: Tuple[str, str]) -> None:
        """Store edge features for episode data."""
        edge_key = (str(edge[0]), str(edge[1]))
        if edge_key not in self.episode_data["edge_features"]:
            data = self.graph.get_edge_data(*edge, default={})
            # Exclude ground truth fields
            safe_data = {
                k: v
                for k, v in data.items()
                if k not in ("is_fraud", "isFraud", "is_money_laundering")
            }
            self.episode_data["edge_features"][edge_key] = safe_data

    def _finalize_episode_data(self) -> None:
        """Finalize episode data for judge evaluation."""
        self.episode_data["metadata"] = {
            "episode_length": self.step_count,
            "unique_nodes": len(self.visited_nodes),
            "flags_used": len(self.flagged_nodes),
            "total_reward": self.detector_episode_reward,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get episode information."""
        fraud_edges = [
            e
            for e in self.visited_edges
            if self._edge_is_fraud(self.graph.get_edge_data(*e, default={}))
        ]

        flag_correct = any(n in self._fraud_nodes for n in self.flagged_nodes)

        return {
            "current_node": self.current_node,
            "visited_nodes": len(self.visited_nodes),
            "step_count": self.step_count,
            "num_neighbors": len(self.current_neighbors),
            "episode_reward": self.detector_episode_reward,
            "flagged_nodes": len(self.flagged_nodes),
            "fraud_edges_found": len(fraud_edges),
            "flag_correct": flag_correct,
            "episode_data": self.episode_data,
        }

    @staticmethod
    def _edge_is_fraud(edge_data: Dict[str, Any]) -> bool:
        """Check if edge is fraudulent."""
        return bool(
            edge_data.get("is_fraud")
            or edge_data.get("isFraud")
            or edge_data.get("is_money_laundering")
        )

    def get_episode_data(self) -> Dict[str, Any]:
        """Get episode data for judge evaluation."""
        return copy.deepcopy(self.episode_data)

    def get_true_labels(self) -> Dict[str, bool]:
        """Get ground truth labels for flagged nodes (for training only)."""
        labels = {}
        for node in self.visited_nodes:
            labels[str(node)] = node in self._fraud_nodes
        return labels

    def render(self) -> None:
        """Render current state."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            print(f"Current Node: {self.current_node}")
            print(f"Visited: {len(self.visited_nodes)} nodes")
            print(f"Flagged: {len(self.flagged_nodes)} nodes")
            print(f"Neighbors: {self.current_neighbors}")
            print(f"Reward: {self.detector_episode_reward:.3f}")

    def close(self) -> None:
        """Clean up resources."""
        pass


# =============================================================================
# Multi-Agent Wrappers for RLlib
# =============================================================================


class RLlibMultiAgentWrapper:
    """
    Wrapper for RLlib multi-agent training.

    Converts MultiAgentAMLEnvironment to RLlib's MultiAgentEnv interface.
    """

    def __init__(self, env: MultiAgentAMLEnvironment):
        """
        Args:
            env: MultiAgentAMLEnvironment instance
        """
        self.env = env
        self._agent_ids = set(env.possible_agents)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset and return observations for all agents."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Return dict keyed by agent ID
        return {self.env.DETECTOR_AGENT: obs}, {self.env.DETECTOR_AGENT: info}

    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Execute actions for all agents."""
        # Get detector action
        detector_action = action_dict.get(self.env.DETECTOR_AGENT, 0)

        obs, reward, terminated, truncated, info = self.env.step(detector_action)

        return (
            {self.env.DETECTOR_AGENT: obs},
            {self.env.DETECTOR_AGENT: reward},
            {self.env.DETECTOR_AGENT: terminated, "__all__": terminated},
            {self.env.DETECTOR_AGENT: truncated, "__all__": truncated},
            {self.env.DETECTOR_AGENT: info},
        )

    @property
    def observation_space(self) -> Dict[str, Space]:
        """Get observation spaces for all agents."""
        return {self.env.DETECTOR_AGENT: self.env.observation_space}

    @property
    def action_space(self) -> Dict[str, Space]:
        """Get action spaces for all agents."""
        return {self.env.DETECTOR_AGENT: self.env.action_space}


# =============================================================================
# Coordinated Multi-Agent Environment
# =============================================================================


class CoordinatedMultiAgentEnv(gym.Env):
    """
    Multi-Agent AML Environment with Coordination Layer.

    Extends MultiAgentAMLEnvironment with:
    - GNN Proposer integration (proposes suspicious nodes/edges)
    - Judge scoring integration (scores proposals)
    - Observation augmentation (proposals/scores augment RL observation)
    - CTDE support (centralized training, decentralized execution)

    This implements Task 3.4: Multi-Agent Coordination.

    Architecture:
    ```
    ┌─────────────────────────────────────────────────┐
    │              CoordinatedMultiAgentEnv            │
    │                                                  │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
    │  │   GNN    │──▶│  Judge   │──▶│ Augment  │    │
    │  │ Proposer │   │  Scorer  │   │   Obs    │    │
    │  └──────────┘   └──────────┘   └────┬─────┘    │
    │        │                            │          │
    │        │         ┌──────────────────┘          │
    │        │         ▼                              │
    │  ┌─────┴────────────────────┐                  │
    │  │   MultiAgentAMLEnv       │                  │
    │  │   (base environment)     │                  │
    │  └──────────────────────────┘                  │
    └─────────────────────────────────────────────────┘
    ```

    Usage:
        ```python
        from rl_money_laundering.multiagent import (
            CoordinatedMultiAgentEnv,
            CoordinationConfig,
            TrainingMode,
        )

        config = CoordinationConfig(
            training_mode=TrainingMode.CTDE,
            use_shared_replay=True,
        )

        env = CoordinatedMultiAgentEnv(
            graph=transaction_graph,
            gnn_encoder=rmganets_encoder,
            judge_model=judge,
            coordination_config=config,
        )

        obs, info = env.reset()
        # obs now includes coordination_augmentation with proposals/scores
        ```
    """

    metadata = {"render_modes": ["human"], "name": "coordinated_multi_agent_aml"}

    def __init__(
        self,
        graph: nx.DiGraph,
        gnn_encoder: Any = None,
        judge_model: Any = None,
        env_config: Optional[DetectorAdversaryConfig] = None,
        coordination_config: Optional["CoordinationConfig"] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize coordinated multi-agent environment.

        Args:
            graph: Transaction graph
            gnn_encoder: RMGANets or StateEncoder for GNN proposer
            judge_model: Judge model for scoring proposals
            env_config: Environment configuration
            coordination_config: Coordination configuration
            render_mode: Rendering mode
        """
        super().__init__()

        # Import coordination components (avoid circular import)
        from .coordination import (
            CoordinationConfig,
            MultiAgentCoordinator,
            ObservationAugmentor,
            CTDEWrapper,
        )

        # Initialize base environment
        self.base_env = MultiAgentAMLEnvironment(
            graph=graph,
            config=env_config,
            render_mode=render_mode,
        )

        # Coordination setup
        self.coordination_config = coordination_config or CoordinationConfig()

        # Initialize coordinator
        self.coordinator = MultiAgentCoordinator(
            config=self.coordination_config,
            gnn_encoder=gnn_encoder,
            judge_model=judge_model,
        )

        # Initialize augmentor
        self.augmentor = ObservationAugmentor(self.coordination_config)

        # CTDE wrapper
        self.ctde = CTDEWrapper(
            coordinator=self.coordinator,
            augmentor=self.augmentor,
            training=True,
        )

        # Store graph reference
        self.graph = graph

        # Augmented observation space
        base_obs_dim = self.base_env.observation_space.shape[0]
        aug_dim = self.augmentor.get_augmentation_dim()
        total_dim = base_obs_dim + aug_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

        # Action space same as base
        self.action_space = self.base_env.action_space

        # Tracking
        self._last_proposals = []
        self._last_scores = []

        logger.info(
            f"CoordinatedMultiAgentEnv initialized: "
            f"obs_dim={total_dim}, mode={self.coordination_config.training_mode.value}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with coordination.

        Returns:
            Tuple of (augmented_observation, info)
        """
        # Reset base environment
        base_obs, info = self.base_env.reset(seed=seed, options=options)

        # Reset coordinator
        self.coordinator.gnn_proposer.reset_episode()

        # Build initial coordination observation
        coord_obs = self._build_coordination_observation()

        # Augment base observation
        augmented_obs = self._augment_observation(base_obs, coord_obs)

        # Add coordination info
        info["coordination"] = {
            "num_proposals": len(self._last_proposals),
            "num_scores": len(self._last_scores),
            "mode": self.coordination_config.training_mode.value,
        }

        return augmented_obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with coordination.

        Args:
            action: Agent action

        Returns:
            Tuple of (augmented_observation, reward, terminated, truncated, info)
        """
        # Execute base step
        base_obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Run coordination step
        coord_obs = self._build_coordination_observation()
        coord_results = self.coordinator.step(coord_obs, action)

        self._last_proposals = coord_results.get("proposals", [])
        self._last_scores = coord_results.get("scores", [])

        # Augment observation
        augmented_obs = self._augment_observation(base_obs, coord_obs)

        # Add coordination info to info dict
        info["coordination"] = {
            "num_proposals": len(self._last_proposals),
            "num_scores": len(self._last_scores),
            "high_priority_proposals": sum(
                1 for p in self._last_proposals
                if hasattr(p, "is_high_priority") and p.is_high_priority
            ),
            "recommended_action": coord_results.get("recommended_action"),
        }

        # Handle episode end
        if terminated or truncated:
            episode_data = self.base_env.get_episode_data()
            true_labels = self.base_env.get_true_labels()
            episode_metrics = self.coordinator.end_episode(episode_data, true_labels)
            info["coordination"]["episode_metrics"] = episode_metrics

        return augmented_obs, reward, terminated, truncated, info

    def _build_coordination_observation(self) -> Dict[str, Any]:
        """Build observation dict for coordination layer."""
        return {
            "graph": self.graph,
            "current_node": self.base_env.current_node,
            "visited_nodes": self.base_env.visited_nodes,
            "flagged_nodes": self.base_env.flagged_nodes,
            "step": self.base_env.step_count,
        }

    def _augment_observation(
        self,
        base_obs: np.ndarray,
        coord_obs: Dict[str, Any],
    ) -> np.ndarray:
        """Augment base observation with coordination signals.

        Args:
            base_obs: Base environment observation
            coord_obs: Coordination observation dict

        Returns:
            Augmented observation vector
        """
        # Get augmentation from coordinator
        augmented_dict = self.augmentor.augment(
            coord_obs,
            self._last_proposals,
            self._last_scores,
            self.coordinator.get_statistics(),
        )

        # Extract augmentation vector
        aug_vector = augmented_dict.get(
            "coordination_augmentation",
            np.zeros(self.augmentor.get_augmentation_dim(), dtype=np.float32),
        )

        # Concatenate base observation with augmentation
        return np.concatenate([base_obs, aug_vector], dtype=np.float32)

    def set_training_mode(self, training: bool) -> None:
        """Switch between training and execution modes.

        Args:
            training: True for training (centralized), False for execution
        """
        self.ctde.set_training(training)

    def get_episode_data(self) -> Dict[str, Any]:
        """Get episode data including coordination info."""
        data = self.base_env.get_episode_data()
        data["coordination"] = {
            "proposals": self._last_proposals,
            "scores": self._last_scores,
            "statistics": self.coordinator.get_statistics(),
        }
        return data

    def get_true_labels(self) -> Dict[str, bool]:
        """Get ground truth labels."""
        return self.base_env.get_true_labels()

    def render(self) -> None:
        """Render current state with coordination info."""
        self.base_env.render()
        if self.base_env.render_mode == "human":
            print(f"Proposals: {len(self._last_proposals)}")
            print(f"Scores: {len(self._last_scores)}")

    def close(self) -> None:
        """Clean up resources."""
        self.base_env.close()


def create_coordinated_env(
    graph: nx.DiGraph,
    gnn_encoder: Any = None,
    judge_model: Any = None,
    env_config: Optional[DetectorAdversaryConfig] = None,
    coordination_config: Optional["CoordinationConfig"] = None,
) -> CoordinatedMultiAgentEnv:
    """
    Factory function to create a coordinated multi-agent environment.

    Args:
        graph: Transaction graph
        gnn_encoder: GNN encoder (RMGANets, StateEncoder)
        judge_model: Judge model for scoring
        env_config: Environment configuration
        coordination_config: Coordination configuration

    Returns:
        Configured CoordinatedMultiAgentEnv

    Example:
        ```python
        env = create_coordinated_env(
            graph=transaction_graph,
            gnn_encoder=state_encoder,
            judge_model=judge,
        )

        obs, info = env.reset()
        action = policy.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        ```
    """
    return CoordinatedMultiAgentEnv(
        graph=graph,
        gnn_encoder=gnn_encoder,
        judge_model=judge_model,
        env_config=env_config,
        coordination_config=coordination_config,
    )
