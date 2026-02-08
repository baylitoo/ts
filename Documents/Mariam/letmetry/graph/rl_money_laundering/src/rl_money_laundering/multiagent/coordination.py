"""
Multi-Agent Coordination for AML Detection

This module implements the coordination layer for multi-agent training:
1. GNN Agent: Proposes suspicious nodes/edges based on graph structure
2. Judge Agent: Scores proposals with reasoning
3. RL Agent: Decides investigation actions based on scored proposals

Training Modes:
- CENTRALIZED: Single policy with shared observations from all agents
- INDEPENDENT: Each agent learns separately with local observations
- CTDE: Centralized Training, Decentralized Execution

Communication Protocol:
- Agents communicate through a message-passing interface
- Messages include proposals, scores, and action recommendations
- Supports both synchronous and asynchronous communication

References:
- Foerster et al. (2018): Counterfactual Multi-Agent Policy Gradients
- Lowe et al. (2017): Multi-Agent Actor-Critic (MADDPG)
- Rashid et al. (2018): QMIX for value decomposition
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class TrainingMode(Enum):
    """Training modes for multi-agent coordination."""
    CENTRALIZED = "centralized"      # Shared policy, joint observations
    INDEPENDENT = "independent"      # Separate policies, local observations
    CTDE = "ctde"                    # Centralized Training, Decentralized Execution


class AgentRole(Enum):
    """Roles in the multi-agent system."""
    GNN_PROPOSER = "gnn_proposer"    # Proposes suspicious nodes/edges
    JUDGE = "judge"                   # Scores proposals
    RL_INVESTIGATOR = "rl_investigator"  # Decides investigation actions


@dataclass
class AgentMessage:
    """Message passed between agents.

    Attributes:
        sender: Role of the sending agent
        recipient: Role of the receiving agent (None for broadcast)
        message_type: Type of message (proposal, score, action, etc.)
        content: Message payload
        timestamp: Step or episode number
        requires_response: Whether sender expects a response
        priority: Message priority (higher = more urgent)
    """
    sender: AgentRole
    recipient: Optional[AgentRole]
    message_type: str
    content: Dict[str, Any]
    timestamp: int
    requires_response: bool = False
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sender": self.sender.value,
            "recipient": self.recipient.value if self.recipient else None,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "requires_response": self.requires_response,
            "priority": self.priority,
        }


@dataclass
class NodeProposal:
    """Proposal for a suspicious node from GNN agent.

    Attributes:
        node_id: Identifier of the proposed node
        suspiciousness_score: GNN's suspiciousness estimate [0, 1]
        features: Relevant node features
        neighbors: List of connected node IDs
        reasoning: Feature importance or attention weights
    """
    node_id: str
    suspiciousness_score: float
    features: Dict[str, float]
    neighbors: List[str]
    reasoning: Dict[str, float] = field(default_factory=dict)

    @property
    def is_high_priority(self) -> bool:
        """Check if proposal is high priority."""
        return self.suspiciousness_score > 0.7


@dataclass
class EdgeProposal:
    """Proposal for a suspicious edge from GNN agent.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        suspiciousness_score: GNN's suspiciousness estimate [0, 1]
        features: Relevant edge features
        reasoning: Feature importance or attention weights
    """
    source_id: str
    target_id: str
    suspiciousness_score: float
    features: Dict[str, float]
    reasoning: Dict[str, float] = field(default_factory=dict)

    @property
    def edge_id(self) -> Tuple[str, str]:
        """Get edge identifier."""
        return (self.source_id, self.target_id)


@dataclass
class JudgeScore:
    """Score from Judge agent for a proposal.

    Attributes:
        proposal_id: Identifier of the scored proposal
        score: Judge's score [-1, 1]
        confidence: Judge's confidence [0, 1]
        fraud_type: Predicted fraud type
        reasoning: Natural language explanation
    """
    proposal_id: str
    score: float
    confidence: float
    fraud_type: Optional[str] = None
    reasoning: str = ""


@dataclass
class CoordinationConfig:
    """Configuration for multi-agent coordination.

    Attributes:
        training_mode: CENTRALIZED, INDEPENDENT, or CTDE
        max_proposals_per_step: Maximum GNN proposals per step
        judge_batch_size: Batch size for judge scoring
        communication_mode: "sync" or "async"
        use_shared_replay: Whether to use shared replay buffer
        proposal_threshold: Minimum score to consider a proposal
        judge_weight_in_action: Weight of judge score in action selection
    """
    training_mode: TrainingMode = TrainingMode.CTDE
    max_proposals_per_step: int = 5
    judge_batch_size: int = 16
    communication_mode: str = "sync"  # "sync" or "async"
    use_shared_replay: bool = True
    proposal_threshold: float = 0.3
    judge_weight_in_action: float = 0.5

    # Agent-specific configs
    gnn_update_frequency: int = 1      # Update GNN every N steps
    judge_update_frequency: int = 100  # Update judge every N episodes
    rl_update_frequency: int = 1       # Update RL every N steps

    # Communication settings
    message_buffer_size: int = 100
    async_timeout_ms: int = 50


class MessageBus:
    """Central message bus for agent communication.

    Handles routing messages between agents with support for:
    - Point-to-point messaging
    - Broadcast messaging
    - Priority-based ordering
    - Async message queuing
    """

    def __init__(self, config: Optional[CoordinationConfig] = None):
        """Initialize message bus.

        Args:
            config: Coordination configuration
        """
        self.config = config or CoordinationConfig()

        # Message queues per agent
        self._queues: Dict[AgentRole, List[AgentMessage]] = {
            role: [] for role in AgentRole
        }

        # Message history for debugging
        self._history: List[AgentMessage] = []
        self._history_limit = 1000

        # Statistics
        self.messages_sent = 0
        self.messages_delivered = 0

    def send(self, message: AgentMessage) -> None:
        """Send a message to target agent(s).

        Args:
            message: Message to send
        """
        self.messages_sent += 1

        # Store in history
        self._history.append(message)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        # Route message
        if message.recipient is None:
            # Broadcast to all agents except sender
            for role in AgentRole:
                if role != message.sender:
                    self._deliver(role, message)
        else:
            self._deliver(message.recipient, message)

    def _deliver(self, recipient: AgentRole, message: AgentMessage) -> None:
        """Deliver message to recipient's queue."""
        queue = self._queues[recipient]

        # Insert by priority (higher priority first)
        insert_idx = 0
        for i, msg in enumerate(queue):
            if message.priority > msg.priority:
                insert_idx = i
                break
            insert_idx = i + 1

        queue.insert(insert_idx, message)

        # Maintain buffer size
        if len(queue) > self.config.message_buffer_size:
            queue.pop()  # Remove lowest priority

        self.messages_delivered += 1

    def receive(
        self,
        agent: AgentRole,
        message_type: Optional[str] = None,
        block: bool = False,
    ) -> Optional[AgentMessage]:
        """Receive next message for an agent.

        Args:
            agent: Agent role receiving
            message_type: Filter by message type (None = any)
            block: Whether to block until message available

        Returns:
            Next message or None if queue empty
        """
        queue = self._queues[agent]

        if not queue:
            return None

        # Filter by type if specified
        if message_type:
            for i, msg in enumerate(queue):
                if msg.message_type == message_type:
                    return queue.pop(i)
            return None

        return queue.pop(0)

    def receive_all(
        self,
        agent: AgentRole,
        message_type: Optional[str] = None,
    ) -> List[AgentMessage]:
        """Receive all pending messages for an agent.

        Args:
            agent: Agent role receiving
            message_type: Filter by message type

        Returns:
            List of messages
        """
        queue = self._queues[agent]

        if message_type:
            messages = [m for m in queue if m.message_type == message_type]
            self._queues[agent] = [m for m in queue if m.message_type != message_type]
        else:
            messages = queue.copy()
            self._queues[agent] = []

        return messages

    def clear(self, agent: Optional[AgentRole] = None) -> None:
        """Clear message queue(s).

        Args:
            agent: Specific agent to clear (None = all)
        """
        if agent:
            self._queues[agent] = []
        else:
            for role in AgentRole:
                self._queues[role] = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "queue_sizes": {
                role.value: len(queue) for role, queue in self._queues.items()
            },
            "history_size": len(self._history),
        }


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""

    def __init__(
        self,
        role: AgentRole,
        message_bus: MessageBus,
        config: Optional[CoordinationConfig] = None,
    ):
        """Initialize base agent.

        Args:
            role: Agent's role in the system
            message_bus: Shared message bus
            config: Coordination configuration
        """
        self.role = role
        self.message_bus = message_bus
        self.config = config or CoordinationConfig()

        self.step_count = 0
        self.episode_count = 0

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Select an action given observation.

        Args:
            observation: Agent's observation

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Any) -> Dict[str, float]:
        """Update agent parameters.

        Args:
            batch: Training batch

        Returns:
            Update metrics
        """
        pass

    def send_message(
        self,
        recipient: Optional[AgentRole],
        message_type: str,
        content: Dict[str, Any],
        requires_response: bool = False,
        priority: int = 0,
    ) -> None:
        """Send a message through the message bus.

        Args:
            recipient: Target agent (None for broadcast)
            message_type: Type of message
            content: Message payload
            requires_response: Whether response expected
            priority: Message priority
        """
        message = AgentMessage(
            sender=self.role,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=self.step_count,
            requires_response=requires_response,
            priority=priority,
        )
        self.message_bus.send(message)

    def receive_messages(
        self,
        message_type: Optional[str] = None,
    ) -> List[AgentMessage]:
        """Receive all pending messages.

        Args:
            message_type: Filter by type

        Returns:
            List of messages
        """
        return self.message_bus.receive_all(self.role, message_type)


class GNNProposerAgent(BaseAgent):
    """GNN-based agent that proposes suspicious nodes/edges.

    This agent uses GNN embeddings to identify potentially fraudulent
    nodes and edges, then sends proposals to the Judge for scoring.

    The GNN encoder is shared with the RL agent but used differently:
    - RL agent uses embeddings for Q-value estimation
    - Proposer uses embeddings for anomaly detection
    """

    def __init__(
        self,
        message_bus: MessageBus,
        gnn_encoder: Any = None,  # StateEncoder
        config: Optional[CoordinationConfig] = None,
        anomaly_threshold: float = 0.5,
    ):
        """Initialize GNN proposer agent.

        Args:
            message_bus: Shared message bus
            gnn_encoder: GNN encoder for node embeddings
            config: Coordination configuration
            anomaly_threshold: Threshold for proposal generation
        """
        super().__init__(AgentRole.GNN_PROPOSER, message_bus, config)

        self.gnn_encoder = gnn_encoder
        self.anomaly_threshold = anomaly_threshold

        # Track proposed nodes to avoid duplicates
        self._proposed_nodes: Set[str] = set()
        self._proposed_edges: Set[Tuple[str, str]] = set()

        # Proposal history for learning
        self._proposal_outcomes: List[Dict[str, Any]] = []

    def act(self, observation: Dict[str, Any]) -> List[Union[NodeProposal, EdgeProposal]]:
        """Generate proposals for suspicious nodes/edges.

        Args:
            observation: Dict containing:
                - graph: NetworkX graph
                - current_node: Current position
                - visited_nodes: Set of visited nodes
                - node_embeddings: Optional pre-computed embeddings

        Returns:
            List of node and edge proposals
        """
        self.step_count += 1

        graph = observation.get("graph")
        current_node = observation.get("current_node")
        visited_nodes = observation.get("visited_nodes", set())

        if graph is None:
            return []

        proposals: List[Union[NodeProposal, EdgeProposal]] = []

        # Get node embeddings (compute if not provided)
        node_embeddings = observation.get("node_embeddings")
        if node_embeddings is None and self.gnn_encoder is not None:
            node_embeddings = self._compute_embeddings(graph, current_node)

        # Generate node proposals
        node_proposals = self._propose_nodes(
            graph, current_node, visited_nodes, node_embeddings
        )
        proposals.extend(node_proposals)

        # Generate edge proposals
        edge_proposals = self._propose_edges(
            graph, current_node, visited_nodes, node_embeddings
        )
        proposals.extend(edge_proposals)

        # Limit proposals per step
        proposals = sorted(
            proposals,
            key=lambda p: p.suspiciousness_score,
            reverse=True
        )[:self.config.max_proposals_per_step]

        # Send proposals to Judge
        if proposals:
            self.send_message(
                recipient=AgentRole.JUDGE,
                message_type="proposals",
                content={
                    "node_proposals": [
                        p for p in proposals if isinstance(p, NodeProposal)
                    ],
                    "edge_proposals": [
                        p for p in proposals if isinstance(p, EdgeProposal)
                    ],
                    "step": self.step_count,
                },
                requires_response=True,
                priority=1,
            )

        return proposals

    def _compute_embeddings(
        self,
        graph: Any,
        center_node: str,
    ) -> Dict[str, np.ndarray]:
        """Compute node embeddings using GNN encoder.

        Args:
            graph: NetworkX graph
            center_node: Center node for local subgraph

        Returns:
            Dict mapping node IDs to embeddings
        """
        if self.gnn_encoder is None:
            return {}

        # Get local subgraph
        k_hop = 2
        subgraph_nodes = {center_node}
        for _ in range(k_hop):
            new_nodes = set()
            for node in subgraph_nodes:
                if node in graph:
                    new_nodes.update(graph.predecessors(node))
                    new_nodes.update(graph.successors(node))
            subgraph_nodes.update(new_nodes)

        # Compute embeddings (simplified - actual implementation uses PyG)
        embeddings = {}
        for node in subgraph_nodes:
            # Placeholder: actual implementation uses GNN forward pass
            node_data = graph.nodes.get(node, {})
            features = np.array([
                float(node_data.get("balance", 0)),
                float(node_data.get("risk_score", 0)),
                float(node_data.get("is_suspicious", 0)),
                float(graph.out_degree(node)),
                float(graph.in_degree(node)),
            ], dtype=np.float32)
            embeddings[node] = features

        return embeddings

    def _propose_nodes(
        self,
        graph: Any,
        current_node: str,
        visited_nodes: Set[str],
        embeddings: Dict[str, np.ndarray],
    ) -> List[NodeProposal]:
        """Generate node proposals based on anomaly scores.

        Args:
            graph: NetworkX graph
            current_node: Current position
            visited_nodes: Already visited nodes
            embeddings: Node embeddings

        Returns:
            List of node proposals
        """
        proposals = []

        # Get candidate nodes (neighbors not yet visited)
        candidates = set()
        for node in visited_nodes:
            if node in graph:
                candidates.update(graph.successors(node))
                candidates.update(graph.predecessors(node))
        candidates -= visited_nodes
        candidates -= self._proposed_nodes

        for node in candidates:
            node_data = graph.nodes.get(node, {})

            # Compute suspiciousness score
            score = self._compute_node_suspiciousness(
                node, node_data, graph, embeddings.get(node)
            )

            if score >= self.anomaly_threshold:
                neighbors = list(graph.successors(node)) + list(graph.predecessors(node))

                proposal = NodeProposal(
                    node_id=node,
                    suspiciousness_score=score,
                    features={
                        "balance": float(node_data.get("balance", 0)),
                        "risk_score": float(node_data.get("risk_score", 0)),
                        "out_degree": float(graph.out_degree(node)),
                        "in_degree": float(graph.in_degree(node)),
                    },
                    neighbors=neighbors[:10],  # Limit neighbors
                    reasoning={"anomaly_score": score},
                )
                proposals.append(proposal)
                self._proposed_nodes.add(node)

        return proposals

    def _propose_edges(
        self,
        graph: Any,
        current_node: str,
        visited_nodes: Set[str],
        embeddings: Dict[str, np.ndarray],
    ) -> List[EdgeProposal]:
        """Generate edge proposals based on anomaly scores.

        Args:
            graph: NetworkX graph
            current_node: Current position
            visited_nodes: Already visited nodes
            embeddings: Node embeddings

        Returns:
            List of edge proposals
        """
        proposals = []

        # Get candidate edges from visited nodes
        for node in visited_nodes:
            if node not in graph:
                continue

            for neighbor in graph.successors(node):
                edge = (node, neighbor)
                if edge in self._proposed_edges:
                    continue

                edge_data = graph.get_edge_data(node, neighbor, default={})

                # Compute suspiciousness score
                score = self._compute_edge_suspiciousness(
                    edge, edge_data, graph, embeddings
                )

                if score >= self.anomaly_threshold:
                    proposal = EdgeProposal(
                        source_id=node,
                        target_id=neighbor,
                        suspiciousness_score=score,
                        features={
                            "amount": float(edge_data.get("amount", 0)),
                            "step": float(edge_data.get("step", 0)),
                        },
                        reasoning={"anomaly_score": score},
                    )
                    proposals.append(proposal)
                    self._proposed_edges.add(edge)

        return proposals

    def _compute_node_suspiciousness(
        self,
        node: str,
        node_data: Dict[str, Any],
        graph: Any,
        embedding: Optional[np.ndarray],
    ) -> float:
        """Compute suspiciousness score for a node.

        Combines multiple signals:
        - Risk score from node data
        - Structural anomalies (degree, centrality)
        - Embedding-based anomaly (if available)

        Args:
            node: Node ID
            node_data: Node attributes
            graph: Full graph
            embedding: Node embedding

        Returns:
            Suspiciousness score [0, 1]
        """
        scores = []

        # Risk score signal
        risk_score = float(node_data.get("risk_score", 0))
        scores.append(risk_score)

        # Structural signals
        out_degree = graph.out_degree(node)
        in_degree = graph.in_degree(node)

        # High degree ratio (many outgoing, few incoming = potential layering)
        if in_degree > 0:
            degree_ratio = out_degree / in_degree
            if degree_ratio > 5:  # High fan-out
                scores.append(0.7)
            elif degree_ratio < 0.2:  # High fan-in (potential collection point)
                scores.append(0.6)

        # Account type signal
        account_type = node_data.get("account_type", "individual")
        if account_type in ("shell_company", "crypto_exchange"):
            scores.append(0.8)
        elif account_type == "foreign":
            scores.append(0.5)

        # Embedding-based anomaly (placeholder)
        if embedding is not None:
            # In practice, compare to mean embedding or use isolation forest
            embedding_anomaly = float(np.std(embedding)) / 2.0
            scores.append(min(embedding_anomaly, 1.0))

        return np.mean(scores) if scores else 0.0

    def _compute_edge_suspiciousness(
        self,
        edge: Tuple[str, str],
        edge_data: Dict[str, Any],
        graph: Any,
        embeddings: Dict[str, np.ndarray],
    ) -> float:
        """Compute suspiciousness score for an edge.

        Args:
            edge: (source, target) tuple
            edge_data: Edge attributes
            graph: Full graph
            embeddings: Node embeddings

        Returns:
            Suspiciousness score [0, 1]
        """
        scores = []

        # Amount anomaly (log-normalized)
        amount = float(edge_data.get("amount", 0))
        if amount > 0:
            log_amount = np.log1p(amount)
            # High amounts are more suspicious
            amount_score = min(log_amount / 15.0, 1.0)  # Normalize to ~$1M
            scores.append(amount_score)

        # Round amount (structuring signal)
        if amount > 0 and amount % 1000 == 0:
            scores.append(0.6)
        if 9000 <= amount <= 10000:  # Just under reporting threshold
            scores.append(0.9)

        # Time-based anomaly (transactions at unusual times)
        # Placeholder - would need actual timestamp analysis

        # Embedding similarity (should endpoints be connected?)
        source_emb = embeddings.get(edge[0])
        target_emb = embeddings.get(edge[1])
        if source_emb is not None and target_emb is not None:
            # Dissimilar endpoints with transaction = potentially suspicious
            similarity = np.dot(source_emb, target_emb) / (
                np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
            )
            if similarity < 0.3:  # Low similarity
                scores.append(0.6)

        return np.mean(scores) if scores else 0.0

    def update(self, batch: Any) -> Dict[str, float]:
        """Update proposer based on feedback.

        Args:
            batch: Contains proposal outcomes

        Returns:
            Update metrics
        """
        # Store outcomes for threshold adjustment
        if isinstance(batch, dict) and "outcomes" in batch:
            self._proposal_outcomes.extend(batch["outcomes"])

        # Adjust threshold based on precision
        if len(self._proposal_outcomes) >= 100:
            recent = self._proposal_outcomes[-100:]
            precision = sum(1 for o in recent if o.get("was_fraud", False)) / len(recent)

            # Adjust threshold to target ~50% precision
            if precision < 0.4:
                self.anomaly_threshold = min(0.9, self.anomaly_threshold + 0.05)
            elif precision > 0.6:
                self.anomaly_threshold = max(0.2, self.anomaly_threshold - 0.05)

            return {
                "proposal_precision": precision,
                "anomaly_threshold": self.anomaly_threshold,
            }

        return {}

    def reset_episode(self) -> None:
        """Reset episode-specific state."""
        self._proposed_nodes = set()
        self._proposed_edges = set()
        self.episode_count += 1


class MultiAgentCoordinator:
    """Coordinates multiple agents during training and execution.

    Handles:
    - Agent initialization and lifecycle
    - Message routing and synchronization
    - Training mode switching
    - Replay buffer management
    - Metrics aggregation
    """

    def __init__(
        self,
        config: Optional[CoordinationConfig] = None,
        gnn_encoder: Any = None,
        judge_model: Any = None,
    ):
        """Initialize coordinator.

        Args:
            config: Coordination configuration
            gnn_encoder: Shared GNN encoder
            judge_model: Judge model for scoring
        """
        self.config = config or CoordinationConfig()

        # Initialize message bus
        self.message_bus = MessageBus(self.config)

        # Initialize agents
        self.gnn_proposer = GNNProposerAgent(
            message_bus=self.message_bus,
            gnn_encoder=gnn_encoder,
            config=self.config,
        )

        self.judge_model = judge_model

        # Training state
        self.training_mode = self.config.training_mode
        self.step_count = 0
        self.episode_count = 0

        # Replay buffers
        self._shared_buffer: List[Dict[str, Any]] = []
        self._agent_buffers: Dict[AgentRole, List[Dict[str, Any]]] = {
            role: [] for role in AgentRole
        }
        self._buffer_capacity = 10000

        # Metrics
        self._step_metrics: List[Dict[str, Any]] = []
        self._episode_metrics: List[Dict[str, Any]] = []

        logger.info(
            f"MultiAgentCoordinator initialized: mode={self.training_mode.value}, "
            f"shared_replay={self.config.use_shared_replay}"
        )

    def step(
        self,
        observation: Dict[str, Any],
        rl_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute one coordination step.

        Args:
            observation: Environment observation
            rl_action: Action from RL agent (if available)

        Returns:
            Coordination results including proposals and scores
        """
        self.step_count += 1
        results: Dict[str, Any] = {
            "step": self.step_count,
            "proposals": [],
            "scores": [],
            "recommended_action": None,
        }

        # 1. GNN Proposer generates proposals
        proposals = self.gnn_proposer.act(observation)
        results["proposals"] = proposals

        # 2. Judge scores proposals (if model available)
        if self.judge_model is not None and proposals:
            scores = self._score_proposals(proposals, observation)
            results["scores"] = scores

            # 3. Combine with RL action recommendation
            if rl_action is not None:
                results["recommended_action"] = self._combine_action_recommendation(
                    rl_action, scores, observation
                )

        # Store metrics
        self._step_metrics.append({
            "step": self.step_count,
            "num_proposals": len(proposals),
            "high_priority_proposals": sum(
                1 for p in proposals
                if isinstance(p, NodeProposal) and p.is_high_priority
            ),
        })

        return results

    def _score_proposals(
        self,
        proposals: List[Union[NodeProposal, EdgeProposal]],
        observation: Dict[str, Any],
    ) -> List[JudgeScore]:
        """Score proposals using Judge model.

        Args:
            proposals: List of proposals to score
            observation: Current observation for context

        Returns:
            List of judge scores
        """
        scores = []

        for proposal in proposals:
            if isinstance(proposal, NodeProposal):
                proposal_id = f"node:{proposal.node_id}"
            else:
                proposal_id = f"edge:{proposal.source_id}->{proposal.target_id}"

            # Get judge score (simplified - actual uses model forward)
            # In production, batch these for efficiency
            try:
                if hasattr(self.judge_model, "compute_reward"):
                    episode_dict = {
                        "proposal_id": proposal_id,
                        "node_id": getattr(proposal, "node_id", None),
                        "suspiciousness": getattr(proposal, "suspiciousness_score", 0.5),
                        "observation": observation,
                    }
                    reward_value, metadata = self.judge_model.compute_reward(
                        episode=episode_dict,
                        conservative=True,
                    )
                    scores.append(JudgeScore(
                        proposal_id=proposal_id,
                        score=reward_value,
                        confidence=1.0 - metadata.get("uncertainty", 0.5),
                        fraud_type=metadata.get("fraud_type"),
                        reasoning=str(metadata.get("fraud_type_probs", "")),
                    ))
                else:
                    # Fallback: use suspiciousness score as proxy
                    scores.append(JudgeScore(
                        proposal_id=proposal_id,
                        score=proposal.suspiciousness_score * 2 - 1,
                        confidence=0.5,
                    ))
            except Exception as e:
                logger.warning(f"Judge scoring failed: {e}")
                scores.append(JudgeScore(
                    proposal_id=proposal_id,
                    score=0.0,
                    confidence=0.0,
                ))

        return scores

    def _combine_action_recommendation(
        self,
        rl_action: int,
        scores: List[JudgeScore],
        observation: Dict[str, Any],
    ) -> int:
        """Combine RL action with judge scores.

        In CTDE mode, the judge scores influence action selection
        but RL agent makes final decision during execution.

        Args:
            rl_action: Original RL agent action
            scores: Judge scores for proposals
            observation: Current observation

        Returns:
            Final recommended action
        """
        if not scores or self.training_mode == TrainingMode.INDEPENDENT:
            return rl_action

        # Find highest scored proposal
        best_score = max(scores, key=lambda s: s.score * s.confidence)

        # If judge strongly recommends investigation (high score + confidence)
        if best_score.score > 0.5 and best_score.confidence > 0.7:
            # Check if RL action aligns with recommendation
            # This is a soft influence, not override
            logger.debug(
                f"Judge recommends investigating {best_score.proposal_id} "
                f"(score={best_score.score:.2f}, conf={best_score.confidence:.2f})"
            )

        # In CTDE, we don't override RL action during execution
        # The influence comes during training via reward shaping
        return rl_action

    def end_episode(
        self,
        episode_data: Dict[str, Any],
        true_labels: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Process end of episode.

        Args:
            episode_data: Episode trajectory data
            true_labels: Ground truth labels

        Returns:
            Episode metrics
        """
        self.episode_count += 1

        # Reset agents
        self.gnn_proposer.reset_episode()
        self.message_bus.clear()

        # Compute episode metrics
        metrics: Dict[str, Any] = {
            "episode": self.episode_count,
            "total_steps": self.step_count,
            "message_bus_stats": self.message_bus.get_statistics(),
        }

        # Aggregate step metrics
        if self._step_metrics:
            metrics["avg_proposals_per_step"] = np.mean(
                [m["num_proposals"] for m in self._step_metrics]
            )

        # Provide feedback to proposer
        if true_labels:
            outcomes = self._compute_proposal_outcomes(episode_data, true_labels)
            self.gnn_proposer.update({"outcomes": outcomes})

        # Store episode metrics
        self._episode_metrics.append(metrics)

        # Clear step metrics
        self._step_metrics = []

        return metrics

    def _compute_proposal_outcomes(
        self,
        episode_data: Dict[str, Any],
        true_labels: Dict[str, bool],
    ) -> List[Dict[str, Any]]:
        """Compute outcomes for proposals made this episode.

        Args:
            episode_data: Episode data
            true_labels: Ground truth

        Returns:
            List of proposal outcomes
        """
        outcomes = []

        flagged_nodes = set(episode_data.get("flagged_nodes", []))

        for node in self.gnn_proposer._proposed_nodes:
            was_flagged = node in flagged_nodes
            was_fraud = true_labels.get(str(node), False)

            outcomes.append({
                "node_id": node,
                "was_flagged": was_flagged,
                "was_fraud": was_fraud,
                "correct": was_flagged == was_fraud,
            })

        return outcomes

    def add_to_replay(
        self,
        experience: Dict[str, Any],
        agent: Optional[AgentRole] = None,
    ) -> None:
        """Add experience to replay buffer.

        Args:
            experience: Experience to store
            agent: Specific agent buffer (None = shared)
        """
        if self.config.use_shared_replay or agent is None:
            self._shared_buffer.append(experience)
            if len(self._shared_buffer) > self._buffer_capacity:
                self._shared_buffer = self._shared_buffer[-self._buffer_capacity:]
        else:
            self._agent_buffers[agent].append(experience)
            if len(self._agent_buffers[agent]) > self._buffer_capacity:
                self._agent_buffers[agent] = self._agent_buffers[agent][-self._buffer_capacity:]

    def sample_replay(
        self,
        batch_size: int,
        agent: Optional[AgentRole] = None,
    ) -> List[Dict[str, Any]]:
        """Sample from replay buffer.

        Args:
            batch_size: Number of samples
            agent: Specific agent buffer (None = shared or all)

        Returns:
            Batch of experiences
        """
        if self.config.use_shared_replay or agent is None:
            buffer = self._shared_buffer
        else:
            buffer = self._agent_buffers[agent]

        if not buffer:
            return []

        indices = np.random.choice(
            len(buffer),
            size=min(batch_size, len(buffer)),
            replace=False,
        )
        return [buffer[i] for i in indices]

    def set_training_mode(self, mode: TrainingMode) -> None:
        """Switch training mode.

        Args:
            mode: New training mode
        """
        old_mode = self.training_mode
        self.training_mode = mode

        logger.info(f"Training mode changed: {old_mode.value} -> {mode.value}")

        # Adjust replay buffer behavior
        if mode == TrainingMode.INDEPENDENT and self.config.use_shared_replay:
            logger.warning(
                "INDEPENDENT mode with shared replay may cause instability"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        stats = {
            "training_mode": self.training_mode.value,
            "total_steps": self.step_count,
            "total_episodes": self.episode_count,
            "message_bus": self.message_bus.get_statistics(),
            "shared_buffer_size": len(self._shared_buffer),
            "agent_buffer_sizes": {
                role.value: len(buffer)
                for role, buffer in self._agent_buffers.items()
            },
        }

        if self._episode_metrics:
            recent = self._episode_metrics[-10:]
            stats["avg_proposals_per_step"] = np.mean(
                [m.get("avg_proposals_per_step", 0) for m in recent]
            )

        return stats


class ObservationAugmentor:
    """Augments RL agent observations with GNN proposals and Judge scores.

    Implements the "Direct Influence" communication pattern where:
    - GNN proposals add suspiciousness signals to node features
    - Judge scores add confidence-weighted fraud indicators
    - These augment the RL agent's observation and can bias action selection

    This enables CTDE: during training, the augmented observation includes
    centralized information (proposals, scores); during execution, the
    agent can still make decisions with partial information.

    Reference:
    - Direct observation augmentation for multi-agent coordination
    - Enables joint GNN+RL gradient flow during training
    """

    def __init__(
        self,
        config: Optional[CoordinationConfig] = None,
        proposal_dim: int = 4,
        score_dim: int = 3,
        max_proposals: int = 10,
    ):
        """Initialize observation augmentor.

        Args:
            config: Coordination configuration
            proposal_dim: Features per proposal (score, type, priority, neighbors)
            score_dim: Features per judge score (score, confidence, fraud_type)
            max_proposals: Maximum proposals to include in observation
        """
        self.config = config or CoordinationConfig()
        self.proposal_dim = proposal_dim
        self.score_dim = score_dim
        self.max_proposals = max_proposals

        # Total augmentation dimension
        self.augmentation_dim = (
            max_proposals * proposal_dim +  # Proposal features
            max_proposals * score_dim +     # Judge score features
            3                                # Global coordination features
        )

    def get_augmentation_dim(self) -> int:
        """Get dimension of augmented observation space."""
        return self.augmentation_dim

    def augment(
        self,
        base_observation: Dict[str, Any],
        proposals: List[Union[NodeProposal, EdgeProposal]],
        scores: List[JudgeScore],
        coordinator_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Augment observation with coordination information.

        Args:
            base_observation: Original RL observation
            proposals: Current GNN proposals
            scores: Current Judge scores
            coordinator_stats: Optional coordinator statistics

        Returns:
            Augmented observation dict
        """
        augmented = dict(base_observation)

        # 1. Encode proposals as feature vector
        proposal_features = self._encode_proposals(proposals)

        # 2. Encode judge scores as feature vector
        score_features = self._encode_scores(scores)

        # 3. Global coordination features
        global_features = self._encode_global_features(
            proposals, scores, coordinator_stats
        )

        # 4. Combine into augmentation vector
        augmentation = np.concatenate([
            proposal_features,
            score_features,
            global_features,
        ])

        augmented["coordination_augmentation"] = augmentation

        # 5. Also add structured data for interpretability
        augmented["proposals_summary"] = {
            "num_proposals": len(proposals),
            "avg_suspiciousness": np.mean(
                [p.suspiciousness_score for p in proposals]
            ) if proposals else 0.0,
            "max_suspiciousness": max(
                [p.suspiciousness_score for p in proposals]
            ) if proposals else 0.0,
            "high_priority_count": sum(
                1 for p in proposals
                if isinstance(p, NodeProposal) and p.is_high_priority
            ),
        }

        augmented["scores_summary"] = {
            "num_scores": len(scores),
            "avg_score": np.mean([s.score for s in scores]) if scores else 0.0,
            "avg_confidence": np.mean([s.confidence for s in scores]) if scores else 0.0,
            "max_score": max([s.score for s in scores]) if scores else 0.0,
        }

        return augmented

    def _encode_proposals(
        self,
        proposals: List[Union[NodeProposal, EdgeProposal]],
    ) -> np.ndarray:
        """Encode proposals as fixed-size feature vector.

        Args:
            proposals: List of proposals

        Returns:
            Feature vector of shape (max_proposals * proposal_dim,)
        """
        features = np.zeros(self.max_proposals * self.proposal_dim, dtype=np.float32)

        # Sort by suspiciousness (highest first)
        sorted_proposals = sorted(
            proposals,
            key=lambda p: p.suspiciousness_score,
            reverse=True
        )[:self.max_proposals]

        for i, proposal in enumerate(sorted_proposals):
            offset = i * self.proposal_dim

            # Feature 0: Suspiciousness score [0, 1]
            features[offset] = proposal.suspiciousness_score

            # Feature 1: Proposal type (0 = node, 1 = edge)
            features[offset + 1] = 0.0 if isinstance(proposal, NodeProposal) else 1.0

            # Feature 2: Priority indicator (high priority = 1)
            if isinstance(proposal, NodeProposal):
                features[offset + 2] = 1.0 if proposal.is_high_priority else 0.0
            else:
                features[offset + 2] = 1.0 if proposal.suspiciousness_score > 0.7 else 0.0

            # Feature 3: Connectivity (normalized neighbor count for nodes)
            if isinstance(proposal, NodeProposal):
                features[offset + 3] = min(len(proposal.neighbors) / 20.0, 1.0)
            else:
                features[offset + 3] = 0.5  # Edge has 2 endpoints

        return features

    def _encode_scores(
        self,
        scores: List[JudgeScore],
    ) -> np.ndarray:
        """Encode judge scores as fixed-size feature vector.

        Args:
            scores: List of judge scores

        Returns:
            Feature vector of shape (max_proposals * score_dim,)
        """
        features = np.zeros(self.max_proposals * self.score_dim, dtype=np.float32)

        # Sort by score * confidence (highest first)
        sorted_scores = sorted(
            scores,
            key=lambda s: s.score * s.confidence,
            reverse=True
        )[:self.max_proposals]

        for i, score in enumerate(sorted_scores):
            offset = i * self.score_dim

            # Feature 0: Judge score [-1, 1] normalized to [0, 1]
            features[offset] = (score.score + 1.0) / 2.0

            # Feature 1: Confidence [0, 1]
            features[offset + 1] = score.confidence

            # Feature 2: Fraud type encoding (simple categorical)
            fraud_type_map = {
                "layering": 0.2,
                "structuring": 0.4,
                "cycling": 0.6,
                "integration": 0.8,
                None: 0.0,
            }
            features[offset + 2] = fraud_type_map.get(score.fraud_type, 0.0)

        return features

    def _encode_global_features(
        self,
        proposals: List[Union[NodeProposal, EdgeProposal]],
        scores: List[JudgeScore],
        stats: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Encode global coordination features.

        Args:
            proposals: Current proposals
            scores: Current scores
            stats: Coordinator statistics

        Returns:
            Feature vector of shape (3,)
        """
        features = np.zeros(3, dtype=np.float32)

        # Feature 0: Proposal density (normalized count)
        features[0] = min(len(proposals) / self.max_proposals, 1.0)

        # Feature 1: Agreement signal (do GNN and Judge agree?)
        if proposals and scores:
            # Compare top proposal suspiciousness with top judge score
            top_proposal_score = max(p.suspiciousness_score for p in proposals)
            top_judge_score = max(s.score for s in scores)
            # Both high or both low = agreement
            agreement = 1.0 - abs(top_proposal_score - (top_judge_score + 1) / 2)
            features[1] = agreement
        else:
            features[1] = 0.5  # Neutral

        # Feature 2: Urgency signal (should agent act now?)
        high_priority = sum(
            1 for p in proposals
            if isinstance(p, NodeProposal) and p.is_high_priority
        )
        high_confidence_scores = sum(
            1 for s in scores if s.confidence > 0.7 and s.score > 0.5
        )
        urgency = min(
            (high_priority + high_confidence_scores) / (self.max_proposals * 2),
            1.0
        )
        features[2] = urgency

        return features

    def augment_node_features(
        self,
        node_features: np.ndarray,
        node_ids: List[str],
        proposals: List[Union[NodeProposal, EdgeProposal]],
        scores: List[JudgeScore],
    ) -> np.ndarray:
        """Augment node features with per-node coordination signals.

        This directly modifies node features for GNN processing,
        enabling end-to-end gradient flow.

        Args:
            node_features: Original node features (num_nodes, feature_dim)
            node_ids: List of node IDs corresponding to features
            proposals: Current proposals
            scores: Current judge scores

        Returns:
            Augmented node features (num_nodes, feature_dim + 3)
        """
        num_nodes = node_features.shape[0]
        augmentation = np.zeros((num_nodes, 3), dtype=np.float32)

        # Create lookup for proposals and scores
        proposal_lookup: Dict[str, float] = {}
        for p in proposals:
            if isinstance(p, NodeProposal):
                proposal_lookup[p.node_id] = p.suspiciousness_score
            else:
                # For edges, mark both endpoints
                proposal_lookup[p.source_id] = max(
                    proposal_lookup.get(p.source_id, 0),
                    p.suspiciousness_score
                )
                proposal_lookup[p.target_id] = max(
                    proposal_lookup.get(p.target_id, 0),
                    p.suspiciousness_score
                )

        score_lookup: Dict[str, Tuple[float, float]] = {}
        for s in scores:
            # Parse proposal_id to get node
            if s.proposal_id.startswith("node:"):
                node_id = s.proposal_id[5:]
                score_lookup[node_id] = (s.score, s.confidence)
            elif s.proposal_id.startswith("edge:"):
                parts = s.proposal_id[5:].split("->")
                if len(parts) == 2:
                    for node_id in parts:
                        if node_id not in score_lookup:
                            score_lookup[node_id] = (s.score, s.confidence)

        # Add augmentation for each node
        for i, node_id in enumerate(node_ids):
            # Column 0: GNN suspiciousness signal
            augmentation[i, 0] = proposal_lookup.get(node_id, 0.0)

            # Column 1: Judge score signal (normalized to [0, 1])
            if node_id in score_lookup:
                score, confidence = score_lookup[node_id]
                augmentation[i, 1] = (score + 1.0) / 2.0 * confidence
            else:
                augmentation[i, 1] = 0.0

            # Column 2: Combined signal (weighted average)
            augmentation[i, 2] = (
                0.4 * augmentation[i, 0] +  # GNN weight
                0.6 * augmentation[i, 1]    # Judge weight (per config)
            )

        # Concatenate with original features
        return np.concatenate([node_features, augmentation], axis=1)


class CTDEWrapper:
    """Wrapper for Centralized Training, Decentralized Execution.

    During training:
    - Full observation augmentation with proposals and scores
    - Access to centralized coordinator state
    - Joint gradient updates across agents

    During execution:
    - Reduced augmentation (only local proposals)
    - No access to centralized state
    - Independent decision making
    """

    def __init__(
        self,
        coordinator: MultiAgentCoordinator,
        augmentor: Optional[ObservationAugmentor] = None,
        training: bool = True,
    ):
        """Initialize CTDE wrapper.

        Args:
            coordinator: Multi-agent coordinator
            augmentor: Observation augmentor (created if None)
            training: Whether in training mode
        """
        self.coordinator = coordinator
        self.augmentor = augmentor or ObservationAugmentor(coordinator.config)
        self.training = training

        # Cache for execution mode
        self._local_proposals_cache: List[Union[NodeProposal, EdgeProposal]] = []

    def process_observation(
        self,
        observation: Dict[str, Any],
        rl_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process observation through coordination pipeline.

        Args:
            observation: Raw environment observation
            rl_action: Current RL action (for logging)

        Returns:
            Augmented observation
        """
        if self.training:
            return self._process_training(observation, rl_action)
        else:
            return self._process_execution(observation, rl_action)

    def _process_training(
        self,
        observation: Dict[str, Any],
        rl_action: Optional[int],
    ) -> Dict[str, Any]:
        """Process observation in training mode (centralized).

        Full access to all coordination information.
        """
        # Run full coordination step
        coord_results = self.coordinator.step(observation, rl_action)

        # Augment observation
        augmented = self.augmentor.augment(
            observation,
            coord_results["proposals"],
            coord_results["scores"],
            self.coordinator.get_statistics(),
        )

        # Add training-specific data
        augmented["coord_training_data"] = {
            "proposals": coord_results["proposals"],
            "scores": coord_results["scores"],
            "recommended_action": coord_results.get("recommended_action"),
        }

        return augmented

    def _process_execution(
        self,
        observation: Dict[str, Any],
        rl_action: Optional[int],
    ) -> Dict[str, Any]:
        """Process observation in execution mode (decentralized).

        Limited to local proposals, no centralized state.
        """
        # Only generate local proposals (no judge scoring)
        proposals = self.coordinator.gnn_proposer.act(observation)
        self._local_proposals_cache = proposals

        # Minimal augmentation without judge scores
        augmented = self.augmentor.augment(
            observation,
            proposals,
            [],  # No judge scores in execution
            None,
        )

        return augmented

    def set_training(self, training: bool) -> None:
        """Switch between training and execution modes.

        Args:
            training: True for training, False for execution
        """
        self.training = training
        logger.info(f"CTDE mode: {'training' if training else 'execution'}")


class RMGANetsCoordinationAdapter:
    """Adapter for integrating RMGANets encoder with coordination layer.

    Connects the RMGANets graph neural network encoder to the multi-agent
    coordination system, enabling:
    - Proper embedding computation using RMGANets architecture
    - Subgraph splitting information for proposal generation
    - Gradient flow from RL back through GNN during training

    This implements the "Joint with RL" GNN update design choice:
    - Shared encoder between proposer and RL agent
    - Gradients flow through both paths
    - Enables end-to-end optimization

    Reference: RMGANets paper (Relation-Aware Multi-Graph Aggregation Networks)
    """

    def __init__(
        self,
        rmganets_encoder: Any,  # RMGANetsEncoder or StateEncoder
        device: str = "cpu",
        use_subgraph_info: bool = True,
    ):
        """Initialize RMGANets adapter.

        Args:
            rmganets_encoder: RMGANets encoder (or StateEncoder wrapping it)
            device: Torch device
            use_subgraph_info: Whether to use subgraph splitting for proposals
        """
        self.encoder = rmganets_encoder
        self.device = device
        self.use_subgraph_info = use_subgraph_info

        # Cache for subgraph information
        self._last_subgraph_info: Optional[Dict[str, Any]] = None

        # Detect encoder type
        self._is_state_encoder = hasattr(rmganets_encoder, "gnn")
        if self._is_state_encoder:
            self._gnn = rmganets_encoder.gnn
        else:
            self._gnn = rmganets_encoder

        # Check if encoder supports subgraph info (RMGANets-specific)
        self._supports_subgraph = hasattr(self._gnn, "get_subgraph_info")

    def compute_embeddings(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        return_subgraph_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Compute node embeddings using RMGANets.

        Args:
            node_features: Node features (num_nodes, feature_dim)
            edge_index: Edge indices (2, num_edges)
            return_subgraph_info: Whether to return subgraph split information

        Returns:
            embeddings: Node embeddings (num_nodes, embedding_dim)
            subgraph_info: Optional dict with high/medium/low similarity edges
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for RMGANets embeddings")

        # Convert to torch tensors
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edge_idx = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        # Forward pass through RMGANets
        with torch.set_grad_enabled(self.encoder.training):
            embeddings = self._gnn(x, edge_idx)

            # Get subgraph info if available and requested
            if return_subgraph_info and self._supports_subgraph:
                edge_high, edge_med, edge_low = self._gnn.get_subgraph_info(x, edge_idx)
                self._last_subgraph_info = {
                    "high_similarity": edge_high.cpu().numpy(),
                    "medium_similarity": edge_med.cpu().numpy(),
                    "low_similarity": edge_low.cpu().numpy(),
                }

        # Convert back to numpy
        embeddings_np = embeddings.detach().cpu().numpy()

        if return_subgraph_info and self._last_subgraph_info:
            return embeddings_np, self._last_subgraph_info
        return embeddings_np

    def compute_anomaly_scores(
        self,
        embeddings: np.ndarray,
        method: str = "isolation_forest",
    ) -> np.ndarray:
        """Compute anomaly scores from embeddings.

        Uses embeddings to identify potentially suspicious nodes
        for proposal generation.

        Args:
            embeddings: Node embeddings (num_nodes, embedding_dim)
            method: Anomaly detection method ("isolation_forest", "lof", "zscore")

        Returns:
            anomaly_scores: Per-node anomaly scores [0, 1]
        """
        if method == "zscore":
            # Simple z-score based anomaly
            mean_emb = np.mean(embeddings, axis=0)
            std_emb = np.std(embeddings, axis=0) + 1e-8
            z_scores = np.abs((embeddings - mean_emb) / std_emb)
            anomaly_scores = np.tanh(np.mean(z_scores, axis=1))

        elif method == "distance":
            # Distance from mean embedding
            mean_emb = np.mean(embeddings, axis=0, keepdims=True)
            distances = np.linalg.norm(embeddings - mean_emb, axis=1)
            max_dist = np.max(distances) + 1e-8
            anomaly_scores = distances / max_dist

        elif method == "isolation_forest":
            # Simplified isolation-forest style scoring
            # (Full sklearn IsolationForest would be better for production)
            n_nodes = embeddings.shape[0]
            anomaly_scores = np.zeros(n_nodes)

            for i in range(n_nodes):
                # Count how many nodes are "more normal" than this one
                distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
                avg_distance = np.mean(distances)
                # Higher average distance = more anomalous
                anomaly_scores[i] = avg_distance

            # Normalize to [0, 1]
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            if max_score > min_score:
                anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                anomaly_scores = np.zeros(n_nodes)

        else:
            raise ValueError(f"Unknown anomaly method: {method}")

        return anomaly_scores

    def get_high_similarity_subgraph(self) -> Optional[np.ndarray]:
        """Get edges from high-similarity subgraph (RMGANets-specific).

        High-similarity edges are processed by both To-GCM and HyGCM,
        often representing core transaction patterns.

        Returns:
            edge_index: High similarity edges (2, num_edges) or None
        """
        if self._last_subgraph_info is None:
            return None
        return self._last_subgraph_info.get("high_similarity")

    def get_suspicious_subgraph(self) -> Optional[np.ndarray]:
        """Get edges likely involved in suspicious activity.

        Low-similarity edges (processed by HyGCM) often correspond
        to unusual transaction patterns that may indicate fraud.

        Returns:
            edge_index: Low similarity edges (2, num_edges) or None
        """
        if self._last_subgraph_info is None:
            return None
        return self._last_subgraph_info.get("low_similarity")


def create_coordinated_training_setup(
    gnn_encoder: Any,
    judge_model: Any,
    config: Optional[CoordinationConfig] = None,
    device: str = "cpu",
) -> Tuple[MultiAgentCoordinator, CTDEWrapper, RMGANetsCoordinationAdapter]:
    """Create a complete coordinated training setup.

    Factory function that instantiates and connects all coordination
    components with proper configuration.

    Args:
        gnn_encoder: RMGANets or StateEncoder for graph processing
        judge_model: Judge model for scoring proposals
        config: Coordination configuration
        device: Torch device

    Returns:
        coordinator: Multi-agent coordinator
        ctde_wrapper: CTDE wrapper for observation augmentation
        rmganets_adapter: RMGANets adapter for embedding computation

    Example:
        ```python
        from rl_money_laundering.multiagent import (
            create_coordinated_training_setup,
            CoordinationConfig,
            TrainingMode,
        )

        config = CoordinationConfig(
            training_mode=TrainingMode.CTDE,
            use_shared_replay=True,
        )

        coordinator, ctde, adapter = create_coordinated_training_setup(
            gnn_encoder=state_encoder,
            judge_model=judge,
            config=config,
        )

        # In training loop:
        augmented_obs = ctde.process_observation(obs)
        action = agent.select_action(augmented_obs)
        ```
    """
    config = config or CoordinationConfig()

    # Create RMGANets adapter
    rmganets_adapter = RMGANetsCoordinationAdapter(
        rmganets_encoder=gnn_encoder,
        device=device,
        use_subgraph_info=True,
    )

    # Create coordinator with GNN encoder
    coordinator = MultiAgentCoordinator(
        config=config,
        gnn_encoder=gnn_encoder,
        judge_model=judge_model,
    )

    # Create CTDE wrapper
    augmentor = ObservationAugmentor(config)
    ctde_wrapper = CTDEWrapper(
        coordinator=coordinator,
        augmentor=augmentor,
        training=True,
    )

    logger.info(
        f"Created coordinated training setup: "
        f"mode={config.training_mode.value}, "
        f"shared_replay={config.use_shared_replay}"
    )

    return coordinator, ctde_wrapper, rmganets_adapter


# Export key classes
__all__ = [
    "TrainingMode",
    "AgentRole",
    "AgentMessage",
    "NodeProposal",
    "EdgeProposal",
    "JudgeScore",
    "CoordinationConfig",
    "MessageBus",
    "BaseAgent",
    "GNNProposerAgent",
    "MultiAgentCoordinator",
    "ObservationAugmentor",
    "CTDEWrapper",
    "RMGANetsCoordinationAdapter",
    "create_coordinated_training_setup",
]
