from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm  # type: ignore[import-untyped]

from .agent import AgentProtocol
from .config import MultiBranchLossConfig
from .environment import AMLDetectionEnv
from .features import BaseNodeFeatureExtractor, NodeFeatureExtractor
from .gnn_encoder import StateEncoder
from .gnn_modules.multi_branch_loss import MultiBranchLoss, SimplifiedMultiBranchLoss
from .utils.graph_replay import GraphReplayBuffer

AuxiliaryOutputs = Dict[str, Any]
HistoryEntry = Dict[str, float]
MultiBranchLossType = Union[MultiBranchLoss, SimplifiedMultiBranchLoss]
TrainingHistory = Dict[str, List[Union[int, float]]]


class AMLTrainer:
    """
    Trainer for temporal graph guardrail tasks (AML is one case study).

    Implements:
    - Curriculum learning (start easy, gradually harder)
    - Episode sampling from high-risk subgraphs
    - Periodic evaluation and checkpointing
    - Logging and monitoring aligned with RL-Guard telemetry
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        agent: AgentProtocol,
        state_encoder: StateEncoder,
        env: AMLDetectionEnv,
        fraud_subgraphs: List[nx.DiGraph],
        output_dir: str = "outputs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        multi_branch_config: Optional[MultiBranchLossConfig] = None,
        feature_extractor: Optional[BaseNodeFeatureExtractor] = None,
        use_graph_replay: bool = False,
        graph_replay_capacity: int = 5000,
    ) -> None:
        self.graph = graph
        self.agent = agent
        self.state_encoder = state_encoder
        self.env = env
        self.fraud_subgraphs = fraud_subgraphs
        self.device = device

        self.base_intrinsic_reward_weight = self.env.intrinsic_reward_weight

        # Output path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.detection_rates: List[float] = []
        self.false_positive_rates: List[float] = []

        self.feature_extractor = feature_extractor or NodeFeatureExtractor(
            include_temporal=True,
            include_network=True,
            node_feature_dim=state_encoder.node_feature_dim,
        )

        self.multi_branch_config: MultiBranchLossConfig = multi_branch_config or MultiBranchLossConfig()
        self.multi_branch_enabled: bool = (
            self.multi_branch_config.enabled
            and getattr(self.state_encoder, "multi_branch_enabled", False)
        )
        self.multi_branch_metrics: List[HistoryEntry] = []
        self.multi_branch_parameters: List[nn.Parameter] = []
        self.multi_branch_loss_fn: Optional[MultiBranchLossType] = None
        self.multi_branch_head: Optional[nn.Linear] = None
        self.multi_branch_optimizer: Optional[Optimizer] = None

        # Always keep the GNN in train mode for RL updates
        self.state_encoder.train()
        self.gnn_optimizer = torch.optim.Adam(
            self.state_encoder.gnn.parameters(),
            lr=1e-4,
        )

        if self.multi_branch_enabled:
            if self.multi_branch_config.variant == "paper":
                self.multi_branch_loss_fn = SimplifiedMultiBranchLoss(
                    lambda_branch=self.multi_branch_config.lambda_branch,
                    epsilon_dqn=self.multi_branch_config.epsilon_dqn,
                )
            else:
                self.multi_branch_loss_fn = MultiBranchLoss(
                    lambda_branch=self.multi_branch_config.lambda_branch,
                    epsilon_dqn=self.multi_branch_config.epsilon_dqn,
                    beta_reg=self.multi_branch_config.beta_reg,
                    temporal_decay=self.multi_branch_config.temporal_decay,
                    adaptive_weighting=self.multi_branch_config.adaptive_weighting,
                )
            self.multi_branch_head = nn.Linear(
                self.state_encoder.embedding_dim,
                getattr(self.state_encoder, "num_classes", 2),
            ).to(self.device)
            self.multi_branch_parameters = list(self.state_encoder.gnn.parameters()) + list(
                self.multi_branch_head.parameters()
            )
            self.multi_branch_optimizer = torch.optim.Adam(
                self.multi_branch_parameters,
                lr=self.multi_branch_config.learning_rate,
            )

        self.use_graph_replay = use_graph_replay
        self.graph_replay_buffer = (
            GraphReplayBuffer(capacity=graph_replay_capacity) if use_graph_replay else None
        )

    def node_feature_extractor(self, node_data: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector for a node."""
        return self.feature_extractor.extract(node_data)

    def _get_node_label(self, node_id: str) -> Optional[int]:
        """Extract binary fraud label for a node if available."""
        data = self.graph.nodes.get(node_id, {})
        for key in ("label", "is_fraud", "is_money_laundering", "isFraud", "fraud", "class"):
            value = data.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _maybe_update_multi_branch(
        self,
        embedding_tensor: torch.Tensor,
        state_vector: Optional[np.ndarray],
        aux_outputs: Optional[AuxiliaryOutputs],
        node_label: Optional[int],
        timestamp_value: float,
    ) -> None:
        """Apply a multi-branch optimisation step if enabled."""
        loss_fn = self.multi_branch_loss_fn
        head = self.multi_branch_head
        optimizer = self.multi_branch_optimizer

        if (
            not self.multi_branch_enabled
            or node_label is None
            or loss_fn is None
            or head is None
            or optimizer is None
            or state_vector is None
        ):
            return

        embedding_tensor = embedding_tensor.to(self.device)
        outputs_main = head(embedding_tensor.unsqueeze(0))

        if aux_outputs is not None and aux_outputs.get("to_branch_logits") is not None:
            outputs_to = cast(torch.Tensor, aux_outputs["to_branch_logits"]).unsqueeze(0)
        else:
            outputs_to = outputs_main.detach()

        if aux_outputs is not None and aux_outputs.get("hy_branch_logits") is not None:
            outputs_hy = cast(torch.Tensor, aux_outputs["hy_branch_logits"]).unsqueeze(0)
        else:
            outputs_hy = torch.zeros((1, 1), device=self.device)

        state_tensor = torch.as_tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dqn_predictions = self.agent.compute_q_values(state_tensor)
            dqn_targets = self.agent.compute_target_q_values(state_tensor)

        targets = torch.tensor([node_label], dtype=torch.long, device=self.device)
        timestamps_tensor = torch.tensor([timestamp_value], dtype=torch.float32, device=self.device)

        subgraph_stats_raw = aux_outputs.get("subgraph_stats") if aux_outputs else None
        subgraph_stats: Optional[Dict[str, int]] = (
            subgraph_stats_raw if isinstance(subgraph_stats_raw, dict) else None
        )

        optimizer.zero_grad()
        loss, metrics = loss_fn(
            outputs_main,
            outputs_to,
            outputs_hy,
            dqn_predictions,
            dqn_targets,
            targets,
            subgraph_stats=subgraph_stats,
            timestamps=timestamps_tensor,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.multi_branch_parameters, 1.0)
        optimizer.step()
        self.multi_branch_metrics.append(metrics)

    def sample_episode_start(self, fraud_rate: float = 0.5) -> Optional[str]:
        """
        Sample starting node for episode (curriculum learning).

        Args:
            fraud_rate: Probability of starting near fraud (0.0-1.0)

        Returns:
            Starting node ID (if available)
        """
        if random.random() < fraud_rate and self.fraud_subgraphs:
            subgraph = random.choice(self.fraud_subgraphs)
            nodes = [str(node) for node in subgraph.nodes()]
            if nodes:
                start_node = random.choice(nodes)
                # Log sampling occasionally for debugging
                if random.random() < 0.01:
                    print(f"[CURRICULUM] Sampled from fraud subgraph, node={start_node}, fraud_rate={fraud_rate:.2f}")
                return start_node

        nodes = [str(node) for node in self.graph.nodes()]
        start_node = random.choice(nodes) if nodes else None
        if random.random() < 0.01:
            print(f"[CURRICULUM] Sampled random node={start_node}, fraud_rate={fraud_rate:.2f}")
        return start_node

    def _build_graph_state(self, center_node: str):
        """Build ordered PyG Data for a center node using the state encoder."""
        subgraph_nodes = self.state_encoder._get_local_subgraph(self.graph, center_node, k=2)
        ordered_nodes = [center_node] + sorted([n for n in subgraph_nodes if n != center_node])
        subgraph = self.graph.subgraph(ordered_nodes)
        ordered_graph = nx.DiGraph()
        ordered_graph.add_nodes_from(ordered_nodes)
        ordered_graph.add_edges_from(subgraph.edges(data=True))
        return self.state_encoder._networkx_to_pyg(ordered_graph, self.feature_extractor)

    def _maybe_update_gnn_td(
        self,
        embedding_tensor: torch.Tensor,
        history_features: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_embedding: Optional[torch.Tensor],
        next_history_features: Optional[np.ndarray],
    ) -> None:
        """Update GNN parameters using a TD-style loss on the current transition."""
        if self.gnn_optimizer is None or self.multi_branch_enabled:
            return
        if not hasattr(self.agent, "compute_q_values") or not hasattr(self.agent, "compute_target_q_values"):
            return

        history_tensor = torch.as_tensor(
            history_features,
            dtype=embedding_tensor.dtype,
            device=embedding_tensor.device,
        )
        state_tensor = torch.cat([embedding_tensor, history_tensor], dim=0).unsqueeze(0)

        q_network = getattr(self.agent, "q_network", None)
        if q_network is None:
            return

        requires_grad_flags = [param.requires_grad for param in q_network.parameters()]
        try:
            for param in q_network.parameters():
                param.requires_grad_(False)

            q_values = self.agent.compute_q_values(state_tensor)
            if q_values.ndim != 2 or action >= q_values.shape[1]:
                return
            q_value = q_values[0, action]

            with torch.no_grad():
                target = torch.tensor(float(reward), device=embedding_tensor.device, dtype=q_value.dtype)
                if not done and next_embedding is not None and next_history_features is not None:
                    next_history_tensor = torch.as_tensor(
                        next_history_features,
                        dtype=embedding_tensor.dtype,
                        device=embedding_tensor.device,
                    )
                    next_state = torch.cat([next_embedding.detach(), next_history_tensor], dim=0).unsqueeze(0)
                    next_q_values = self.agent.compute_target_q_values(next_state)
                    if next_q_values.ndim == 2:
                        max_next_q = next_q_values.max(dim=1).values[0]
                        gamma = float(getattr(self.agent, "gamma", 0.99))
                        target = target + gamma * max_next_q

            loss = F.mse_loss(q_value, target)
            self.gnn_optimizer.zero_grad()
            loss.backward()
            self.gnn_optimizer.step()
        finally:
            for param, flag in zip(q_network.parameters(), requires_grad_flags):
                param.requires_grad_(flag)

    def _maybe_update_gnn_from_replay(self, batch_size: int) -> None:
        """Update GNN parameters using graph replay if enabled."""
        if self.graph_replay_buffer is None or self.gnn_optimizer is None:
            return
        if len(self.graph_replay_buffer) < batch_size:
            return
        if not hasattr(self.agent, "compute_q_values") or not hasattr(self.agent, "compute_target_q_values"):
            return

        batch = self.graph_replay_buffer.sample(batch_size)
        graphs = batch.graphs.to(self.device)
        next_graphs = batch.next_graphs.to(self.device)

        if self.state_encoder.supports_temporal:
            edge_time = getattr(graphs, "edge_time", None)
            next_edge_time = getattr(next_graphs, "edge_time", None)
            node_embeddings = self.state_encoder.gnn(
                graphs.x, graphs.edge_index, edge_time=edge_time
            )
            next_embeddings = self.state_encoder.gnn(
                next_graphs.x, next_graphs.edge_index, edge_time=next_edge_time
            )
        else:
            node_embeddings = self.state_encoder.gnn(graphs.x, graphs.edge_index)
            next_embeddings = self.state_encoder.gnn(next_graphs.x, next_graphs.edge_index)

        current_indices = graphs.ptr[:-1]
        next_indices = next_graphs.ptr[:-1]
        current_embeddings = node_embeddings[current_indices]
        next_state_embeddings = next_embeddings[next_indices]

        histories = batch.histories.to(self.device)
        next_histories = batch.next_histories.to(self.device)
        states = torch.cat([current_embeddings, histories], dim=1)
        next_states = torch.cat([next_state_embeddings, next_histories], dim=1)

        q_network = getattr(self.agent, "q_network", None)
        if q_network is None:
            return

        requires_grad_flags = [param.requires_grad for param in q_network.parameters()]
        try:
            for param in q_network.parameters():
                param.requires_grad_(False)

            q_values = self.agent.compute_q_values(states)
            if q_values.ndim != 2:
                return
            actions = batch.actions.to(self.device)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.agent.compute_target_q_values(next_states)
                max_next_q = next_q_values.max(dim=1).values
                gamma = float(getattr(self.agent, "gamma", 0.99))
                targets = batch.rewards.to(self.device) + (1.0 - batch.dones.to(self.device)) * gamma * max_next_q

            loss = F.mse_loss(q_selected, targets)
            self.gnn_optimizer.zero_grad()
            loss.backward()
            self.gnn_optimizer.step()
        finally:
            for param, flag in zip(q_network.parameters(), requires_grad_flags):
                param.requires_grad_(flag)

    def run_episode(
        self,
        max_steps: int = 20,
        epsilon: Optional[float] = None,
        render: bool = False,
        start_node: Optional[str] = None,
        batch_size: int = 64,
        update_gnn: bool = True,
    ) -> Dict[str, Any]:
        """Run a single training episode."""
        reset_options = {"start_node": start_node} if start_node else None
        _, info = self.env.reset(options=reset_options)
        info_dict: Dict[str, Any] = info if isinstance(info, dict) else {}
        final_info: Dict[str, Any] = dict(info_dict)

        episode_reward = 0.0
        episode_length = 0
        done = False

        trajectory: List[Dict[str, Any]] = []

        while not done and episode_length < max_steps:
            current_node = self.env.current_node
            if current_node is None:
                break

            use_grad = self.multi_branch_enabled or update_gnn
            if use_grad:
                embedding_tensor, history_features, aux_outputs = self.state_encoder.forward_state(
                    graph=self.graph,
                    current_node=current_node,
                    visited_nodes=self.env.visited_nodes,
                    visited_edges=list(self.env.visited_edges),
                    node_feature_extractor=self.node_feature_extractor,
                    return_auxiliary=self.multi_branch_enabled,
                )
            else:
                with torch.no_grad():
                    embedding_tensor, history_features, aux_outputs = self.state_encoder.forward_state(
                        graph=self.graph,
                        current_node=current_node,
                        visited_nodes=self.env.visited_nodes,
                        visited_edges=list(self.env.visited_edges),
                        node_feature_extractor=self.node_feature_extractor,
                        return_auxiliary=False,
                    )

            state_vector = np.concatenate([embedding_tensor.detach().cpu().numpy().reshape(-1), history_features])

            if self.multi_branch_enabled:
                node_label = self._get_node_label(current_node)
                timestamp_value = info_dict.get("step_count", episode_length)
                self._maybe_update_multi_branch(
                    embedding_tensor,
                    state_vector,
                    aux_outputs,
                    node_label,
                    float(timestamp_value),
                )

            valid_actions = list(range(len(self.env.current_neighbors))) + [self.env.FLAG_ACTION]

            # Extract neighbor hints (risk scores) for guided exploration during training
            # NOTE: neighbor_hints is only passed during training (epsilon > 0)
            # During evaluation (epsilon=0), this is None for pure exploitation
            neighbor_hints = None
            if epsilon is not None and epsilon > 0 and self.env.current_neighbors:
                # Get risk scores for each neighbor + FLAG action
                # Calculate heuristic risk score based on node connectivity patterns
                neighbor_risk_scores = []
                for neighbor in self.env.current_neighbors:
                    # Heuristic: nodes with more outgoing edges and high amounts are riskier
                    out_edges = list(self.graph.out_edges(neighbor, data=True))

                    # Base risk on out-degree (more outgoing = higher risk for layering)
                    out_degree_risk = min(len(out_edges) / 10.0, 1.0)  # Normalize by 10

                    # High amounts are suspicious
                    if out_edges:
                        amounts = [data.get('amount', 0) for _, _, data in out_edges]
                        avg_amount = np.mean(amounts) if amounts else 0
                        amount_risk = min(avg_amount / 100000.0, 1.0)  # Normalize
                    else:
                        amount_risk = 0.5

                    # Combine signals (weighted average)
                    risk_score = 0.6 * out_degree_risk + 0.4 * amount_risk
                    risk_score = float(np.clip(risk_score, 0.1, 0.9))  # Keep in reasonable range

                    neighbor_risk_scores.append(risk_score)

                # FLAG action gets median risk (neutral preference)
                neighbor_risk_scores.append(0.5)
                neighbor_hints = np.array(neighbor_risk_scores, dtype=np.float32)

            action = self.agent.select_action(state_vector, valid_actions, epsilon, neighbor_hints)

            # Track FLAG action usage
            is_flag_action = (action == self.env.FLAG_ACTION)

            # Get action confidence for hybrid reward system
            # Compute confidence from Q-values (softmax over risk-adjusted Q-values for valid actions)
            with torch.no_grad():
                state_tensor = torch.as_tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_tensor = self.agent.compute_q_values(state_tensor)
                q_values = q_tensor.squeeze(0).detach().cpu().numpy()
                q_values = np.atleast_1d(q_values)

                confidence = 0.5
                if valid_actions:
                    valid_indices = np.array(valid_actions, dtype=int)
                    q_values = np.asarray(q_values, dtype=np.float32)
                    mask = np.full_like(q_values, -np.inf, dtype=np.float32)
                    mask[valid_indices] = 0.0
                    masked_q = q_values + mask
                    valid_q_values = masked_q[valid_indices]
                    finite_mask = np.isfinite(valid_q_values)
                    if finite_mask.any():
                        max_q = np.max(valid_q_values[finite_mask])
                        probs = np.zeros_like(valid_q_values, dtype=np.float32)
                        probs[finite_mask] = np.exp(valid_q_values[finite_mask] - max_q)
                        prob_sum = float(probs.sum())
                        if prob_sum > 0.0:
                            probs /= prob_sum
                            if action in valid_actions:
                                action_idx = valid_actions.index(action)
                                confidence = float(probs[action_idx])

            _, reward, terminated, truncated, info = self.env.step(action, confidence=confidence)
            info_dict = info if isinstance(info, dict) else {}
            final_info = dict(info_dict)
            next_current_node = self.env.current_node

            # Log important events (1% sample rate to avoid spam)
            if random.random() < 0.01:
                if is_flag_action:
                    flag_used = bool(info_dict.get("flag_used", False))
                    flag_correct = bool(info_dict.get("flag_correct", False))
                    if flag_used:
                        if (flag_correct and reward <= 0.0) or ((not flag_correct) and reward >= 0.0):
                            print(f"[BUG] Flag reward sign mismatch | Correct: {flag_correct} | Reward: {reward:+.2f}")
                        flagged_nodes = info_dict.get("flagged_nodes", 0)
                        print(f"[FLAG] Action used | Correct: {flag_correct} | "
                              f"Confidence: {confidence:.2f} | Reward: {reward:+.2f} | "
                              f"Total flags: {flagged_nodes}")
                elif reward > 0.5:  # Significant positive reward (proximity/curiosity/fraud)
                    print(f"[EXPLORE] Good move | Reward: {reward:+.2f} | "
                          f"Step: {self.env.step_count}")

            if not (terminated or truncated) and next_current_node is not None:
                with torch.no_grad():
                    next_embedding_tensor, next_history_features, _ = self.state_encoder.forward_state(
                        graph=self.graph,
                        current_node=next_current_node,
                        visited_nodes=self.env.visited_nodes,
                        visited_edges=list(self.env.visited_edges),
                        node_feature_extractor=self.node_feature_extractor,
                        return_auxiliary=False,
                    )
                next_state = np.concatenate(
                    [next_embedding_tensor.detach().cpu().numpy().reshape(-1), next_history_features]
                )
            else:
                next_state = state_vector

            is_fraud_transition = bool(
                info_dict.get("fraud_edge_encountered", info_dict.get("encountered_fraud_edge", False))
                or info_dict.get("flag_correct", False)
            )

            if update_gnn:
                self._maybe_update_gnn_td(
                    embedding_tensor=embedding_tensor,
                    history_features=history_features,
                    action=action,
                    reward=reward,
                    done=terminated or truncated,
                    next_embedding=next_embedding_tensor if not (terminated or truncated) else None,
                    next_history_features=next_history_features if not (terminated or truncated) else None,
                )
                if self.graph_replay_buffer is not None:
                    graph_state = self._build_graph_state(current_node)
                    if next_current_node is None:
                        next_graph_state = graph_state
                        next_history = history_features
                    else:
                        next_graph_state = self._build_graph_state(next_current_node)
                        next_history = next_history_features if next_history_features is not None else history_features
                    self.graph_replay_buffer.push(
                        graph_state=graph_state,
                        history_features=history_features,
                        action=action,
                        reward=reward,
                        next_graph_state=next_graph_state,
                        next_history_features=next_history,
                        done=terminated or truncated,
                        is_fraud=is_fraud_transition,
                    )

            self.agent.store_transition(
                state=state_vector,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated or truncated,
                is_fraud=is_fraud_transition,
            )

            if len(self.agent.replay_buffer) >= batch_size:
                _ = self.agent.train_step(batch_size=batch_size)
                if update_gnn and self.graph_replay_buffer is not None:
                    self._maybe_update_gnn_from_replay(batch_size=batch_size)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            trajectory.append(
                {
                    "state": state_vector,
                    "action": action,
                    "reward": reward,
                    "info": dict(info_dict),
                }
            )

            if render:
                self.env.render()

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "fraud_edges_found": final_info.get("fraud_edges_found", 0),
            "fraud_rate": final_info.get("fraud_rate", 0.0),
            "trajectory": trajectory,
            "info": final_info,
        }

    def train(
        self,
        num_episodes: int = 1000,
        curriculum_schedule: Optional[List[float]] = None,
        eval_frequency: int = 50,
        checkpoint_frequency: int = 100,
        target_update_frequency: int = 10,
        batch_size: int = 64,
    ) -> TrainingHistory:
        """Train agent with curriculum learning."""

        def _seed_everything(seed: int = 1337) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        _seed_everything(int(os.environ.get("SEED", "1337")))
        if curriculum_schedule is None:
            curriculum_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.02]

        episodes_per_stage = max(1, num_episodes // len(curriculum_schedule))

        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Total episodes: {num_episodes}")
        print(f"Curriculum stages: {len(curriculum_schedule)}")
        print(f"Episodes per stage: {episodes_per_stage}")
        print(f"Device: {self.device}")
        print()
        print("METRICS LEGEND:")
        print("  • Avg Reward: Average episode reward (last 10 episodes)")
        print("  • Epsilon: Exploration rate (1.0=random, 0.0=greedy)")
        print("  • Intrinsic W: Weight for curiosity/proximity rewards")
        print("  • Found%: % of episodes where agent ENCOUNTERED fraud edges during exploration")
        print("  • Flagged%: % of episodes where agent USED the FLAG action")
        print("  • Curriculum%: % of episodes starting from fraud subgraphs (curriculum learning)")
        print("  • Detection Rate: % of evaluation episodes with CORRECT fraud detection")
        print("  • FPR: False Positive Rate (% of evaluations that were false alarms)")
        print("=" * 80)

        initial_fraud_rate = max(curriculum_schedule[0], 1e-6)

        # Tracking variables for diagnostics
        episodes_with_fraud = 0
        episodes_with_flag = 0
        episodes_from_fraud_subgraph = 0

        for episode in tqdm(range(num_episodes), desc="Training"):
            stage_idx = min(episode // episodes_per_stage, len(curriculum_schedule) - 1)
            fraud_rate = curriculum_schedule[stage_idx]

            scaled_intrinsic = self.base_intrinsic_reward_weight * (fraud_rate / initial_fraud_rate)
            self.env.intrinsic_reward_weight = max(0.2, scaled_intrinsic)

            # Track whether we sample from fraud subgraph
            sampled_from_fraud = (random.random() < fraud_rate and self.fraud_subgraphs)
            if sampled_from_fraud:
                episodes_from_fraud_subgraph += 1

            start_node = self.sample_episode_start(fraud_rate)

            # Progressive episode length: start short, increase as curriculum reduces
            # This matches eval episodes (200 steps) by end of training
            progress = episode / num_episodes
            min_steps = 20
            max_steps = 200
            current_max_steps = int(min_steps + (max_steps - min_steps) * progress)

            episode_stats = self.run_episode(
                max_steps=current_max_steps,
                epsilon=self.agent.epsilon,
                render=False,
                start_node=start_node,
                batch_size=batch_size,
            )

            self.episode_rewards.append(float(episode_stats["episode_reward"]))
            self.episode_lengths.append(int(episode_stats["episode_length"]))
            self.agent.episode_rewards.append(float(episode_stats["episode_reward"]))

            # Track diagnostic metrics
            fraud_found = episode_stats.get("fraud_edges_found", 0) > 0
            if fraud_found:
                episodes_with_fraud += 1

            # Check if any action in trajectory was FLAG
            trajectory = episode_stats.get("trajectory", [])
            used_flag = any(step.get("action") == self.env.FLAG_ACTION for step in trajectory)
            if used_flag:
                episodes_with_flag += 1

            self.agent.decay_epsilon()

            if episode % target_update_frequency == 0:
                self.agent.update_target_network()

            if episode % eval_frequency == 0 and episode > 0:
                self._evaluate(episode, batch_size=batch_size)

            if episode % checkpoint_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)

            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0

                # Calculate diagnostic rates for last 10 episodes
                window_start = max(0, episode - 9)
                window_size = episode - window_start + 1
                fraud_rate_pct = (episodes_with_fraud / max(episode + 1, 1)) * 100
                flag_rate_pct = (episodes_with_flag / max(episode + 1, 1)) * 100
                curriculum_rate_pct = (episodes_from_fraud_subgraph / max(episode + 1, 1)) * 100

                tqdm.write(
                    f"Ep {episode:4d} | "
                    f"Stage {stage_idx + 1}/{len(curriculum_schedule)} | "
                    f"Reward: {avg_reward:+6.2f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"IntW: {self.env.intrinsic_reward_weight:.2f} | "
                    f"Found%: {fraud_rate_pct:4.1f} | "
                    f"Flagged%: {flag_rate_pct:4.1f} | "
                    f"Curriculum%: {curriculum_rate_pct:4.1f}"
                )

            if self.multi_branch_enabled and hasattr(self.multi_branch_loss_fn, "step_epoch"):
                cast(MultiBranchLoss, self.multi_branch_loss_fn).step_epoch()
            if self.multi_branch_enabled:
                self._maybe_log_multi_branch_metrics(episode)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        self._evaluate(num_episodes, is_final=True, batch_size=batch_size)
        self._save_checkpoint(num_episodes, is_final=True)

        return {
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "detection_rates": list(self.detection_rates),
            "false_positive_rates": list(self.false_positive_rates),
            "losses": list(getattr(self.agent, "losses", [])),
        }

    def _evaluate(
        self,
        episode: int,
        num_eval_episodes: int = 20,
        is_final: bool = False,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance with PROPER METRICS FOR RARE EVENTS.

        Computes:
        - Precision/Recall/F1 at episode level
        - Confusion matrix (TP/FP/FN/TN)
        - Per-episode event statistics
        """
        eval_type = "Final Evaluation" if is_final else f"Evaluation at Episode {episode}"
        print(f"{'=' * 80}")
        print(eval_type)
        print(f"{'=' * 80}")
        print(f"Running {num_eval_episodes} evaluation episodes per pass (greedy policy, ??=0.0)...")

        was_training = self.state_encoder.gnn.training
        self.state_encoder.eval()

        def _run_eval_pass(label: str, *, seed_from_fraud: bool) -> Dict[str, float]:
            print(f"{label}")
            print('-' * 80)

            eval_rewards: List[float] = []

            tp_episodes = 0
            fp_episodes = 0
            fn_episodes = 0
            tn_episodes = 0

            total_fraud_edges_encountered = 0
            total_fraud_episodes = 0
            total_flagged = 0
            total_min_dist: List[float] = []

            for _ in range(num_eval_episodes):
                start_node = None
                if seed_from_fraud:
                    start_node = self.sample_episode_start(fraud_rate=1.0)
                stats = self.run_episode(
                    max_steps=200,
                    epsilon=0.0,
                    render=False,
                    start_node=start_node,
                    batch_size=batch_size,
                    update_gnn=False,
                )
                eval_rewards.append(float(stats['episode_reward']))

                info = stats.get('info', {})
                fraud_edge_encountered = info.get('fraud_edge_encountered', info.get('encounter_fraud_edge', False))
                episode_contains_fraud = info.get('episode_contains_fraud', fraud_edge_encountered)
                flag_used = info.get('flag_used', False)
                flag_correct = info.get('flag_correct', False)
                min_dist = info.get('min_distance_to_fraud', 999)

                if fraud_edge_encountered:
                    total_fraud_edges_encountered += 1
                if episode_contains_fraud:
                    total_fraud_episodes += 1
                if flag_used:
                    total_flagged += 1
                if min_dist < 999:
                    total_min_dist.append(min_dist)

                if flag_used:
                    if flag_correct:
                        tp_episodes += 1
                    else:
                        fp_episodes += 1
                else:
                    if episode_contains_fraud:
                        fn_episodes += 1
                    else:
                        tn_episodes += 1

            mean_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
            precision = tp_episodes / (tp_episodes + fp_episodes + 1e-12) if (tp_episodes + fp_episodes) > 0 else 0.0
            recall = tp_episodes / (tp_episodes + fn_episodes + 1e-12) if (tp_episodes + fn_episodes) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall + 1e-12)) if (precision + recall) > 0 else 0.0

            fraud_edge_encounter_rate = total_fraud_edges_encountered / num_eval_episodes
            fraud_episode_rate = total_fraud_episodes / num_eval_episodes
            flag_rate = total_flagged / num_eval_episodes
            avg_min_dist = float(np.mean(total_min_dist)) if total_min_dist else 999.0

            print('CONFUSION MATRIX (episode-level):')
            print(f"  TP: {tp_episodes:3d}  |  FP: {fp_episodes:3d}")
            print(f"  FN: {fn_episodes:3d}  |  TN: {tn_episodes:3d}")
            print()
            print('PERFORMANCE METRICS:')
            print(f"  ??? Avg Reward: {mean_reward:+.2f}")
            print(f"  ??? Precision:  {precision * 100:5.1f}% (% of flags that were correct)")
            print(f"  ??? Recall:     {recall * 100:5.1f}% (% of frauds that were flagged)")
            print(f"  ??? F1 Score:   {f1 * 100:5.1f}%")
            print()
            print('EVENT STATISTICS:')
            print(f"  - Fraud Edge Encounter Rate: {fraud_edge_encounter_rate * 100:5.1f}% ({total_fraud_edges_encountered}/{num_eval_episodes} episodes)")
            print(f"  - Episodes With Fraud Present: {fraud_episode_rate * 100:5.1f}% ({total_fraud_episodes}/{num_eval_episodes} episodes)")
            print(f"  - Flag Usage Rate:      {flag_rate * 100:5.1f}% ({total_flagged}/{num_eval_episodes} episodes)")
            print(f"  - Avg Min Distance:     {avg_min_dist:.1f} hops")
            print('-' * 80)

            return {
                'mean_reward': mean_reward,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': float(tp_episodes),
                'fp': float(fp_episodes),
                'fn': float(fn_episodes),
                'tn': float(tn_episodes),
                'fraud_edge_encounter_rate': fraud_edge_encounter_rate,
                'fraud_episode_rate': fraud_episode_rate,
                'flag_rate': flag_rate,
                'avg_min_dist': avg_min_dist,
            }

        random_summary = _run_eval_pass('Random-start evaluation', seed_from_fraud=False)
        self.detection_rates.append(random_summary['recall'])
        self.false_positive_rates.append(random_summary['fp'] / num_eval_episodes)

        fraud_summary: Dict[str, float] | None = None
        if self.fraud_subgraphs:
            fraud_summary = _run_eval_pass('Fraud-seeded evaluation', seed_from_fraud=True)

        print(f"{'=' * 80}")

        if fraud_summary is not None:
            random_summary['fraud_seeded_precision'] = fraud_summary['precision']
            random_summary['fraud_seeded_recall'] = fraud_summary['recall']
            random_summary['fraud_seeded_f1'] = fraud_summary['f1']

        if was_training:
            self.state_encoder.train()

        return random_summary

    def _save_checkpoint(self, episode: int, is_final: bool = False) -> None:
        """Save model checkpoint."""
        filename = "final_model.pt" if is_final else f"checkpoint_ep{episode}.pt"
        filepath = self.output_dir / "checkpoints" / filename

        self.agent.save(str(filepath))
        print(f"Saved checkpoint: {filepath}")

        stats_file = self.output_dir / "training_stats.npz"
        losses_array = np.asarray(getattr(self.agent, "losses", []), dtype=np.float32)
        np.savez(
            stats_file,
            episode_rewards=np.asarray(self.episode_rewards, dtype=np.float32),
            episode_lengths=np.asarray(self.episode_lengths, dtype=np.int32),
            detection_rates=np.asarray(self.detection_rates, dtype=np.float32),
            false_positive_rates=np.asarray(self.false_positive_rates, dtype=np.float32),
            losses=losses_array,
        )

    def _maybe_log_multi_branch_metrics(self, episode: int) -> None:
        """Aggregate and log multi-branch metrics."""
        if not self.multi_branch_metrics or self.multi_branch_config.log_frequency <= 0:
            return

        if episode % self.multi_branch_config.log_frequency != 0:
            return

        start_index = max(0, len(self.multi_branch_metrics) - self.multi_branch_config.log_frequency)
        recent_metrics = self.multi_branch_metrics[start_index:]

        if not recent_metrics:
            return

        mean_metrics = {
            key: float(np.mean([entry[key] for entry in recent_metrics if key in entry]))
            for key in recent_metrics[-1].keys()
        }

        tqdm.write(
            "Multi-branch metrics (episode {}): {}".format(
                episode,
                ", ".join(f"{k}={v:.4f}" for k, v in mean_metrics.items()),
            )
        )

        max_keep = self.multi_branch_config.log_frequency * 10
        if len(self.multi_branch_metrics) > max_keep:
            self.multi_branch_metrics = self.multi_branch_metrics[-max_keep:]