"""
GNN RLModule for RLlib (New API Stack)

Custom RLModule that integrates GNN state encoder with DQN for graph-based RL.
Follows Ray 2.40+ new API stack patterns.
"""

from typing import Any, Dict, Mapping
import inspect
import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

from ..gnn_encoder import StateEncoder
from .batch_utils import GraphBatchCollator


class GNNDQNModule(TorchRLModule):
    """
    Custom RLModule integrating GNN encoder with DQN.

    Processes graph-structured observations through:
    1. GNN encoder (RMGANets, GraphSAGE, GAT, etc.)
    2. Q-network head

    Compatible with RLlib's new API stack (Ray 2.40+).
    """

    def setup(self) -> None:
        """Initialize GNN encoder and Q-network."""
        # Extract config parameters
        model_config = self.config.model_config_dict

        node_feature_dim = model_config["node_feature_dim"]
        action_dim = self.config.action_space.n
        gnn_type = model_config.get("gnn_type", "rmganets")
        embedding_dim = model_config.get("embedding_dim", 32)
        history_dim = model_config.get("history_dim", 16)
        multi_branch = model_config.get("multi_branch", False)
        use_dqn_enhancement = model_config.get("use_dqn_enhancement", False)

        # Initialize GNN state encoder
        self.gnn_encoder = StateEncoder(
            node_feature_dim=node_feature_dim,
            gnn_type=gnn_type,
            embedding_dim=embedding_dim,
            history_dim=history_dim,
            device=str(self.config.model_config_dict.get("_device", "cpu")),
            multi_branch=multi_branch,
            use_dqn_enhancement=use_dqn_enhancement,
            num_classes=2,
        )

        # Q-network head (takes concatenated embedding + history)
        state_dim = embedding_dim + history_dim
        hidden_dims = model_config.get("hidden_dims", [128, 128, 64])

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.q_network = nn.Sequential(*layers)

        # Initialize batch collator for efficient graph processing
        self.batch_collator = GraphBatchCollator()

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward pass for inference (greedy action selection).

        In inference mode, we select actions greedily based on Q-values.
        No exploration, no gradients.

        Args:
            batch: Batch dict containing Columns.OBS with graph structure
            **kwargs: Additional arguments (e.g., 't' for timestep) - ignored for DQN

        Returns:
            Dict with Columns.ACTION_DIST_INPUTS (Q-values for greedy selection)
        """
        # Extract graph observations (use new API Columns)
        obs = batch[Columns.OBS]

        # Encode graph state through GNN
        state = self._encode_graph_batch(obs)

        # Compute Q-values
        q_values = self.q_network(state)

        # Return using new API column names
        return {Columns.ACTION_DIST_INPUTS: q_values}

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward pass for exploration (with stochastic action selection).

        In exploration mode, we return Q-values which the DQN algorithm
        will use with epsilon-greedy exploration (handled by algorithm).

        Args:
            batch: Batch dict containing Columns.OBS with graph structure
            **kwargs: Additional arguments (e.g., 't' for timestep) - ignored for DQN

        Returns:
            Dict with Columns.ACTION_DIST_INPUTS (Q-values for exploration)
        """
        # For DQN, exploration and inference have same forward logic
        # Epsilon-greedy is handled by the DQN algorithm, not the RLModule
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward pass for training (compute Q-values for loss calculation).

        In training mode, we compute Q-values for both current and next states
        to enable temporal difference learning (DQN loss).

        Args:
            batch: Batch dict containing Columns.OBS and Columns.NEXT_OBS
            **kwargs: Additional arguments (e.g., 't' for timestep) - ignored for DQN

        Returns:
            Dict with Columns.ACTION_DIST_INPUTS for current and next states
        """
        # Encode current states (use new API Columns)
        obs = batch[Columns.OBS]
        state = self._encode_graph_batch(obs)
        q_values = self.q_network(state)

        # DQN requires both ACTION_DIST_INPUTS and QF_PREDS (Q-function predictions)
        output = {
            Columns.ACTION_DIST_INPUTS: q_values,
            "qf_preds": q_values,  # DQN learner expects this key
        }

        # Compute next state Q-values for TD target (if available)
        if Columns.NEXT_OBS in batch:
            next_obs = batch[Columns.NEXT_OBS]
            next_state = self._encode_graph_batch(next_obs)
            next_q_values = self.q_network(next_state)
            # Add next state Q-values with proper keys
            output[Columns.ACTION_DIST_INPUTS + "_next"] = next_q_values
            output["qf_next_preds"] = next_q_values  # DQN learner expects this too
            # Target network predictions (in new API, RLlib handles target network separately,
            # but we need to provide the key with current network predictions)
            output["qf_target_next_preds"] = next_q_values  # DQN learner expects this for TD target

        return output

    def _encode_graph_batch(self, graph_obs_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode batch of graph observations through GNN (OPTIMIZED with PyG Batch).

        This version uses PyTorch Geometric's Batch class for efficient parallel
        processing of multiple graphs. ~10x faster than sequential processing.

        Args:
            graph_obs_batch: Dict with keys:
                - node_features: (batch_size, max_nodes, node_feature_dim)
                - edge_index: (batch_size, 2, max_edges)
                - num_nodes: (batch_size,)
                - num_edges: (batch_size,)
                - context_features: (batch_size, context_feature_dim)

        Returns:
            state: (batch_size, embedding_dim + history_dim) state vectors
        """
        device = graph_obs_batch["node_features"].device

        # Convert to PyG Batch for parallel processing
        pyg_batch = self.batch_collator(graph_obs_batch)
        pyg_batch = pyg_batch.to(device)

        # Forward through GNN in parallel (single call handles all graphs!)
        if pyg_batch.x.shape[0] > 0:  # Check if there are any nodes
            # Process all graphs together
            if hasattr(self.gnn_encoder.gnn, 'supports_temporal'):
                # Temporal GNNs (TGAT, TGN) - edge_time support
                edge_time = getattr(pyg_batch, 'edge_time', None)
                node_embeddings = self.gnn_encoder.gnn(
                    pyg_batch.x,
                    pyg_batch.edge_index,
                    edge_time=edge_time
                )
            else:
                # Standard GNNs (GraphSAGE, GAT, RMGANets)
                # Check if GNN supports return_auxiliary parameter (RMGANets does, others don't)
                forward_sig = inspect.signature(self.gnn_encoder.gnn.forward)
                if 'return_auxiliary' in forward_sig.parameters:
                    # RMGANets with optional multi-branch outputs
                    node_embeddings = self.gnn_encoder.gnn(
                        pyg_batch.x,
                        pyg_batch.edge_index,
                        return_auxiliary=False  # Don't need auxiliary outputs here
                    )
                else:
                    # Standard GNN forward (GraphSAGE, GAT, etc.)
                    node_embeddings = self.gnn_encoder.gnn(
                        pyg_batch.x,
                        pyg_batch.edge_index
                    )

            # Extract first node embedding from each graph (current node of interest)
            # PyG Batch provides .batch tensor that maps nodes to graphs
            first_node_embeddings = self.batch_collator.extract_first_nodes(
                node_embeddings,
                pyg_batch
            )  # (batch_size, embedding_dim)
        else:
            # Fallback for empty graphs
            batch_size = graph_obs_batch["node_features"].shape[0]
            first_node_embeddings = torch.zeros(
                batch_size,
                self.gnn_encoder.embedding_dim,
                device=device
            )

        # Concatenate node embedding + context features (history)
        context = pyg_batch.context  # (batch_size, context_dim)
        state = torch.cat([first_node_embeddings, context], dim=1)  # (batch_size, state_dim)

        return state


# Legacy: For compatibility with older RLlib examples, provide alternative registration
def make_gnn_dqn_module(config: RLModuleConfig) -> GNNDQNModule:
    """
    Factory function for creating GNNDQNModule.

    Args:
        config: RLModule configuration

    Returns:
        Initialized GNNDQNModule
    """
    return GNNDQNModule(config)
