"""
Adaptive Topology Graph Convolution Module (To-GCM)

Implements Equations 8-9 from RMGANets paper:
- Adaptive convolution on medium & high similarity subgraphs
- Multi-scale receptive field processing
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AdaptiveTopoGCM(nn.Module):
    """
    Adaptive Topology Graph Convolution Module.

    Key features:
    - Processes medium and high similarity subgraphs
    - Adaptive graph convolution with shared node weighting
    - Multi-scale feature aggregation

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Separate convolutions for high and medium similarity subgraphs
        self.conv_high = GCNConv(in_channels, out_channels // 2)
        self.conv_medium = GCNConv(in_channels, out_channels // 2)

        # Merged convolution for shared nodes (Equation 9)
        self.conv_merged = GCNConv(in_channels, out_channels)

        self.dropout = nn.Dropout(dropout)

    def merge_adjacency(
        self,
        edge_index_high: torch.Tensor,
        edge_index_medium: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create merged adjacency matrix with weighted shared nodes (Equation 9).

        Shared nodes (appearing in both subgraphs) get 2x weight.

        Args:
            edge_index_high: High similarity edges [2, E_h]
            edge_index_medium: Medium similarity edges [2, E_m]
            num_nodes: Number of nodes

        Returns:
            edge_index_merged: Merged edges
            edge_weight_merged: Edge weights (2.0 for shared, 1.0 otherwise)
        """
        # Concatenate edge indices
        edge_index_merged = torch.cat([edge_index_high, edge_index_medium], dim=1)

        # Find shared edges
        edge_ids_merged = edge_index_merged[0] * num_nodes + edge_index_merged[1]

        # Count occurrences (shared edges appear twice)
        unique_ids, counts = torch.unique(edge_ids_merged, return_counts=True)

        # Create weight map
        weight_map = torch.ones(edge_ids_merged.size(0), device=edge_index_merged.device)

        # Assign 2.0 weight to first occurrence of shared edges
        for idx, edge_id in enumerate(edge_ids_merged):
            if (counts[unique_ids == edge_id] > 1).any():
                weight_map[idx] = 2.0

        # Remove duplicates (keep first occurrence with weight 2.0)
        edge_index_merged, inverse_indices = torch.unique(edge_index_merged, dim=1, return_inverse=True)
        edge_weight_merged = torch.zeros(edge_index_merged.size(1), device=edge_index_merged.device)

        # Aggregate weights for unique edges
        for i, inv_idx in enumerate(inverse_indices):
            edge_weight_merged[inv_idx] = max(edge_weight_merged[inv_idx].item(), weight_map[i].item())

        return edge_index_merged, edge_weight_merged

    def forward(
        self,
        H: torch.Tensor,
        edge_index_high: torch.Tensor,
        edge_index_medium: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: Adaptive convolution on medium/high similarity subgraphs.

        Equation 8: H₁ = σ(A^∇hm ⊙ H ⊙ W₁) ⊗ [σ(A^∇h ⊙ H ⊙ W₂) ⊕ σ(A^∇m ⊙ H ⊙ W₃)]

        Args:
            H: Input features from Att-GCM [num_nodes, in_channels]
            edge_index_high: High similarity edges
            edge_index_medium: Medium similarity edges

        Returns:
            H1: Output features [num_nodes, out_channels]
        """
        num_nodes = H.size(0)

        # Process high similarity subgraph
        if edge_index_high.size(1) > 0:
            H_high = self.conv_high(H, edge_index_high)
            H_high = F.elu(H_high)
        else:
            H_high = torch.zeros(num_nodes, self.out_channels // 2, device=H.device)

        # Process medium similarity subgraph
        if edge_index_medium.size(1) > 0:
            H_medium = self.conv_medium(H, edge_index_medium)
            H_medium = F.elu(H_medium)
        else:
            H_medium = torch.zeros(num_nodes, self.out_channels // 2, device=H.device)

        # Equation 8: Concatenate high and medium features
        H_cat = torch.cat([H_high, H_medium], dim=1)  # [N, out_channels]

        # Process merged subgraph with shared node weighting (Equation 9)
        if edge_index_high.size(1) > 0 or edge_index_medium.size(1) > 0:
            edge_index_merged, edge_weight_merged = self.merge_adjacency(
                edge_index_high, edge_index_medium, num_nodes
            )
            H_merged = self.conv_merged(H, edge_index_merged, edge_weight=edge_weight_merged)
            H_merged = F.elu(H_merged)
        else:
            H_merged = torch.zeros(num_nodes, self.out_channels, device=H.device)

        # Element-wise multiplication (Hadamard product ⊗)
        H1 = H_cat * H_merged
        H1 = self.dropout(H1)

        return H1
