"""
Attention-Relation Graph Convolution Module (Att-GCM)

Implements Equations 5-7 from RMGANets paper:
- Multi-head attention for edge weight computation
- Similarity-based subgraph splitting into high/medium/low
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AttentionGCM(nn.Module):
    """
    Attention-Relation Graph Convolution Module.

    Key features:
    - Multi-head attention (h̄=8 heads) for edge weighting
    - Subgraph splitting by similarity thresholds
    - BatchNorm1d preprocessing

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_heads: Number of attention heads (default: 8, from paper)
        T1: High similarity threshold (default: 0.7, from paper)
        T2: Low similarity threshold (default: 0.3, from paper)
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        T1: float = 0.7,
        T2: float = 0.3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.T1 = T1
        self.T2 = T2

        # Equation 1: BatchNorm preprocessing
        self.batch_norm = nn.BatchNorm1d(in_channels)

        # Multi-head attention components
        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        # Query, Key, Value projections for multi-head attention
        self.q_linear = nn.Linear(in_channels, out_channels)
        self.k_linear = nn.Linear(in_channels, out_channels)
        self.v_linear = nn.Linear(in_channels, out_channels)

        # Output projection
        self.out_linear = nn.Linear(out_channels, out_channels)

        # GCN layer for initial convolution (Equation 2)
        self.gcn = GCNConv(in_channels, out_channels)

        self.dropout = nn.Dropout(dropout)

    def compute_edge_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge weights using multi-head attention (Equations 5-6).

        A_ij = ||δ(x_i) - δ(x_j)||₂  where δ(·) is multi-head attention output

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            edge_weights: Edge weights [num_edges]
        """
        num_nodes = x.size(0)

        # Multi-head attention
        # Q, K, V projections
        Q = self.q_linear(x).view(num_nodes, self.num_heads, self.head_dim)  # [N, h, d]
        K = self.k_linear(x).view(num_nodes, self.num_heads, self.head_dim)  # [N, h, d]
        V = self.v_linear(x).view(num_nodes, self.num_heads, self.head_dim)  # [N, h, d]

        # Compute attention scores for all heads
        # attn_scores[i,j] = Q_i · K_j^T / sqrt(d)
        attn_scores = torch.einsum('nhd,mhd->nmh', Q, K) / (self.head_dim ** 0.5)  # [N, N, h]

        # Apply softmax per head
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, N, h]

        # Compute attention output: δ(x) = Σ attn_weights · V
        attn_output = torch.einsum('nmh,mhd->nhd', attn_weights, V)  # [N, h, d]
        attn_output = attn_output.reshape(num_nodes, -1)  # [N, h*d]

        # Equation 5: Compute edge weights as L2 distance
        row, col = edge_index
        edge_weights = torch.norm(attn_output[row] - attn_output[col], p=2, dim=1)  # [num_edges]

        # Normalize to [0, 1]
        edge_weights = edge_weights / (edge_weights.max() + 1e-8)

        return edge_weights

    def split_subgraphs(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split graph into high/medium/low similarity subgraphs (Equation 7).

        A^∇h_ij: A_ij ≥ T₁         (high similarity)
        A^∇m_ij: T₂ ≤ A_ij < T₁    (medium similarity)
        A^∇l_ij: A_ij < T₂         (low similarity)

        Args:
            edge_index: Edge connectivity [2, num_edges]
            edge_weights: Edge weights [num_edges]

        Returns:
            edge_index_high: High similarity edges
            edge_index_medium: Medium similarity edges
            edge_index_low: Low similarity edges
        """
        # Create masks for each similarity level
        mask_high = edge_weights >= self.T1
        mask_medium = (edge_weights >= self.T2) & (edge_weights < self.T1)
        mask_low = edge_weights < self.T2

        # Filter edges
        edge_index_high = edge_index[:, mask_high]
        edge_index_medium = edge_index[:, mask_medium]
        edge_index_low = edge_index[:, mask_low]

        return edge_index_high, edge_index_medium, edge_index_low

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass: Compute attention-based convolution and split subgraphs.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            H: Convolved features [num_nodes, out_channels]
            subgraphs: (edge_index_high, edge_index_medium, edge_index_low)
        """
        # Equation 1: BatchNorm preprocessing
        x_norm = self.batch_norm(x)

        # Equation 2: Initial graph convolution
        # H⁽¹⁾ = σ(Ã · Φ(x) · W)
        H = self.gcn(x_norm, edge_index)
        H = F.elu(H)
        H = self.dropout(H)

        # Equations 5-6: Compute edge weights via multi-head attention
        edge_weights = self.compute_edge_weights(x, edge_index)

        # Equation 7: Split into subgraphs by similarity
        edge_index_high, edge_index_medium, edge_index_low = self.split_subgraphs(
            edge_index, edge_weights
        )

        return H, (edge_index_high, edge_index_medium, edge_index_low)
