"""
Hybrid Enhanced Graph Convolution Module (HyGCM)

Implements Equations 10-11 from RMGANets paper:
- Hybrid processing of high and low similarity subgraphs
- Local attention embedding for high similarity
- Standard GCN for low similarity (global context)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class HybridEnhancedGCM(nn.Module):
    """
    Hybrid Enhanced Graph Convolution Module.

    Key features:
    - Local attention embedding for high similarity subgraph
    - GCN convolution for low similarity subgraph (global context)
    - Hybrid feature fusion

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_heads: Number of attention heads for local attention
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # Local attention module for high similarity subgraph
        self.local_attention = GATConv(
            in_channels,
            out_channels // 2,
            heads=num_heads,
            dropout=dropout,
            concat=False  # Average multi-head outputs
        )

        # GCN for low similarity subgraph (global context)
        self.global_conv = GCNConv(in_channels, out_channels // 2)

        # Merged convolution for hybrid processing
        self.conv_hybrid = GCNConv(in_channels, out_channels)

        # Local Attention Module (LAM) - Equation 11
        self.lam_weight = nn.Parameter(torch.FloatTensor([0.5]))  # α parameter

        self.dropout = nn.Dropout(dropout)

    def local_attention_module(
        self,
        h_low: torch.Tensor,
        h_high: torch.Tensor
    ) -> torch.Tensor:
        """
        Local Attention Module (Equation 11).

        LAM = h_low + α · h_high

        Args:
            h_low: Features from low similarity processing
            h_high: Features from high similarity processing

        Returns:
            LAM output
        """
        # Ensure dimensions match
        if h_low.size(1) != h_high.size(1):
            # Pad smaller dimension
            if h_low.size(1) < h_high.size(1):
                pad_size = h_high.size(1) - h_low.size(1)
                h_low = F.pad(h_low, (0, pad_size))
            else:
                pad_size = h_low.size(1) - h_high.size(1)
                h_high = F.pad(h_high, (0, pad_size))

        # Apply LAM: h_low + α · h_high
        lam_output = h_low + self.lam_weight * h_high

        return lam_output

    def merge_high_low(
        self,
        edge_index_high: torch.Tensor,
        edge_index_low: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge high and low similarity edges (Equation 10: A^∇hl).

        Args:
            edge_index_high: High similarity edges
            edge_index_low: Low similarity edges

        Returns:
            edge_index_merged: Merged edges
        """
        # Simply concatenate (paper doesn't specify special weighting)
        edge_index_merged = torch.cat([edge_index_high, edge_index_low], dim=1)

        # Remove duplicate edges
        edge_index_merged = torch.unique(edge_index_merged, dim=1)

        return edge_index_merged

    def forward(
        self,
        H: torch.Tensor,
        edge_index_high: torch.Tensor,
        edge_index_low: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: Hybrid convolution on high/low similarity subgraphs.

        Equation 10: H₂ = σ(A^∇hl ⊙ H ⊙ W₄) + LAM([...])

        Args:
            H: Input features from Att-GCM [num_nodes, in_channels]
            edge_index_high: High similarity edges
            edge_index_low: Low similarity edges

        Returns:
            H2: Output features [num_nodes, out_channels]
        """
        num_nodes = H.size(0)

        # Process high similarity with local attention
        if edge_index_high.size(1) > 0:
            h_high = self.local_attention(H, edge_index_high)
            h_high = F.elu(h_high)
        else:
            h_high = torch.zeros(num_nodes, self.out_channels // 2, device=H.device)

        # Process low similarity with GCN (global context)
        if edge_index_low.size(1) > 0:
            h_low = self.global_conv(H, edge_index_low)
            h_low = F.elu(h_low)
        else:
            h_low = torch.zeros(num_nodes, self.out_channels // 2, device=H.device)

        # Apply Local Attention Module (Equation 11)
        h_lam = self.local_attention_module(h_low, h_high)

        # Merge high and low subgraphs
        edge_index_merged = self.merge_high_low(edge_index_high, edge_index_low)

        # Hybrid convolution on merged subgraph (Equation 10)
        if edge_index_merged.size(1) > 0:
            H_hybrid = self.conv_hybrid(H, edge_index_merged)
            H_hybrid = F.elu(H_hybrid)
        else:
            H_hybrid = torch.zeros(num_nodes, self.out_channels, device=H.device)

        # Concatenate LAM output with hybrid convolution
        # Ensure dimensions match
        if h_lam.size(1) < self.out_channels:
            h_lam = F.pad(h_lam, (0, self.out_channels - h_lam.size(1)))
        elif h_lam.size(1) > self.out_channels:
            h_lam = h_lam[:, :self.out_channels]

        # Final output: H₂ = H_hybrid + LAM
        H2 = H_hybrid + h_lam
        H2 = self.dropout(H2)

        return H2
