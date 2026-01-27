"""
RMGANets Encoder - Main Integration Module

Combines all RMGANets components:
1. Att-GCM: Multi-head attention with subgraph splitting
2. To-GCM: Adaptive topology convolution
3. HyGCM: Hybrid enhanced convolution
4. Feature Fusion: Critical +11% F1 gain component
5. DQN Enhancement: Feature refinement (optional)

Designed to integrate seamlessly with existing StateEncoder.
"""

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from .att_gcm import AttentionGCM
from .to_gcm import AdaptiveTopoGCM
from .hy_gcm import HybridEnhancedGCM
from .fusion import FeatureFusionModule


class RMGANetsEncoder(nn.Module):
    """
    RMGANets encoder architecture.

    Implements the full RMGANets pipeline:
    - Att-GCM + subgraph splitting
    - Parallel To-GCM and HyGCM on different subgraphs
    - Feature fusion (critical component)
    - Optional DQN enhancement

    Args:
        node_feature_dim: Input node feature dimension
        hidden_dim: Hidden dimension (default: 160 from paper)
        embedding_dim: Final embedding dimension
        num_att_heads: Number of attention heads (default: 8 from paper)
        T1: High similarity threshold (default: 0.7)
        T2: Low similarity threshold (default: 0.3)
        dropout: Dropout rate
        use_dqn_enhancement: Whether to use DQN feature enhancement
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 160,
        embedding_dim: int = 64,
        num_att_heads: int = 8,
        T1: float = 0.7,
        T2: float = 0.3,
        dropout: float = 0.1,
        use_dqn_enhancement: bool = False
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_dqn_enhancement = use_dqn_enhancement

        # Module 1: Att-GCM with subgraph splitting
        self.att_gcm = AttentionGCM(
            in_channels=node_feature_dim,
            out_channels=hidden_dim,
            num_heads=num_att_heads,
            T1=T1,
            T2=T2,
            dropout=dropout
        )

        # Module 2: To-GCM (processes medium & high similarity)
        self.to_gcm = AdaptiveTopoGCM(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dropout=dropout
        )

        # Module 3: HyGCM (processes high & low similarity)
        self.hy_gcm = HybridEnhancedGCM(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_heads=4,
            dropout=dropout
        )

        # Feature Fusion Module (Equation 12) - Critical component
        self.fusion = FeatureFusionModule(
            to_gcm_dim=hidden_dim,
            hy_gcm_dim=hidden_dim,
            att_gcm_dim=hidden_dim,
            fusion_dim=hidden_dim * 2,  # Expand for richer representation
            dropout=dropout
        )

        # Final projection to embedding dimension
        self.output_projection = nn.Linear(hidden_dim * 2, embedding_dim)

        # Optional DQN enhancement layer (for future integration)
        self.dqn_enhance: Optional[nn.Linear] = (
            nn.Linear(embedding_dim, embedding_dim) if use_dqn_enhancement else None
        )

        # Layer normalization for final output
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tensor:
        """
        Forward pass through RMGANets architecture.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            embeddings: Node embeddings [num_nodes, embedding_dim]
        """
        # Step 1: Att-GCM + subgraph splitting (Equations 1-7)
        H, (edge_index_high, edge_index_medium, edge_index_low) = self.att_gcm(x, edge_index)
        H = cast(Tensor, H)
        edge_index_high = cast(Tensor, edge_index_high)
        edge_index_medium = cast(Tensor, edge_index_medium)
        edge_index_low = cast(Tensor, edge_index_low)

        # Step 2: To-GCM on medium & high similarity (Equations 8-9)
        H1 = cast(Tensor, self.to_gcm(H, edge_index_high, edge_index_medium))

        # Step 3: HyGCM on high & low similarity (Equations 10-11)
        H2 = cast(Tensor, self.hy_gcm(H, edge_index_high, edge_index_low))

        # Step 4: Feature Fusion (Equation 12) - CRITICAL (+11% F1)
        H_fused = cast(Tensor, self.fusion(H1, H2, H))

        # Step 5: Project to embedding dimension
        embeddings = self.output_projection(H_fused)

        # Optional DQN enhancement
        if self.use_dqn_enhancement and self.dqn_enhance is not None:
            embeddings = embeddings + self.dqn_enhance(embeddings)  # Residual connection

        # Final normalization
        embeddings = self.output_norm(embeddings)

        return cast(Tensor, embeddings)

        return cast(Tensor, embeddings)

        return embeddings

    def get_subgraph_info(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get subgraph splits for analysis.

        Args:
            x: Node features
            edge_index: Edge connectivity

        Returns:
            edge_index_high: High similarity edges
            edge_index_medium: Medium similarity edges
            edge_index_low: Low similarity edges
        """
        with torch.no_grad():
            _, (edge_index_high, edge_index_medium, edge_index_low) = self.att_gcm(x, edge_index)

        return edge_index_high, edge_index_medium, edge_index_low
