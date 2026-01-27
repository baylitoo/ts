"""
Feature Fusion Module

Implements Equation 12 from RMGANets paper:
H_final = Cat(H1, H2, H)

This is the CRITICAL component that provides +11% F1 gain according to ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module - Combines outputs from all 3 GCM modules.

    According to RMGANets ablation studies (Tables 4-6):
    - Individual modules: F1 ~78-79%
    - Fusion of all 3:    F1 =90.43% (+11.66% improvement!)

    This is the most critical component of RMGANets.

    Args:
        to_gcm_dim: Output dimension of To-GCM (H1)
        hy_gcm_dim: Output dimension of HyGCM (H2)
        att_gcm_dim: Output dimension of Att-GCM (H)
        fusion_dim: Output dimension after fusion
        dropout: Dropout rate
    """

    def __init__(
        self,
        to_gcm_dim: int,
        hy_gcm_dim: int,
        att_gcm_dim: int,
        fusion_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.to_gcm_dim = to_gcm_dim
        self.hy_gcm_dim = hy_gcm_dim
        self.att_gcm_dim = att_gcm_dim

        # Total concatenated dimension
        self.concat_dim = to_gcm_dim + hy_gcm_dim + att_gcm_dim

        # Fusion projection layer
        self.fusion_linear = nn.Linear(self.concat_dim, fusion_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(fusion_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H1: torch.Tensor,  # From To-GCM
        H2: torch.Tensor,  # From HyGCM
        H: torch.Tensor    # From Att-GCM
    ) -> torch.Tensor:
        """
        Fuse features from all three modules (Equation 12).

        H_final = Cat(H1, H2, H)

        Args:
            H1: Features from To-GCM [num_nodes, to_gcm_dim]
            H2: Features from HyGCM [num_nodes, hy_gcm_dim]
            H: Features from Att-GCM [num_nodes, att_gcm_dim]

        Returns:
            H_fused: Fused features [num_nodes, fusion_dim]
        """
        # Equation 12: Concatenate all features
        H_concat = torch.cat([H1, H2, H], dim=1)  # [num_nodes, concat_dim]

        # Project to fusion dimension
        H_fused = self.fusion_linear(H_concat)

        # Apply layer normalization
        H_fused = self.layer_norm(H_fused)

        # Activation and dropout
        H_fused = F.leaky_relu(H_fused, negative_slope=0.2)
        H_fused = self.dropout(H_fused)

        return H_fused
