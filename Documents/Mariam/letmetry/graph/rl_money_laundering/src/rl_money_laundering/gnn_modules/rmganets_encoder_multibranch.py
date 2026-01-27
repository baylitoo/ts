"""
RMGANets Encoder with Multi-Branch Outputs

Enhanced version that outputs auxiliary predictions for multi-branch loss.
Supports both standard mode (single output) and multi-branch mode (4 outputs).
"""

from typing import Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .att_gcm import AttentionGCM
from .to_gcm import AdaptiveTopoGCM
from .hy_gcm import HybridEnhancedGCM
from .fusion import FeatureFusionModule


class RMGANetsMultiBranchEncoder(nn.Module):
    """
    RMGANets encoder with multi-branch outputs for advanced loss.

    Outputs:
    1. Main embeddings (always)
    2. To-GCM branch predictions (if multi_branch=True)
    3. HyGCM branch predictions (if multi_branch=True)
    4. Subgraph statistics (if multi_branch=True)

    This enables multi-branch loss training while remaining backward compatible.

    Args:
        node_feature_dim: Input node feature dimension
        hidden_dim: Hidden dimension (default: 160)
        embedding_dim: Final embedding dimension
        num_classes: Number of output classes (for auxiliary heads)
        num_att_heads: Number of attention heads (default: 8)
        T1: High similarity threshold (default: 0.7)
        T2: Low similarity threshold (default: 0.3)
        dropout: Dropout rate
        multi_branch: Enable multi-branch outputs
        use_dqn_enhancement: Use DQN enhancement layer
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 160,
        embedding_dim: int = 64,
        num_classes: int = 2,  # Binary fraud detection
        num_att_heads: int = 8,
        T1: float = 0.7,
        T2: float = 0.3,
        dropout: float = 0.1,
        multi_branch: bool = False,
        use_dqn_enhancement: bool = False
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.multi_branch = multi_branch
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

        # Module 4: Feature Fusion
        self.fusion = FeatureFusionModule(
            to_gcm_dim=hidden_dim,
            hy_gcm_dim=hidden_dim,
            att_gcm_dim=hidden_dim,
            fusion_dim=hidden_dim * 2,
            dropout=dropout
        )

        # Final projection to embedding dimension
        self.output_projection = nn.Linear(hidden_dim * 2, embedding_dim)

        # Output normalization
        self.output_norm = nn.LayerNorm(embedding_dim)

        # Multi-branch auxiliary heads (only if multi_branch=True)
        if multi_branch:
            # To-GCM auxiliary classifier (for Focal Loss)
            self.to_branch_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )

            # HyGCM auxiliary classifier (for BCE Loss - binary)
            self.hy_branch_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)  # Binary output
            )

        # Optional DQN enhancement
        self.dqn_enhance: Optional[nn.Linear] = (
            nn.Linear(embedding_dim, embedding_dim) if use_dqn_enhancement else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_auxiliary: Optional[bool] = None
    ) -> Union[
        Tensor,
        Tuple[Tensor, Dict[str, Union[Tensor, Dict[str, int]]]]
    ]:
        """
        Forward pass with optional multi-branch outputs.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            return_auxiliary: Override multi_branch setting (None = use self.multi_branch)

        Returns:
            If multi_branch=False or return_auxiliary=False:
                embeddings: Node embeddings [num_nodes, embedding_dim]

            If multi_branch=True or return_auxiliary=True:
                embeddings: Node embeddings [num_nodes, embedding_dim]
                auxiliary: Dict containing:
                    - 'to_branch': To-GCM predictions [num_nodes, num_classes]
                    - 'hy_branch': HyGCM predictions [num_nodes, 1]
                    - 'subgraph_stats': Dict with edge counts
                    - 'H1': To-GCM features (for analysis)
                    - 'H2': HyGCM features (for analysis)
        """
        # Determine whether to return auxiliary outputs
        return_aux = return_auxiliary if return_auxiliary is not None else self.multi_branch

        # Step 1: Att-GCM + subgraph splitting
        H, (edge_index_high, edge_index_medium, edge_index_low) = self.att_gcm(x, edge_index)
        H = cast(Tensor, H)
        edge_index_high = cast(Tensor, edge_index_high)
        edge_index_medium = cast(Tensor, edge_index_medium)
        edge_index_low = cast(Tensor, edge_index_low)

        # Step 2: To-GCM on medium & high similarity
        H1 = cast(Tensor, self.to_gcm(H, edge_index_high, edge_index_medium))

        # Step 3: HyGCM on high & low similarity
        H2 = cast(Tensor, self.hy_gcm(H, edge_index_high, edge_index_low))

        # Step 4: Feature Fusion (CRITICAL)
        H_fused = cast(Tensor, self.fusion(H1, H2, H))

        # Step 5: Project to embedding dimension
        embeddings = cast(Tensor, self.output_projection(H_fused))

        # Optional DQN enhancement
        if self.use_dqn_enhancement and self.dqn_enhance is not None:
            embeddings = embeddings + self.dqn_enhance(embeddings)

        # Final normalization
        embeddings = cast(Tensor, self.output_norm(embeddings))

        # Return simple embeddings if not multi-branch
        if not return_aux:
            return embeddings

        # Compute auxiliary outputs
        auxiliary: Dict[str, Union[Tensor, Dict[str, int]]] = {}

        # To-GCM branch predictions (for Focal Loss)
        if hasattr(self, 'to_branch_head'):
            auxiliary['to_branch'] = cast(Tensor, self.to_branch_head(H1))
        else:
            # Fallback: use simple linear layer
            random_weight = torch.randn(
                self.num_classes, self.hidden_dim, device=H1.device, dtype=H1.dtype
            )
            auxiliary['to_branch'] = F.linear(H1, random_weight)

        # HyGCM branch predictions (for BCE Loss)
        if hasattr(self, 'hy_branch_head'):
            auxiliary['hy_branch'] = cast(Tensor, self.hy_branch_head(H2))
        else:
            # Fallback: use simple linear layer
            random_weight_hy = torch.randn(
                1, self.hidden_dim, device=H2.device, dtype=H2.dtype
            )
            auxiliary['hy_branch'] = F.linear(H2, random_weight_hy)

        # Subgraph statistics
            auxiliary['subgraph_stats'] = {
                'num_high': int(edge_index_high.size(1)),
                'num_medium': int(edge_index_medium.size(1)),
                'num_low': int(edge_index_low.size(1)),
                'total': int(edge_index.size(1))
            }

        # Features for analysis
        auxiliary['H1'] = H1  # To-GCM features
        auxiliary['H2'] = H2  # HyGCM features
        auxiliary['H'] = H    # Att-GCM features
        return embeddings, auxiliary

    def get_subgraph_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, int]:
        """
        Get subgraph statistics without computing full forward pass.

        Useful for monitoring training.

        Args:
            x: Node features
            edge_index: Edge connectivity

        Returns:
            Dict with 'num_high', 'num_medium', 'num_low', 'total'
        """
        with torch.no_grad():
            _, (edge_index_high, edge_index_medium, edge_index_low) = self.att_gcm(x, edge_index)

        return {
            'num_high': edge_index_high.size(1),
            'num_medium': edge_index_medium.size(1),
            'num_low': edge_index_low.size(1),
            'total': edge_index.size(1),
            'ratio_high': edge_index_high.size(1) / max(edge_index.size(1), 1),
            'ratio_medium': edge_index_medium.size(1) / max(edge_index.size(1), 1),
            'ratio_low': edge_index_low.size(1) / max(edge_index.size(1), 1)
        }

    def enable_multi_branch(self) -> None:
        """Enable multi-branch mode"""
        self.multi_branch = True

    def disable_multi_branch(self) -> None:
        """Disable multi-branch mode (back to standard mode)"""
        self.multi_branch = False


# Backward compatibility: alias for standard encoder
class RMGANetsEncoder(RMGANetsMultiBranchEncoder):
    """
    Standard RMGANets encoder (backward compatible).

    Defaults to multi_branch=False for compatibility with existing code.
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
        super().__init__(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_classes=2,
            num_att_heads=num_att_heads,
            T1=T1,
            T2=T2,
            dropout=dropout,
            multi_branch=False,  # Default to standard mode
            use_dqn_enhancement=use_dqn_enhancement
        )
