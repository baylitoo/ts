"""
Multi-Branch Loss Function for RMGANets

Implements Equation 14 from RMGANets paper with improvements:
ζ_total = ζ_M + λ·(ζ_a + ζ_b) + ε·ζ_d

Improvements over paper:
1. Focal Loss with class balancing for extreme imbalance
2. Temporal-aware DQN loss (weight recent transactions higher)
3. Adaptive loss weighting based on training progress
4. Subgraph-aware regularization
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses training on hard examples by down-weighting easy ones.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.25 from paper)
        gamma: Focusing parameter (default: 2.0, paper uses 0.2 for γ in total loss)
        reduction: Loss reduction method
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [batch_size, num_classes] (logits)
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemporalDQNLoss(nn.Module):
    """
    Temporal-aware DQN loss that weights recent transactions higher.

    Improvement over paper: Uses temporal features to prioritize recent patterns.

    Args:
        base_loss: Base loss function (MAE or Huber)
        temporal_decay: Decay factor for temporal weighting (0 = no decay)
    """

    def __init__(
        self,
        base_loss: str = 'mae',
        temporal_decay: float = 0.1
    ):
        super().__init__()
        self.temporal_decay = temporal_decay

        if base_loss == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: DQN predictions [batch_size, ...]
            targets: DQN targets [batch_size, ...]
            timestamps: Optional temporal weights [batch_size] (0=old, 1=recent)

        Returns:
            Weighted DQN loss
        """
        # Compute base loss
        loss = self.base_loss_fn(predictions, targets)

        # Apply temporal weighting if available
        if timestamps is not None and self.temporal_decay > 0:
            # Exponential decay: recent transactions weighted higher
            temporal_weights = torch.exp(self.temporal_decay * timestamps)
            temporal_weights = temporal_weights / temporal_weights.mean()  # Normalize
            loss = loss * temporal_weights.unsqueeze(-1) if loss.dim() > 1 else loss * temporal_weights

        return loss.mean()


class SubgraphRegularization(nn.Module):
    """
    Regularization based on subgraph split quality.

    Improvement over paper: Penalizes poor subgraph splits to encourage
    meaningful high/medium/low similarity separation.

    Args:
        target_high_ratio: Target proportion of high similarity edges (0.3-0.4)
        target_low_ratio: Target proportion of low similarity edges (0.2-0.3)
    """

    def __init__(
        self,
        target_high_ratio: float = 0.35,
        target_low_ratio: float = 0.25
    ):
        super().__init__()
        self.target_high = target_high_ratio
        self.target_low = target_low_ratio

    def forward(
        self,
        num_edges_high: int,
        num_edges_medium: int,
        num_edges_low: int
    ) -> torch.Tensor:
        """
        Args:
            num_edges_high: Number of high similarity edges
            num_edges_medium: Number of medium similarity edges
            num_edges_low: Number of low similarity edges

        Returns:
            Regularization loss penalizing imbalanced splits
        """
        total_edges = num_edges_high + num_edges_medium + num_edges_low

        if total_edges == 0:
            return torch.tensor(0.0)

        # Actual ratios
        ratio_high = num_edges_high / total_edges
        ratio_low = num_edges_low / total_edges

        # Penalize deviation from target
        loss_high = (ratio_high - self.target_high) ** 2
        loss_low = (ratio_low - self.target_low) ** 2

        return torch.tensor(loss_high + loss_low)


class MultiBranchLoss(nn.Module):
    """
    Multi-Branch Loss for RMGANets with improvements.

    Implements Equation 14 from paper:
    ζ_total = ζ_M + λ·(ζ_a + ζ_b) + ε·ζ_d + β·ζ_reg

    Where:
    - ζ_M: Main classification loss (Cross-Entropy)
    - ζ_a: To-GCM branch loss (Focal Loss for hard examples)
    - ζ_b: HyGCM branch loss (Binary Cross-Entropy)
    - ζ_d: DQN enhancement loss (Temporal-aware MAE)
    - ζ_reg: Subgraph regularization (NEW!)

    Improvements over paper:
    1. Focal Loss with class balancing
    2. Temporal-aware DQN loss
    3. Adaptive loss weighting
    4. Subgraph quality regularization

    Args:
        lambda_branch: Weight for branch losses (default: 0.25 from paper)
        epsilon_dqn: Weight for DQN loss (default: 0.4 from paper)
        beta_reg: Weight for regularization (default: 0.1, NEW!)
        gamma_focal: Focal loss gamma (default: 0.2 from paper)
        adaptive_weighting: Use adaptive loss weighting based on training progress
        temporal_weighting: Use temporal weighting for DQN loss
        temporal_decay: Exponential decay factor for temporal weighting
    """

    def __init__(
        self,
        lambda_branch: float = 0.25,
        epsilon_dqn: float = 0.4,
        beta_reg: float = 0.1,
        gamma_focal: float = 0.2,
        adaptive_weighting: bool = True,
        temporal_weighting: bool = True,
        temporal_decay: float = 0.1
    ):
        super().__init__()

        # Loss weights (from paper)
        self.lambda_branch = lambda_branch
        self.epsilon_dqn = epsilon_dqn
        self.beta_reg = beta_reg

        # Adaptive weighting
        self.adaptive_weighting = adaptive_weighting
        self.temporal_weighting = temporal_weighting
        self.temporal_decay = temporal_decay

        # Loss components
        self.ce_loss = nn.CrossEntropyLoss()  # Main loss (ζ_M)
        self.focal_loss = FocalLoss(gamma=gamma_focal)  # To-GCM branch (ζ_a)
        self.bce_loss = nn.BCEWithLogitsLoss()  # HyGCM branch (ζ_b)
        self.dqn_loss = TemporalDQNLoss(
            base_loss='mae',
            temporal_decay=temporal_decay if temporal_weighting else 0.0
        )  # DQN enhancement (ζ_d)
        self.subgraph_reg = SubgraphRegularization()  # Regularization (ζ_reg)

        # Tracking for adaptive weighting
        self.register_buffer('epoch', torch.tensor(0))
        self.register_buffer('loss_history', torch.zeros(4))  # Track last 4 loss components

    def forward(
        self,
        outputs_main: torch.Tensor,
        outputs_to: torch.Tensor,
        outputs_hy: torch.Tensor,
        dqn_predictions: torch.Tensor,
        dqn_targets: torch.Tensor,
        targets: torch.Tensor,
        subgraph_stats: Optional[Dict[str, int]] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-branch loss.

        Args:
            outputs_main: Main classification output [batch_size, num_classes]
            outputs_to: To-GCM branch output [batch_size, num_classes]
            outputs_hy: HyGCM branch output [batch_size, 1] (binary)
            dqn_predictions: DQN predicted Q-values [batch_size, ...]
            dqn_targets: DQN target Q-values [batch_size, ...]
            targets: Ground truth labels [batch_size]
            subgraph_stats: Dict with 'num_high', 'num_medium', 'num_low' edges
            timestamps: Optional temporal weights [batch_size]

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # 1. Main classification loss (ζ_M)
        loss_main = self.ce_loss(outputs_main, targets)

        # 2. To-GCM branch loss (ζ_a) - Focal Loss for hard examples
        loss_focal = self.focal_loss(outputs_to, targets)

        # 3. HyGCM branch loss (ζ_b) - Binary Cross-Entropy
        # Convert targets to binary (fraud vs non-fraud)
        targets_binary = (targets > 0).float().unsqueeze(1)
        loss_bce = self.bce_loss(outputs_hy, targets_binary)

        # 4. DQN enhancement loss (ζ_d) - Temporal-aware MAE
        loss_dqn = self.dqn_loss(dqn_predictions, dqn_targets, timestamps)

        # 5. Subgraph regularization (ζ_reg) - NEW!
        loss_reg = torch.tensor(0.0, device=outputs_main.device)
        if subgraph_stats is not None:
            loss_reg = self.subgraph_reg(
                subgraph_stats.get('num_high', 0),
                subgraph_stats.get('num_medium', 0),
                subgraph_stats.get('num_low', 0)
            ).to(outputs_main.device)

        # Adaptive loss weighting (if enabled)
        lambda_eff = self.lambda_branch
        epsilon_eff = self.epsilon_dqn
        beta_eff = self.beta_reg

        if self.adaptive_weighting and self.epoch > 10:
            # Gradually reduce branch weights as training progresses
            # Early: focus on branches (exploration)
            # Late: focus on main loss (exploitation)
            decay_factor = torch.exp(-self.epoch / 100.0)
            lambda_eff = self.lambda_branch * (0.5 + 0.5 * decay_factor)
            epsilon_eff = self.epsilon_dqn * (0.5 + 0.5 * decay_factor)

        # Equation 14: Total loss
        total_loss = (
            loss_main +
            lambda_eff * (loss_focal + loss_bce) +
            epsilon_eff * loss_dqn +
            beta_eff * loss_reg
        )

        # Update tracking
        self.loss_history = torch.tensor([
            loss_main.item(),
            loss_focal.item(),
            loss_bce.item(),
            loss_dqn.item()
        ])

        # Return loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'main_ce': loss_main.item(),
            'to_gcm_focal': loss_focal.item(),
            'hy_gcm_bce': loss_bce.item(),
            'dqn_mae': loss_dqn.item(),
            'subgraph_reg': loss_reg.item(),
            'lambda_eff': lambda_eff if isinstance(lambda_eff, float) else lambda_eff.item(),
            'epsilon_eff': epsilon_eff if isinstance(epsilon_eff, float) else epsilon_eff.item()
        }

        return total_loss, loss_dict

    def step_epoch(self) -> None:
        """Increment epoch counter for adaptive weighting"""
        self.epoch += 1

    def get_loss_statistics(self) -> Dict[str, float]:
        """Get statistics on loss components"""
        return {
            'mean_main': self.loss_history[0].item(),
            'mean_focal': self.loss_history[1].item(),
            'mean_bce': self.loss_history[2].item(),
            'mean_dqn': self.loss_history[3].item()
        }


class SimplifiedMultiBranchLoss(nn.Module):
    """
    Simplified Multi-Branch Loss without adaptive weighting.

    Use this if you want the exact paper implementation without improvements.

    Args:
        lambda_branch: Weight for branch losses (default: 0.25)
        epsilon_dqn: Weight for DQN loss (default: 0.4)
    """

    def __init__(
        self,
        lambda_branch: float = 0.25,
        epsilon_dqn: float = 0.4
    ):
        super().__init__()

        self.lambda_branch = lambda_branch
        self.epsilon_dqn = epsilon_dqn

        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=0.2)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        outputs_main: torch.Tensor,
        outputs_to: torch.Tensor,
        outputs_hy: torch.Tensor,
        dqn_predictions: torch.Tensor,
        dqn_targets: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-branch loss (paper version)"""

        # Main loss
        loss_main = self.ce_loss(outputs_main, targets)

        # Branch losses
        loss_focal = self.focal_loss(outputs_to, targets)
        targets_binary = (targets > 0).float().unsqueeze(1)
        loss_bce = self.bce_loss(outputs_hy, targets_binary)

        # DQN loss
        loss_dqn = self.mae_loss(dqn_predictions, dqn_targets)

        # Total loss (Equation 14)
        total_loss = (
            loss_main +
            self.lambda_branch * (loss_focal + loss_bce) +
            self.epsilon_dqn * loss_dqn
        )

        loss_dict = {
            'total': total_loss.item(),
            'main': loss_main.item(),
            'focal': loss_focal.item(),
            'bce': loss_bce.item(),
            'dqn': loss_dqn.item()
        }

        return total_loss, loss_dict


