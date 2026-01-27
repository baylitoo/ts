"""
Ranking losses for pairwise and listwise learning-to-rank.

These losses are used to train the pairwise ranker for prioritizing
investigation queues based on suspiciousness.

Supported losses:
1. Margin Ranking Loss (pairwise)
2. Contrastive Loss (pairwise)
3. ListNet Loss (listwise)
4. LambdaRank Loss (listwise, NDCG-aware)

Usage:
    from rl_money_laundering.pretraining.ranking_loss import (
        MarginRankingLoss,
        ContrastiveLoss,
        ListNetLoss,
        create_ranking_loss,
    )

    loss_fn = MarginRankingLoss(margin=0.5)
    loss = loss_fn(score_a, score_b, labels)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RankingLossType(Enum):
    """Types of ranking losses."""
    MARGIN = "margin"
    CONTRASTIVE = "contrastive"
    LISTNET = "listnet"
    LAMBDARANK = "lambdarank"
    TRIPLET = "triplet"


@dataclass
class RankingLossConfig:
    """Configuration for ranking losses."""

    loss_type: RankingLossType = RankingLossType.MARGIN
    margin: float = 0.5  # For margin-based losses
    temperature: float = 1.0  # For softmax-based losses
    reduction: str = "mean"  # "mean", "sum", "none"


if HAS_TORCH:

    class MarginRankingLoss(nn.Module):
        """Margin ranking loss for pairwise comparisons.

        Given scores for pairs (A, B) and labels indicating which is greater,
        enforces that the score difference exceeds a margin.

        Loss = max(0, margin - label * (score_a - score_b))

        When label=+1: A should score higher than B by at least margin.
        When label=-1: B should score higher than A by at least margin.
        """

        def __init__(
            self,
            margin: float = 0.5,
            reduction: str = "mean",
        ) -> None:
            """Initialize margin ranking loss.

            Args:
                margin: Minimum score difference required.
                reduction: How to reduce batch dimension.
            """
            super().__init__()
            self.margin = margin
            self.reduction = reduction

        def forward(
            self,
            score_a: torch.Tensor,
            score_b: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            """Compute margin ranking loss.

            Args:
                score_a: Scores for items A [batch].
                score_b: Scores for items B [batch].
                labels: +1 if A > B, -1 if B > A [batch].

            Returns:
                Scalar loss value.
            """
            return F.margin_ranking_loss(
                score_a,
                score_b,
                labels.float(),
                margin=self.margin,
                reduction=self.reduction,
            )

    class ContrastiveLoss(nn.Module):
        """Contrastive loss for learning discriminative embeddings.

        Similar to margin ranking but with exponential penalty.

        For similar pairs (label=1):
            Loss = distance^2

        For dissimilar pairs (label=0):
            Loss = max(0, margin - distance)^2
        """

        def __init__(
            self,
            margin: float = 1.0,
            reduction: str = "mean",
        ) -> None:
            """Initialize contrastive loss.

            Args:
                margin: Minimum distance for dissimilar pairs.
                reduction: How to reduce batch dimension.
            """
            super().__init__()
            self.margin = margin
            self.reduction = reduction

        def forward(
            self,
            embedding_a: torch.Tensor,
            embedding_b: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            """Compute contrastive loss.

            Args:
                embedding_a: Embeddings for items A [batch, hidden].
                embedding_b: Embeddings for items B [batch, hidden].
                labels: 1 if similar, 0 if dissimilar [batch].

            Returns:
                Scalar loss value.
            """
            # Euclidean distance
            distances = F.pairwise_distance(embedding_a, embedding_b)

            # Similar pairs: minimize distance
            similar_loss = labels.float() * distances.pow(2)

            # Dissimilar pairs: enforce margin
            dissimilar_loss = (1 - labels.float()) * F.relu(
                self.margin - distances
            ).pow(2)

            loss = similar_loss + dissimilar_loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss

    class TripletMarginLoss(nn.Module):
        """Triplet margin loss for anchor-positive-negative comparisons.

        Enforces that anchor is closer to positive than negative by margin.

        Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
        """

        def __init__(
            self,
            margin: float = 1.0,
            p: int = 2,
            reduction: str = "mean",
        ) -> None:
            """Initialize triplet margin loss.

            Args:
                margin: Minimum difference between distances.
                p: Norm degree for distance (default: 2 for Euclidean).
                reduction: How to reduce batch dimension.
            """
            super().__init__()
            self.margin = margin
            self.p = p
            self.reduction = reduction

        def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor,
        ) -> torch.Tensor:
            """Compute triplet margin loss.

            Args:
                anchor: Anchor embeddings [batch, hidden].
                positive: Positive embeddings [batch, hidden].
                negative: Negative embeddings [batch, hidden].

            Returns:
                Scalar loss value.
            """
            return F.triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=self.margin,
                p=self.p,
                reduction=self.reduction,
            )

    class ListNetLoss(nn.Module):
        """ListNet loss for listwise learning-to-rank.

        Treats ranking as a probability distribution over permutations.
        Minimizes cross-entropy between predicted and ground truth distributions.

        This is more effective than pairwise losses when you have
        multiple items to rank simultaneously.
        """

        def __init__(
            self,
            temperature: float = 1.0,
            reduction: str = "mean",
        ) -> None:
            """Initialize ListNet loss.

            Args:
                temperature: Temperature for softmax (lower = sharper).
                reduction: How to reduce batch dimension.
            """
            super().__init__()
            self.temperature = temperature
            self.reduction = reduction

        def forward(
            self,
            scores: torch.Tensor,
            relevance: torch.Tensor,
        ) -> torch.Tensor:
            """Compute ListNet loss.

            Args:
                scores: Predicted scores [batch, num_items].
                relevance: Ground truth relevance scores [batch, num_items].

            Returns:
                Scalar loss value.
            """
            # Convert to probability distributions
            pred_probs = F.softmax(scores / self.temperature, dim=-1)
            true_probs = F.softmax(relevance / self.temperature, dim=-1)

            # Cross-entropy loss
            loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=-1)

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss

    class LambdaRankLoss(nn.Module):
        """LambdaRank loss for NDCG-aware learning-to-rank.

        Weights pairwise comparisons by their impact on NDCG.
        More effective at optimizing ranking metrics directly.
        """

        def __init__(
            self,
            sigma: float = 1.0,
            reduction: str = "mean",
            k: Optional[int] = None,  # NDCG@k
        ) -> None:
            """Initialize LambdaRank loss.

            Args:
                sigma: Steepness of sigmoid in lambda gradient.
                reduction: How to reduce batch dimension.
                k: Cutoff for NDCG calculation (None = full list).
            """
            super().__init__()
            self.sigma = sigma
            self.reduction = reduction
            self.k = k

        def forward(
            self,
            scores: torch.Tensor,
            relevance: torch.Tensor,
        ) -> torch.Tensor:
            """Compute LambdaRank loss.

            Args:
                scores: Predicted scores [batch, num_items].
                relevance: Ground truth relevance scores [batch, num_items].

            Returns:
                Scalar loss value.
            """
            batch_size, num_items = scores.shape
            device = scores.device

            # Compute NDCG weights
            dcg_weights = self._compute_dcg_weights(relevance)

            # Compute pairwise score differences
            score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [batch, i, j]

            # Compute pairwise relevance differences
            rel_diff = relevance.unsqueeze(2) - relevance.unsqueeze(1)  # [batch, i, j]

            # Only consider pairs where i is more relevant than j
            mask = (rel_diff > 0).float()

            # Lambda gradient: sigmoid of score difference
            lambda_ij = torch.sigmoid(-self.sigma * score_diff)

            # Weight by NDCG impact
            delta_ndcg = self._compute_delta_ndcg(relevance, dcg_weights)

            # Compute loss
            loss = lambda_ij * delta_ndcg.abs() * mask

            # Sum over pairs
            loss = loss.sum(dim=[1, 2])

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss

        def _compute_dcg_weights(self, relevance: torch.Tensor) -> torch.Tensor:
            """Compute DCG discount weights."""
            batch_size, num_items = relevance.shape
            device = relevance.device

            # Sort by relevance to get ideal ranking
            _, ideal_order = torch.sort(relevance, dim=1, descending=True)

            # DCG discount: 1 / log2(rank + 1)
            ranks = torch.arange(1, num_items + 1, device=device, dtype=torch.float)
            discounts = 1.0 / torch.log2(ranks + 1)

            return discounts

        def _compute_delta_ndcg(
            self,
            relevance: torch.Tensor,
            dcg_weights: torch.Tensor,
        ) -> torch.Tensor:
            """Compute change in NDCG from swapping positions i and j."""
            batch_size, num_items = relevance.shape
            device = relevance.device

            # Sort to get current ranking
            _, current_order = torch.sort(relevance, dim=1, descending=True)

            # Compute NDCG impact of swapping each pair
            # Simplified: use gain difference * discount difference
            gains = (2.0 ** relevance - 1).unsqueeze(2) - (2.0 ** relevance - 1).unsqueeze(1)
            discount_diff = dcg_weights.unsqueeze(1) - dcg_weights.unsqueeze(0)

            delta_ndcg = gains * discount_diff.unsqueeze(0)

            return delta_ndcg

    class MultiObjectiveRankingLoss(nn.Module):
        """Combines multiple ranking losses with learnable weights.

        Useful when you want to optimize for multiple ranking objectives
        simultaneously (e.g., NDCG and MAP).
        """

        def __init__(
            self,
            losses: List[nn.Module],
            weights: Optional[List[float]] = None,
            learnable_weights: bool = False,
        ) -> None:
            """Initialize multi-objective loss.

            Args:
                losses: List of loss modules.
                weights: Initial weights for each loss.
                learnable_weights: Whether to learn the weights.
            """
            super().__init__()
            self.losses = nn.ModuleList(losses)
            self.num_losses = len(losses)

            if weights is None:
                weights = [1.0 / self.num_losses] * self.num_losses

            if learnable_weights:
                # Log-space for positivity
                self.log_weights = nn.Parameter(torch.log(torch.tensor(weights)))
            else:
                self.register_buffer(
                    "log_weights",
                    torch.log(torch.tensor(weights)),
                )

        @property
        def weights(self) -> torch.Tensor:
            return torch.exp(self.log_weights)

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """Compute weighted sum of losses."""
            total_loss = 0.0
            weights = self.weights

            for i, loss_fn in enumerate(self.losses):
                loss = loss_fn(*args, **kwargs)
                total_loss = total_loss + weights[i] * loss

            return total_loss


def create_ranking_loss(
    loss_type: str = "margin",
    **kwargs,
) -> "nn.Module":
    """Factory function to create ranking losses.

    Args:
        loss_type: Type of loss (margin, contrastive, triplet, listnet, lambdarank).
        **kwargs: Additional arguments for specific loss types.

    Returns:
        Initialized loss module.
    """
    if not HAS_TORCH:
        raise ImportError("torch required for ranking losses")

    if loss_type == "margin":
        return MarginRankingLoss(
            margin=kwargs.get("margin", 0.5),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss_type == "contrastive":
        return ContrastiveLoss(
            margin=kwargs.get("margin", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss_type == "triplet":
        return TripletMarginLoss(
            margin=kwargs.get("margin", 1.0),
            p=kwargs.get("p", 2),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss_type == "listnet":
        return ListNetLoss(
            temperature=kwargs.get("temperature", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss_type == "lambdarank":
        return LambdaRankLoss(
            sigma=kwargs.get("sigma", 1.0),
            reduction=kwargs.get("reduction", "mean"),
            k=kwargs.get("k", None),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Ranking metrics for evaluation
def compute_ndcg(
    scores: "torch.Tensor",
    relevance: "torch.Tensor",
    k: Optional[int] = None,
) -> "torch.Tensor":
    """Compute Normalized Discounted Cumulative Gain.

    Args:
        scores: Predicted scores [batch, num_items].
        relevance: Ground truth relevance [batch, num_items].
        k: Cutoff (None = full list).

    Returns:
        NDCG scores [batch].
    """
    if not HAS_TORCH:
        raise ImportError("torch required")

    batch_size, num_items = scores.shape
    device = scores.device

    if k is None:
        k = num_items

    # Sort by predicted scores
    _, pred_order = torch.sort(scores, dim=1, descending=True)

    # Get relevance in predicted order
    sorted_relevance = torch.gather(relevance, 1, pred_order)[:, :k]

    # DCG
    ranks = torch.arange(1, k + 1, device=device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks + 1)
    gains = 2.0 ** sorted_relevance - 1
    dcg = (gains * discounts).sum(dim=1)

    # Ideal DCG
    ideal_relevance, _ = torch.sort(relevance, dim=1, descending=True)
    ideal_relevance = ideal_relevance[:, :k]
    ideal_gains = 2.0 ** ideal_relevance - 1
    idcg = (ideal_gains * discounts).sum(dim=1)

    # NDCG
    ndcg = dcg / (idcg + 1e-10)

    return ndcg


def compute_map(
    scores: "torch.Tensor",
    relevance: "torch.Tensor",
    threshold: float = 0.5,
) -> "torch.Tensor":
    """Compute Mean Average Precision.

    Args:
        scores: Predicted scores [batch, num_items].
        relevance: Ground truth relevance [batch, num_items].
        threshold: Threshold for binary relevance.

    Returns:
        MAP scores [batch].
    """
    if not HAS_TORCH:
        raise ImportError("torch required")

    batch_size, num_items = scores.shape

    # Sort by predicted scores
    _, pred_order = torch.sort(scores, dim=1, descending=True)

    # Get binary relevance in predicted order
    binary_relevance = (relevance > threshold).float()
    sorted_relevance = torch.gather(binary_relevance, 1, pred_order)

    # Compute precision at each position
    cumsum = torch.cumsum(sorted_relevance, dim=1)
    ranks = torch.arange(1, num_items + 1, device=scores.device, dtype=torch.float)
    precisions = cumsum / ranks

    # Average precision (only at relevant positions)
    ap = (precisions * sorted_relevance).sum(dim=1) / (sorted_relevance.sum(dim=1) + 1e-10)

    return ap
