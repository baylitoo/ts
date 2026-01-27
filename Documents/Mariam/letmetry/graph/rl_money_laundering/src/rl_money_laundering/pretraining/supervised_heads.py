"""
Supervised classification heads for AML judge fine-tuning.

After TAPT, the model has good domain-specific embeddings, but needs
decision behavior for AML tasks. These supervised heads transform
the pretrained encoder into a task-specific classifier/scorer.

Supported tasks:
1. Typology Classification (multi-label)
2. Suspiciousness Scoring (binary/regression)
3. Evidence Sentence Selection (token/sentence classification)
4. Pairwise Ranking (comparative judgment)

Usage:
    from rl_money_laundering.pretraining.supervised_heads import (
        TypologyClassificationHead,
        SuspiciousnessScorer,
        EvidenceSelector,
        PairwiseRanker,
    )

    # Load pretrained encoder
    encoder = AutoModel.from_pretrained("./models/tapt")

    # Add typology classification head
    model = TypologyClassificationHead(
        encoder=encoder,
        num_labels=12,
        freeze_encoder=True,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch not installed. Install with: pip install torch")

try:
    from transformers import (
        AutoModel,
        AutoConfig,
        PreTrainedModel,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class PoolingStrategy(Enum):
    """Strategies for pooling encoder outputs."""
    CLS = "cls"  # Use [CLS] token representation
    MEAN = "mean"  # Mean of all token representations
    MAX = "max"  # Max pooling over tokens
    ATTENTION = "attention"  # Attention-weighted pooling


# AML Typology Labels (standard classification targets)
AML_TYPOLOGIES = [
    "layering",
    "structuring",
    "shell_company",
    "trade_based",
    "crypto_laundering",
    "pep_corruption",
    "tax_evasion",
    "sanctions_evasion",
    "human_trafficking",
    "drug_trafficking",
    "fraud",
    "bribery",
]


@dataclass
class HeadConfig:
    """Configuration for supervised heads."""

    # Model settings
    hidden_size: int = 768  # Encoder hidden size
    num_labels: int = 2  # Number of output labels
    dropout_prob: float = 0.1

    # Pooling
    pooling: PoolingStrategy = PoolingStrategy.CLS

    # Encoder freezing
    freeze_encoder: bool = True
    freeze_encoder_layers: Optional[int] = None  # Freeze first N layers

    # Multi-task
    multi_task: bool = False
    task_weights: Optional[Dict[str, float]] = None

    # Calibration
    temperature: float = 1.0  # For temperature scaling
    use_label_smoothing: bool = False
    label_smoothing_alpha: float = 0.1


if HAS_TORCH and HAS_TRANSFORMERS:

    class PoolingLayer(nn.Module):
        """Flexible pooling layer for encoder outputs."""

        def __init__(
            self,
            hidden_size: int,
            pooling: PoolingStrategy = PoolingStrategy.CLS,
        ) -> None:
            super().__init__()
            self.pooling = pooling
            self.hidden_size = hidden_size

            if pooling == PoolingStrategy.ATTENTION:
                self.attention = nn.Linear(hidden_size, 1)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Pool encoder outputs.

            Args:
                hidden_states: Encoder outputs [batch, seq_len, hidden_size].
                attention_mask: Attention mask [batch, seq_len].

            Returns:
                Pooled representation [batch, hidden_size].
            """
            if self.pooling == PoolingStrategy.CLS:
                return hidden_states[:, 0]

            elif self.pooling == PoolingStrategy.MEAN:
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                    return sum_embeddings / sum_mask
                else:
                    return hidden_states.mean(dim=1)

            elif self.pooling == PoolingStrategy.MAX:
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    hidden_states = hidden_states.masked_fill(mask_expanded == 0, -1e9)
                return hidden_states.max(dim=1)[0]

            elif self.pooling == PoolingStrategy.ATTENTION:
                scores = self.attention(hidden_states).squeeze(-1)
                if attention_mask is not None:
                    scores = scores.masked_fill(attention_mask == 0, -1e9)
                weights = F.softmax(scores, dim=1)
                return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    class TypologyClassificationHead(nn.Module):
        """Multi-label classification head for AML typologies.

        Predicts which AML typologies apply to a given text.
        Output is a multi-hot vector of applicable typologies.

        Architecture:
            [Encoder] -> [Pooling] -> [Dropout] -> [Linear] -> [Sigmoid]
        """

        def __init__(
            self,
            encoder: PreTrainedModel,
            num_labels: int = len(AML_TYPOLOGIES),
            config: Optional[HeadConfig] = None,
            label_names: Optional[List[str]] = None,
        ) -> None:
            """Initialize typology classification head.

            Args:
                encoder: Pretrained encoder model.
                num_labels: Number of typology labels.
                config: Head configuration.
                label_names: Names of labels for interpretability.
            """
            super().__init__()

            self.encoder = encoder
            self.config = config or HeadConfig(num_labels=num_labels)
            self.num_labels = num_labels
            self.label_names = label_names or AML_TYPOLOGIES[:num_labels]

            # Get hidden size from encoder
            if hasattr(encoder.config, "hidden_size"):
                hidden_size = encoder.config.hidden_size
            else:
                hidden_size = self.config.hidden_size

            # Build classification head
            self.pooler = PoolingLayer(hidden_size, self.config.pooling)
            self.dropout = nn.Dropout(self.config.dropout_prob)
            self.classifier = nn.Linear(hidden_size, num_labels)

            # Freeze encoder if specified
            if self.config.freeze_encoder:
                self._freeze_encoder()

        def _freeze_encoder(self) -> None:
            """Freeze encoder parameters."""
            for param in self.encoder.parameters():
                param.requires_grad = False

            if self.config.freeze_encoder_layers is not None:
                # Unfreeze last N layers
                if hasattr(self.encoder, "encoder"):
                    layers = self.encoder.encoder.layer
                elif hasattr(self.encoder, "transformer"):
                    layers = self.encoder.transformer.layer
                else:
                    return

                num_layers = len(layers)
                start_unfreeze = max(0, num_layers - self.config.freeze_encoder_layers)

                for i in range(start_unfreeze, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_hidden_states: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                input_ids: Input token IDs [batch, seq_len].
                attention_mask: Attention mask [batch, seq_len].
                labels: Multi-hot labels [batch, num_labels].
                return_hidden_states: Whether to return encoder hidden states.

            Returns:
                Dictionary with logits, loss (if labels provided), and probabilities.
            """
            # Encode
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states,
            )
            hidden_states = encoder_outputs.last_hidden_state

            # Pool
            pooled = self.pooler(hidden_states, attention_mask)

            # Classify
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)

            # Apply temperature scaling for calibration
            logits = logits / self.config.temperature

            # Compute loss if labels provided
            loss = None
            if labels is not None:
                if self.config.use_label_smoothing:
                    # Label smoothing for multi-label
                    alpha = self.config.label_smoothing_alpha
                    smoothed_labels = labels * (1 - alpha) + alpha / self.num_labels
                    loss = F.binary_cross_entropy_with_logits(logits, smoothed_labels)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            result = {
                "logits": logits,
                "probabilities": torch.sigmoid(logits),
            }

            if loss is not None:
                result["loss"] = loss

            if return_hidden_states:
                result["hidden_states"] = encoder_outputs.hidden_states

            return result

        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            threshold: float = 0.5,
        ) -> Dict[str, Any]:
            """Predict typologies for input.

            Args:
                input_ids: Input token IDs.
                attention_mask: Attention mask.
                threshold: Probability threshold for positive prediction.

            Returns:
                Dictionary with predictions and probabilities.
            """
            self.eval()
            with torch.no_grad():
                outputs = self.forward(input_ids, attention_mask)

            probs = outputs["probabilities"]
            predictions = (probs > threshold).int()

            # Convert to label names
            batch_labels = []
            for i in range(predictions.size(0)):
                sample_labels = [
                    self.label_names[j]
                    for j in range(self.num_labels)
                    if predictions[i, j] == 1
                ]
                batch_labels.append(sample_labels)

            return {
                "predictions": predictions,
                "probabilities": probs,
                "labels": batch_labels,
            }

    class SuspiciousnessScorer(nn.Module):
        """Suspiciousness scoring head.

        Predicts a suspiciousness score in [0, 1] for a given text.
        Can be trained as binary classification or regression.

        Architecture:
            [Encoder] -> [Pooling] -> [Dropout] -> [Linear] -> [Sigmoid]
        """

        def __init__(
            self,
            encoder: PreTrainedModel,
            config: Optional[HeadConfig] = None,
            regression: bool = False,
        ) -> None:
            """Initialize suspiciousness scorer.

            Args:
                encoder: Pretrained encoder model.
                config: Head configuration.
                regression: If True, train as regression; else binary classification.
            """
            super().__init__()

            self.encoder = encoder
            self.config = config or HeadConfig(num_labels=1)
            self.regression = regression

            # Get hidden size
            if hasattr(encoder.config, "hidden_size"):
                hidden_size = encoder.config.hidden_size
            else:
                hidden_size = self.config.hidden_size

            # Build scoring head
            self.pooler = PoolingLayer(hidden_size, self.config.pooling)
            self.dropout = nn.Dropout(self.config.dropout_prob)

            # Deeper head for better calibration
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_prob),
                nn.Linear(hidden_size // 2, 1),
            )

            # Freeze encoder if specified
            if self.config.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                input_ids: Input token IDs [batch, seq_len].
                attention_mask: Attention mask [batch, seq_len].
                labels: Suspiciousness labels [batch] (0-1 for regression, 0/1 for classification).

            Returns:
                Dictionary with scores, loss, and logits.
            """
            # Encode
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = encoder_outputs.last_hidden_state

            # Pool and score
            pooled = self.pooler(hidden_states, attention_mask)
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled).squeeze(-1)

            # Apply temperature scaling
            logits = logits / self.config.temperature

            # Compute scores
            if self.regression:
                scores = torch.sigmoid(logits)
            else:
                scores = torch.sigmoid(logits)

            # Compute loss
            loss = None
            if labels is not None:
                if self.regression:
                    # MSE loss for regression
                    loss = F.mse_loss(scores, labels.float())
                else:
                    # BCE loss for classification
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            return {
                "scores": scores,
                "logits": logits,
                "loss": loss,
            }

        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Get suspiciousness scores.

            Args:
                input_ids: Input token IDs.
                attention_mask: Attention mask.

            Returns:
                Suspiciousness scores in [0, 1].
            """
            self.eval()
            with torch.no_grad():
                outputs = self.forward(input_ids, attention_mask)
            return outputs["scores"]

    class EvidenceSelector(nn.Module):
        """Evidence sentence selection head.

        Identifies which sentences/tokens are "red flags" in a document.
        Supports both token-level (BIO tagging) and sentence-level classification.

        Architecture (token-level):
            [Encoder] -> [Dropout] -> [Linear] -> [CRF/Softmax]

        Architecture (sentence-level):
            [Encoder] -> [Sentence Pooling] -> [Linear] -> [Sigmoid]
        """

        def __init__(
            self,
            encoder: PreTrainedModel,
            config: Optional[HeadConfig] = None,
            token_level: bool = True,
            num_tags: int = 3,  # B-EVIDENCE, I-EVIDENCE, O
        ) -> None:
            """Initialize evidence selector.

            Args:
                encoder: Pretrained encoder model.
                config: Head configuration.
                token_level: If True, do token classification; else sentence classification.
                num_tags: Number of BIO tags for token classification.
            """
            super().__init__()

            self.encoder = encoder
            self.config = config or HeadConfig()
            self.token_level = token_level
            self.num_tags = num_tags

            # Get hidden size
            if hasattr(encoder.config, "hidden_size"):
                hidden_size = encoder.config.hidden_size
            else:
                hidden_size = self.config.hidden_size

            self.dropout = nn.Dropout(self.config.dropout_prob)

            if token_level:
                # Token classification head
                self.classifier = nn.Linear(hidden_size, num_tags)
            else:
                # Sentence classification head
                self.classifier = nn.Linear(hidden_size, 1)

            # Freeze encoder if specified
            if self.config.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            sentence_boundaries: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                input_ids: Input token IDs [batch, seq_len].
                attention_mask: Attention mask [batch, seq_len].
                labels: Token labels [batch, seq_len] or sentence labels [batch, num_sentences].
                sentence_boundaries: Sentence boundary indices for sentence-level [batch, num_sentences, 2].

            Returns:
                Dictionary with logits, predictions, and loss.
            """
            # Encode
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = encoder_outputs.last_hidden_state

            hidden_states = self.dropout(hidden_states)

            if self.token_level:
                # Token classification
                logits = self.classifier(hidden_states)

                loss = None
                if labels is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, self.num_tags),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                return {
                    "logits": logits,
                    "predictions": logits.argmax(dim=-1),
                    "loss": loss,
                }
            else:
                # Sentence classification
                if sentence_boundaries is None:
                    raise ValueError("sentence_boundaries required for sentence-level classification")

                batch_size = hidden_states.size(0)
                num_sentences = sentence_boundaries.size(1)

                # Pool each sentence
                sentence_reps = []
                for b in range(batch_size):
                    batch_reps = []
                    for s in range(num_sentences):
                        start, end = sentence_boundaries[b, s]
                        if start >= 0 and end > start:
                            sent_rep = hidden_states[b, start:end].mean(dim=0)
                        else:
                            sent_rep = torch.zeros(hidden_states.size(-1), device=hidden_states.device)
                        batch_reps.append(sent_rep)
                    sentence_reps.append(torch.stack(batch_reps))

                sentence_reps = torch.stack(sentence_reps)  # [batch, num_sentences, hidden]
                logits = self.classifier(sentence_reps).squeeze(-1)

                loss = None
                if labels is not None:
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())

                return {
                    "logits": logits,
                    "predictions": (torch.sigmoid(logits) > 0.5).int(),
                    "loss": loss,
                }

        def predict_evidence_spans(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> List[List[Tuple[int, int]]]:
            """Predict evidence spans in token-level mode.

            Args:
                input_ids: Input token IDs.
                attention_mask: Attention mask.

            Returns:
                List of (start, end) spans for each sample.
            """
            if not self.token_level:
                raise ValueError("predict_evidence_spans only available in token-level mode")

            self.eval()
            with torch.no_grad():
                outputs = self.forward(input_ids, attention_mask)

            predictions = outputs["predictions"]
            batch_spans = []

            # B=0, I=1, O=2 (assumed)
            for batch_idx in range(predictions.size(0)):
                spans = []
                in_span = False
                start = 0

                for i, tag in enumerate(predictions[batch_idx].tolist()):
                    if tag == 0:  # B-EVIDENCE
                        if in_span:
                            spans.append((start, i))
                        in_span = True
                        start = i
                    elif tag == 1:  # I-EVIDENCE
                        if not in_span:
                            in_span = True
                            start = i
                    else:  # O
                        if in_span:
                            spans.append((start, i))
                            in_span = False

                if in_span:
                    spans.append((start, predictions.size(1)))

                batch_spans.append(spans)

            return batch_spans

    class PairwiseRanker(nn.Module):
        """Pairwise ranking head for comparative judgment.

        Given two cases (A, B), predicts which is more suspicious.
        Trained with margin ranking loss.

        This enables prioritization of investigation queues.

        Architecture:
            [Encoder(A)] -> [Score(A)]
            [Encoder(B)] -> [Score(B)]
            Ranking Loss: max(0, margin - (Score(A) - Score(B)))
        """

        def __init__(
            self,
            encoder: PreTrainedModel,
            config: Optional[HeadConfig] = None,
            margin: float = 0.5,
        ) -> None:
            """Initialize pairwise ranker.

            Args:
                encoder: Pretrained encoder model.
                config: Head configuration.
                margin: Margin for ranking loss.
            """
            super().__init__()

            self.encoder = encoder
            self.config = config or HeadConfig()
            self.margin = margin

            # Get hidden size
            if hasattr(encoder.config, "hidden_size"):
                hidden_size = encoder.config.hidden_size
            else:
                hidden_size = self.config.hidden_size

            # Scoring head (shared for both inputs)
            self.pooler = PoolingLayer(hidden_size, self.config.pooling)
            self.dropout = nn.Dropout(self.config.dropout_prob)
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_prob),
                nn.Linear(hidden_size // 2, 1),
            )

            # Freeze encoder if specified
            if self.config.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        def _encode_and_score(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Encode input and compute score."""
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = encoder_outputs.last_hidden_state

            pooled = self.pooler(hidden_states, attention_mask)
            pooled = self.dropout(pooled)
            score = self.scorer(pooled).squeeze(-1)

            return score

        def forward(
            self,
            input_ids_a: torch.Tensor,
            input_ids_b: torch.Tensor,
            attention_mask_a: Optional[torch.Tensor] = None,
            attention_mask_b: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                input_ids_a: Token IDs for case A [batch, seq_len].
                input_ids_b: Token IDs for case B [batch, seq_len].
                attention_mask_a: Attention mask for A.
                attention_mask_b: Attention mask for B.
                labels: Ranking labels [batch]:
                    +1 if A is more suspicious than B
                    -1 if B is more suspicious than A

            Returns:
                Dictionary with scores and loss.
            """
            # Score both inputs
            score_a = self._encode_and_score(input_ids_a, attention_mask_a)
            score_b = self._encode_and_score(input_ids_b, attention_mask_b)

            # Compute ranking loss
            loss = None
            if labels is not None:
                # Margin ranking loss
                # When label=1: A should be higher than B
                # When label=-1: B should be higher than A
                loss = F.margin_ranking_loss(
                    score_a,
                    score_b,
                    labels.float(),
                    margin=self.margin,
                )

            # Predictions: 1 if A > B, -1 otherwise
            predictions = torch.where(
                score_a > score_b,
                torch.ones_like(score_a),
                -torch.ones_like(score_a),
            )

            return {
                "score_a": score_a,
                "score_b": score_b,
                "predictions": predictions,
                "loss": loss,
            }

        def rank_batch(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Score a batch of items for ranking.

            Args:
                input_ids: Token IDs [batch, seq_len].
                attention_mask: Attention mask.

            Returns:
                Scores [batch] for sorting.
            """
            self.eval()
            with torch.no_grad():
                scores = self._encode_and_score(input_ids, attention_mask)
            return scores


def create_supervised_head(
    head_type: str,
    encoder: "PreTrainedModel",
    num_labels: int = 2,
    **kwargs,
) -> "nn.Module":
    """Factory function to create supervised heads.

    Args:
        head_type: Type of head (typology, suspiciousness, evidence, ranking).
        encoder: Pretrained encoder model.
        num_labels: Number of output labels.
        **kwargs: Additional arguments for specific head types.

    Returns:
        Initialized supervised head.
    """
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        raise ImportError("torch and transformers required for supervised heads")

    config = HeadConfig(num_labels=num_labels, **kwargs)

    if head_type == "typology":
        return TypologyClassificationHead(encoder, num_labels, config)
    elif head_type == "suspiciousness":
        return SuspiciousnessScorer(encoder, config, regression=kwargs.get("regression", False))
    elif head_type == "evidence":
        return EvidenceSelector(encoder, config, token_level=kwargs.get("token_level", True))
    elif head_type == "ranking":
        return PairwiseRanker(encoder, config, margin=kwargs.get("margin", 0.5))
    else:
        raise ValueError(f"Unknown head type: {head_type}")
