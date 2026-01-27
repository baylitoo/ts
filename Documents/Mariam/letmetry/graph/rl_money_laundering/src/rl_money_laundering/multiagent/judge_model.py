"""
Learned Judge Model for Multi-Agent AML Detection

This module implements a ModernBERT-based reward model that evaluates
detector episodes and produces scalar rewards with interpretable fraud
type predictions. Designed for banking AML context with audit requirements.

Key Components:
- JudgeModel: Fine-tuned ModernBERT with reward/fraud-type heads
- JudgeConfig: Configuration for model architecture and inference
- EpisodeSerializer: Canonical episode-to-text conversion (order-invariant)
- GraphTextFusion: Optional hybrid graph+text encoding

Safety Features:
- Conservative reward clipping
- Uncertainty penalties via ensemble disagreement
- No ground-truth leakage at inference
- Deterministic inference for audit reproducibility

References:
- Christiano et al. (2017): Deep RL from Human Preferences
- Ouyang et al. (2022): InstructGPT / RLHF reward modeling
- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class JudgeConfig:
    """Configuration for the learned judge model.

    Args:
        model_name: HuggingFace model identifier (ModernBERT recommended)
        max_length: Maximum sequence length for tokenization
        hidden_dim: Hidden dimension for reward/classification heads
        num_fraud_types: Number of fraud type categories
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_rank: LoRA rank (if use_lora=True)
        dropout: Dropout rate for heads
        reward_clip: Clip reward output to [-clip, +clip]
        uncertainty_penalty_weight: Weight for ensemble disagreement penalty
        device: Computation device
        deterministic: Enable deterministic inference for audit
        ensemble_size: Number of models for ensemble (1 = single model)
        freeze_encoder: Freeze encoder weights (use for inference)
        output_fraud_types: Whether to output fraud type predictions
    """

    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    hidden_dim: int = 256
    num_fraud_types: int = 5  # layering, structuring, smurfing, shell_company, other
    use_lora: bool = True
    lora_rank: int = 16
    dropout: float = 0.1
    reward_clip: float = 1.0
    uncertainty_penalty_weight: float = 0.1
    device: str = "auto"
    deterministic: bool = True
    ensemble_size: int = 1
    freeze_encoder: bool = False
    output_fraud_types: bool = True

    # Fraud type labels (for interpretability)
    fraud_type_labels: List[str] = field(
        default_factory=lambda: [
            "layering",
            "structuring",
            "smurfing",
            "shell_company",
            "legitimate",
        ]
    )

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Episode Serialization (Critical for avoiding shortcuts)
# =============================================================================


class EpisodeSerializer:
    """Serialize episodes to canonical text format for judge input.

    CRITICAL: This serialization must be:
    1. Order-invariant (canonical sorting)
    2. Node-ID agnostic (no raw IDs, only features)
    3. Deterministic (same episode -> same text)
    4. Ground-truth free (no fraud labels in input)

    Failure modes prevented:
    - Node ID leakage: IDs are hashed, not exposed
    - Order sensitivity: Nodes/edges sorted by canonical key
    - Numeric precision: Features quantized to fixed precision
    - Tokenization issues: Consistent formatting
    """

    def __init__(
        self,
        quantize_precision: int = 2,
        max_nodes: int = 50,
        max_edges: int = 100,
        include_temporal: bool = True,
        anonymize_ids: bool = True,
    ):
        """
        Args:
            quantize_precision: Decimal places for numeric features
            max_nodes: Maximum nodes to include in serialization
            max_edges: Maximum edges to include in serialization
            include_temporal: Include temporal features (velocity, periodicity)
            anonymize_ids: Hash node IDs instead of using raw values
        """
        self.quantize_precision = quantize_precision
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.include_temporal = include_temporal
        self.anonymize_ids = anonymize_ids

    def serialize(self, episode: Dict[str, Any]) -> str:
        """Convert episode to canonical text representation.

        Args:
            episode: Episode dictionary with keys:
                - visited_nodes: List of node IDs
                - visited_edges: List of (src, dst) tuples
                - actions: List of actions taken
                - node_features: Dict[node_id, feature_dict]
                - edge_features: Dict[edge, feature_dict]
                - metadata: Episode-level metadata

        Returns:
            Canonical text representation
        """
        parts: List[str] = []

        # Episode summary
        metadata = episode.get("metadata", {})
        parts.append(self._serialize_metadata(metadata))

        # Node features (sorted canonically)
        node_features = episode.get("node_features", {})
        visited_nodes = episode.get("visited_nodes", [])
        parts.append(self._serialize_nodes(visited_nodes, node_features))

        # Edge features (sorted canonically)
        edge_features = episode.get("edge_features", {})
        visited_edges = episode.get("visited_edges", [])
        parts.append(self._serialize_edges(visited_edges, edge_features))

        # Action sequence
        actions = episode.get("actions", [])
        parts.append(self._serialize_actions(actions))

        # Temporal patterns (if enabled)
        if self.include_temporal:
            parts.append(self._serialize_temporal_patterns(episode))

        return " ".join(filter(None, parts))

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize episode metadata."""
        items = [
            f"episode_length:{metadata.get('episode_length', 0)}",
            f"unique_nodes:{metadata.get('unique_nodes', 0)}",
            f"flags_used:{metadata.get('flags_used', 0)}",
        ]
        return f"[META] {' '.join(items)}"

    def _serialize_nodes(
        self, node_ids: List[str], features: Dict[str, Dict[str, Any]]
    ) -> str:
        """Serialize node features with canonical ordering."""
        if not node_ids:
            return "[NODES] empty"

        # Create canonical node representations
        node_reprs: List[Tuple[str, str]] = []
        for node_id in node_ids[: self.max_nodes]:
            node_feat = features.get(str(node_id), {})
            repr_str = self._node_to_string(node_id, node_feat)
            # Sort key: feature hash for canonical ordering
            sort_key = self._feature_hash(node_feat)
            node_reprs.append((sort_key, repr_str))

        # Sort by feature hash (order-invariant)
        node_reprs.sort(key=lambda x: x[0])
        node_strs = [r[1] for r in node_reprs]

        return f"[NODES] {' | '.join(node_strs)}"

    def _serialize_edges(
        self, edges: List[Tuple[str, str]], features: Dict[Any, Dict[str, Any]]
    ) -> str:
        """Serialize edge features with canonical ordering."""
        if not edges:
            return "[EDGES] empty"

        edge_reprs: List[Tuple[str, str]] = []
        for edge in edges[: self.max_edges]:
            edge_key = (str(edge[0]), str(edge[1]))
            edge_feat = features.get(edge_key, features.get(edge, {}))
            repr_str = self._edge_to_string(edge, edge_feat)
            sort_key = self._feature_hash(edge_feat)
            edge_reprs.append((sort_key, repr_str))

        edge_reprs.sort(key=lambda x: x[0])
        edge_strs = [r[1] for r in edge_reprs]

        return f"[EDGES] {' | '.join(edge_strs)}"

    def _serialize_actions(self, actions: List[int]) -> str:
        """Serialize action sequence."""
        if not actions:
            return "[ACTIONS] none"

        # Compress action sequence
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        action_parts = [f"a{a}:{c}" for a, c in sorted(action_counts.items())]
        return f"[ACTIONS] {' '.join(action_parts)} total:{len(actions)}"

    def _serialize_temporal_patterns(self, episode: Dict[str, Any]) -> str:
        """Extract and serialize temporal patterns."""
        patterns: List[str] = []

        edge_features = episode.get("edge_features", {})
        if not edge_features:
            return ""

        # Compute velocity (transactions per time unit)
        timestamps = [
            f.get("timestamp", f.get("time", f.get("step", 0)))
            for f in edge_features.values()
        ]
        if len(timestamps) > 1:
            valid_ts = [t for t in timestamps if t is not None]
            if len(valid_ts) > 1:
                time_span = max(valid_ts) - min(valid_ts) + 1
                velocity = len(valid_ts) / time_span
                patterns.append(f"velocity:{self._quantize(velocity)}")

        # Compute amount statistics
        amounts = [
            f.get("amount", 0) for f in edge_features.values() if "amount" in f
        ]
        if amounts:
            patterns.append(f"amt_mean:{self._quantize(np.mean(amounts))}")
            patterns.append(f"amt_std:{self._quantize(np.std(amounts))}")
            patterns.append(f"amt_max:{self._quantize(max(amounts))}")

        if patterns:
            return f"[TEMPORAL] {' '.join(patterns)}"
        return ""

    def _node_to_string(self, node_id: str, features: Dict[str, Any]) -> str:
        """Convert node to string representation."""
        parts: List[str] = []

        # Anonymized ID (hash prefix)
        if self.anonymize_ids:
            anon_id = self._anonymize_id(str(node_id))
            parts.append(f"n:{anon_id}")
        else:
            parts.append(f"n:{node_id}")

        # Key features (no ground truth!)
        if "account_type" in features:
            parts.append(f"type:{features['account_type']}")
        if "risk_score" in features:
            parts.append(f"risk:{self._quantize(features['risk_score'])}")
        if "balance" in features:
            # Log-scale for large values
            bal = features["balance"]
            log_bal = np.sign(bal) * np.log1p(abs(bal))
            parts.append(f"bal:{self._quantize(log_bal)}")
        if "degree" in features:
            parts.append(f"deg:{features['degree']}")

        return " ".join(parts)

    def _edge_to_string(
        self, edge: Tuple[str, str], features: Dict[str, Any]
    ) -> str:
        """Convert edge to string representation."""
        parts: List[str] = []

        # Anonymized edge
        if self.anonymize_ids:
            src_anon = self._anonymize_id(str(edge[0]))
            dst_anon = self._anonymize_id(str(edge[1]))
            parts.append(f"e:{src_anon}->{dst_anon}")
        else:
            parts.append(f"e:{edge[0]}->{edge[1]}")

        # Key features (no is_fraud!)
        if "amount" in features:
            amt = features["amount"]
            log_amt = np.sign(amt) * np.log1p(abs(amt))
            parts.append(f"amt:{self._quantize(log_amt)}")
        if "timestamp" in features or "time" in features or "step" in features:
            ts = features.get("timestamp", features.get("time", features.get("step")))
            parts.append(f"t:{ts}")
        if "transaction_type" in features:
            parts.append(f"txtype:{features['transaction_type']}")

        return " ".join(parts)

    def _quantize(self, value: float) -> str:
        """Quantize numeric value to fixed precision."""
        if not np.isfinite(value):
            return "0.00"
        return f"{value:.{self.quantize_precision}f}"

    def _anonymize_id(self, node_id: str) -> str:
        """Hash node ID to prevent leakage."""
        hash_bytes = hashlib.sha256(node_id.encode()).hexdigest()[:8]
        return hash_bytes

    def _feature_hash(self, features: Dict[str, Any]) -> str:
        """Create deterministic hash of features for canonical ordering."""
        # Exclude any potential ground truth fields
        safe_features = {
            k: v
            for k, v in features.items()
            if k not in ("is_fraud", "label", "ground_truth", "fraud", "is_money_laundering")
        }
        feature_str = json.dumps(safe_features, sort_keys=True, default=str)
        return hashlib.sha256(feature_str.encode()).hexdigest()


# =============================================================================
# Graph-Text Fusion (Optional hybrid encoding)
# =============================================================================


class GraphTextFusion(nn.Module):
    """Fuse graph embeddings with text embeddings for hybrid judge.

    Architecture:
        graph_embedding (from GNN) + text_embedding (from encoder)
        -> cross-attention -> fused representation

    This allows the judge to leverage both:
    - Structural patterns (cycles, fan-out) from graph encoder
    - Semantic patterns (velocity descriptions) from text encoder
    """

    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        fused_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            graph_dim: Dimension of graph embeddings
            text_dim: Dimension of text embeddings
            fused_dim: Output dimension after fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.fused_dim = fused_dim

        # Project to common dimension
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.text_proj = nn.Linear(text_dim, fused_dim)

        # Cross-attention: text attends to graph
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        graph_embedding: Tensor,
        text_embedding: Tensor,
    ) -> Tensor:
        """
        Fuse graph and text embeddings.

        Args:
            graph_embedding: [batch_size, graph_dim] or [batch_size, num_nodes, graph_dim]
            text_embedding: [batch_size, text_dim]

        Returns:
            fused: [batch_size, fused_dim]
        """
        # Handle different graph embedding shapes
        if graph_embedding.dim() == 2:
            graph_embedding = graph_embedding.unsqueeze(1)  # [B, 1, D]

        # Project to common space
        graph_proj = self.graph_proj(graph_embedding)  # [B, N, fused_dim]
        text_proj = self.text_proj(text_embedding)  # [B, fused_dim]
        text_proj = text_proj.unsqueeze(1)  # [B, 1, fused_dim]

        # Cross-attention: text queries graph
        attended, _ = self.cross_attention(
            query=text_proj,
            key=graph_proj,
            value=graph_proj,
        )  # [B, 1, fused_dim]

        attended = attended.squeeze(1)  # [B, fused_dim]
        text_proj = text_proj.squeeze(1)  # [B, fused_dim]

        # Concatenate and fuse
        combined = torch.cat([attended, text_proj], dim=-1)  # [B, fused_dim*2]
        fused = self.fusion(combined)  # [B, fused_dim]

        return fused


# =============================================================================
# Judge Model
# =============================================================================


class JudgeModel(nn.Module):
    """Learned reward model for AML detection episodes.

    Architecture (MVP):
        ModernBERT encoder -> [CLS] embedding
        -> reward head (scalar in [-1, 1])
        -> fraud_type head (optional, multiclass)

    Safety Features:
        - Reward clipping to prevent extreme values
        - Uncertainty estimation via MC dropout or ensemble
        - Conservative reward computation (penalize uncertainty)

    Training Modes:
        1. Regression: MSE/Huber loss on scalar rewards
        2. Preference: Bradley-Terry ranking loss on episode pairs
        3. Multi-task: Joint reward + fraud-type classification
    """

    def __init__(self, config: JudgeConfig):
        """Initialize judge model.

        Args:
            config: JudgeConfig with model settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Episode serializer
        self.serializer = EpisodeSerializer()

        # Initialize encoder (lazy loading to handle missing transformers)
        self._encoder: Optional[nn.Module] = None
        self._tokenizer: Optional[Any] = None
        self._encoder_dim: int = 768  # ModernBERT-base hidden size

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(self._encoder_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Fraud type head (optional)
        self.fraud_type_head: Optional[nn.Module] = None
        if config.output_fraud_types:
            self.fraud_type_head = nn.Sequential(
                nn.Linear(self._encoder_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_fraud_types),
            )

        # Uncertainty head (for conservative rewards)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self._encoder_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus(),  # Positive uncertainty
        )

        # Graph-text fusion (optional, for hybrid mode)
        self.graph_fusion: Optional[GraphTextFusion] = None

        # Move to device
        self.to(self.device)

        # Version tracking for audit
        self._version: str = "1.0.0"
        self._training_episodes: int = 0

    @property
    def encoder(self) -> nn.Module:
        """Lazy-load encoder to handle missing transformers gracefully."""
        if self._encoder is None:
            self._encoder = self._load_encoder()
        return self._encoder

    @property
    def tokenizer(self) -> Any:
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    def _load_encoder(self) -> nn.Module:
        """Load the text encoder model."""
        try:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )

            # Apply LoRA if configured
            if self.config.use_lora:
                encoder = self._apply_lora(encoder)

            # Freeze if configured
            if self.config.freeze_encoder:
                for param in encoder.parameters():
                    param.requires_grad = False

            return encoder.to(self.device)

        except ImportError:
            logger.warning(
                "transformers not installed. Using mock encoder for testing."
            )
            return self._create_mock_encoder()

    def _load_tokenizer(self) -> Any:
        """Load the tokenizer."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
        except ImportError:
            logger.warning("transformers not installed. Using mock tokenizer.")
            return self._create_mock_tokenizer()

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank * 2,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=self.config.dropout,
                bias="none",
            )
            return get_peft_model(model, lora_config)
        except ImportError:
            logger.warning("peft not installed. Skipping LoRA.")
            return model

    def _create_mock_encoder(self) -> nn.Module:
        """Create a mock encoder for testing without transformers."""

        class MockEncoder(nn.Module):
            def __init__(self, hidden_size: int = 768):
                super().__init__()
                self.hidden_size = hidden_size
                self.embeddings = nn.Embedding(30000, hidden_size)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        batch_first=True,
                    ),
                    num_layers=2,
                )

            def forward(
                self,
                input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                **kwargs: Any,
            ) -> Any:
                x = self.embeddings(input_ids)
                x = self.encoder(x)

                class Output:
                    def __init__(self, last_hidden_state: Tensor):
                        self.last_hidden_state = last_hidden_state

                return Output(x)

        return MockEncoder(self._encoder_dim).to(self.device)

    def _create_mock_tokenizer(self) -> Any:
        """Create a mock tokenizer for testing."""

        class MockTokenizer:
            def __init__(self, vocab_size: int = 30000, max_length: int = 512):
                self.vocab_size = vocab_size
                self.max_length = max_length

            def __call__(
                self,
                texts: Union[str, List[str]],
                padding: bool = True,
                truncation: bool = True,
                max_length: Optional[int] = None,
                return_tensors: Optional[str] = None,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                if isinstance(texts, str):
                    texts = [texts]

                max_len = max_length or self.max_length

                # Simple hash-based tokenization
                input_ids = []
                for text in texts:
                    tokens = [
                        hash(word) % self.vocab_size
                        for word in text.split()[:max_len]
                    ]
                    tokens = tokens + [0] * (max_len - len(tokens))
                    input_ids.append(tokens[:max_len])

                result = {
                    "input_ids": input_ids,
                    "attention_mask": [
                        [1 if t != 0 else 0 for t in ids] for ids in input_ids
                    ],
                }

                if return_tensors == "pt":
                    result = {
                        k: torch.tensor(v) for k, v in result.items()
                    }

                return result

        return MockTokenizer(max_length=self.config.max_length)

    def enable_graph_fusion(
        self,
        graph_dim: int,
        num_heads: int = 4,
    ) -> None:
        """Enable graph-text fusion for hybrid judge.

        Args:
            graph_dim: Dimension of graph embeddings from GNN
            num_heads: Number of attention heads for fusion
        """
        self.graph_fusion = GraphTextFusion(
            graph_dim=graph_dim,
            text_dim=self._encoder_dim,
            fused_dim=self._encoder_dim,
            num_heads=num_heads,
            dropout=self.config.dropout,
        ).to(self.device)

        logger.info(f"Enabled graph-text fusion (graph_dim={graph_dim})")

    def forward(
        self,
        episodes: Union[List[Dict[str, Any]], Dict[str, Any]],
        graph_embeddings: Optional[Tensor] = None,
        return_uncertainty: bool = True,
        return_fraud_types: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass: episode -> reward + optional fraud types.

        Args:
            episodes: Single episode or list of episodes
            graph_embeddings: Optional graph embeddings for fusion
            return_uncertainty: Whether to return uncertainty estimates
            return_fraud_types: Whether to return fraud type predictions

        Returns:
            Dictionary with:
                - reward: [batch_size, 1] scalar rewards in [-1, 1]
                - uncertainty: [batch_size, 1] uncertainty estimates (if requested)
                - fraud_type_logits: [batch_size, num_types] logits (if requested)
                - fraud_type_probs: [batch_size, num_types] probabilities (if requested)
        """
        # Handle single episode
        if isinstance(episodes, dict):
            episodes = [episodes]

        # Serialize episodes to text
        texts = [self.serializer.serialize(ep) for ep in episodes]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get encoder output
        with torch.set_grad_enabled(self.training):
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get [CLS] embedding (first token)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]  # [B, D]

        # Optional graph-text fusion
        if self.graph_fusion is not None and graph_embeddings is not None:
            cls_embedding = self.graph_fusion(graph_embeddings, cls_embedding)

        # Compute reward
        raw_reward = self.reward_head(cls_embedding)  # [B, 1] in [-1, 1]

        # Clip reward for safety
        reward = torch.clamp(
            raw_reward,
            -self.config.reward_clip,
            self.config.reward_clip,
        )

        outputs: Dict[str, Tensor] = {"reward": reward}

        # Compute uncertainty
        if return_uncertainty:
            uncertainty = self.uncertainty_head(cls_embedding)
            outputs["uncertainty"] = uncertainty

            # Conservative reward: penalize uncertain predictions
            conservative_reward = reward - self.config.uncertainty_penalty_weight * uncertainty
            outputs["conservative_reward"] = conservative_reward

        # Compute fraud types
        if return_fraud_types and self.fraud_type_head is not None:
            fraud_logits = self.fraud_type_head(cls_embedding)
            fraud_probs = F.softmax(fraud_logits, dim=-1)
            outputs["fraud_type_logits"] = fraud_logits
            outputs["fraud_type_probs"] = fraud_probs

        return outputs

    def compute_reward(
        self,
        episode: Dict[str, Any],
        graph_embedding: Optional[Tensor] = None,
        conservative: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward for a single episode (inference API).

        Args:
            episode: Episode dictionary
            graph_embedding: Optional graph embedding
            conservative: Use conservative (uncertainty-penalized) reward

        Returns:
            Tuple of (reward_value, metadata_dict)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                episodes=[episode],
                graph_embeddings=graph_embedding.unsqueeze(0) if graph_embedding is not None else None,
                return_uncertainty=True,
                return_fraud_types=self.config.output_fraud_types,
            )

        # Select reward type
        if conservative and "conservative_reward" in outputs:
            reward = outputs["conservative_reward"].item()
        else:
            reward = outputs["reward"].item()

        # Build metadata
        metadata: Dict[str, Any] = {
            "raw_reward": outputs["reward"].item(),
            "uncertainty": outputs.get("uncertainty", torch.tensor(0.0)).item(),
            "model_version": self._version,
        }

        if "fraud_type_probs" in outputs:
            probs = outputs["fraud_type_probs"][0].cpu().numpy()
            predicted_type_idx = int(np.argmax(probs))
            metadata["fraud_type"] = self.config.fraud_type_labels[predicted_type_idx]
            metadata["fraud_type_confidence"] = float(probs[predicted_type_idx])
            metadata["fraud_type_probs"] = {
                label: float(p)
                for label, p in zip(self.config.fraud_type_labels, probs)
            }

        return reward, metadata

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config.__dict__,
            "reward_head": self.reward_head.state_dict(),
            "uncertainty_head": self.uncertainty_head.state_dict(),
            "version": self._version,
            "training_episodes": self._training_episodes,
        }

        if self.fraud_type_head is not None:
            checkpoint["fraud_type_head"] = self.fraud_type_head.state_dict()

        if self.graph_fusion is not None:
            checkpoint["graph_fusion"] = self.graph_fusion.state_dict()

        # Save encoder separately if not using HuggingFace
        if hasattr(self._encoder, "state_dict"):
            checkpoint["encoder"] = self._encoder.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved judge model to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.reward_head.load_state_dict(checkpoint["reward_head"])
        self.uncertainty_head.load_state_dict(checkpoint["uncertainty_head"])
        self._version = checkpoint.get("version", "1.0.0")
        self._training_episodes = checkpoint.get("training_episodes", 0)

        if "fraud_type_head" in checkpoint and self.fraud_type_head is not None:
            self.fraud_type_head.load_state_dict(checkpoint["fraud_type_head"])

        if "graph_fusion" in checkpoint and self.graph_fusion is not None:
            self.graph_fusion.load_state_dict(checkpoint["graph_fusion"])

        if "encoder" in checkpoint and self._encoder is not None:
            self._encoder.load_state_dict(checkpoint["encoder"])

        logger.info(f"Loaded judge model from {path} (v{self._version})")

    def get_version_info(self) -> Dict[str, Any]:
        """Get version information for audit logging."""
        return {
            "version": self._version,
            "model_name": self.config.model_name,
            "training_episodes": self._training_episodes,
            "config": self.config.__dict__,
        }
