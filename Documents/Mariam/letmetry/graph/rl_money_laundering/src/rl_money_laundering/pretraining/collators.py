"""
Data collators and dataset classes for pretraining.

Supports:
- MLM (Masked Language Modeling) for BERT-style models
- Span MLM (Span Masked Language Modeling) for TAPT with geometric span lengths
- CLM (Causal Language Modeling) for GPT-style models
- Span corruption for T5-style models
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class AMLTextSample:
    """A single text sample for pretraining."""

    text: str
    source: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AMLTextDataset(Dataset):
    """Dataset for AML text corpus.

    Loads text samples from JSONL files and prepares them for tokenization.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer: Any,
        max_length: int = 512,
        text_column: str = "text",
        cache_tokenized: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            file_path: Path to JSONL file.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            text_column: Column name for text.
            cache_tokenized: Cache tokenized samples.
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.cache_tokenized = cache_tokenized

        self.samples: List[AMLTextSample] = []
        self._tokenized_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load samples from file."""
        logger.info(f"Loading samples from {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    text = data.get(self.text_column, "")
                    if not text:
                        continue

                    sample = AMLTextSample(
                        text=text,
                        source=data.get("source"),
                        category=data.get("category"),
                        metadata=data.get("metadata"),
                    )
                    self.samples.append(sample)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized sample."""
        if self.cache_tokenized and idx in self._tokenized_cache:
            return self._tokenized_cache[idx]

        sample = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            sample.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension
        result = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.cache_tokenized:
            self._tokenized_cache[idx] = result

        return result


class StreamingAMLDataset(IterableDataset):
    """Streaming dataset for large corpora.

    Reads samples line-by-line without loading entire file into memory.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Any,
        max_length: int = 512,
        text_column: str = "text",
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            file_paths: Path(s) to JSONL file(s).
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            text_column: Column name for text.
            shuffle_buffer_size: Size of shuffle buffer.
            seed: Random seed.
        """
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __iter__(self):
        """Iterate through samples with shuffling."""
        buffer = []
        rng = random.Random(self.seed)

        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        text = data.get(self.text_column, "")
                        if not text:
                            continue

                        buffer.append(text)

                        if len(buffer) >= self.shuffle_buffer_size:
                            rng.shuffle(buffer)
                            for text in buffer[:self.shuffle_buffer_size // 2]:
                                yield self._tokenize(text)
                            buffer = buffer[self.shuffle_buffer_size // 2:]

                    except json.JSONDecodeError:
                        continue

        # Flush remaining buffer
        rng.shuffle(buffer)
        for text in buffer:
            yield self._tokenize(text)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


@dataclass
class MLMDataCollator:
    """Data collator for Masked Language Modeling.

    Implements dynamic masking where tokens are masked at training time.
    Supports whole-word masking for better semantic understanding.
    """

    tokenizer: Any
    mlm_probability: float = 0.15
    whole_word_masking: bool = True
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a mask token for MLM")

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Collate examples and apply masking.

        Args:
            examples: List of tokenized examples.

        Returns:
            Batch with input_ids, attention_mask, and labels.
        """
        # Stack tensors
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }

        # Create labels (copy of input_ids)
        labels = batch["input_ids"].clone()

        # Apply masking
        if self.whole_word_masking:
            masked_inputs, labels = self._whole_word_mask(
                batch["input_ids"],
                labels,
            )
        else:
            masked_inputs, labels = self._random_mask(
                batch["input_ids"],
                labels,
            )

        batch["input_ids"] = masked_inputs
        batch["labels"] = labels

        return batch

    def _random_mask(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random token masking."""
        # Create probability matrix
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = self._get_special_tokens_mask(inputs)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Don't mask padding
        padding_mask = inputs.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of time: Replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(inputs.shape, 0.8)
        ).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time: Replace with random token
        indices_random = torch.bernoulli(
            torch.full(inputs.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer),
            inputs.shape,
            dtype=torch.long,
        )
        inputs[indices_random] = random_words[indices_random]

        # 10% of time: Keep original token (already done)

        return inputs, labels

    def _whole_word_mask(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply whole-word masking.

        Masks entire words rather than subword tokens for better semantics.
        """
        batch_size, seq_length = inputs.shape

        for batch_idx in range(batch_size):
            input_ids = inputs[batch_idx].tolist()

            # Get word boundaries
            word_ids = self._get_word_ids(input_ids)

            # Group token indices by word
            word_to_tokens: Dict[int, List[int]] = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id not in word_to_tokens:
                        word_to_tokens[word_id] = []
                    word_to_tokens[word_id].append(token_idx)

            # Randomly select words to mask
            num_words = len(word_to_tokens)
            num_to_mask = max(1, int(num_words * self.mlm_probability))

            words_to_mask = random.sample(
                list(word_to_tokens.keys()),
                min(num_to_mask, num_words),
            )

            # Mask selected words
            for word_id in words_to_mask:
                token_indices = word_to_tokens[word_id]

                for token_idx in token_indices:
                    # Decide masking strategy
                    prob = random.random()
                    if prob < 0.8:
                        # Replace with [MASK]
                        inputs[batch_idx, token_idx] = self.tokenizer.mask_token_id
                    elif prob < 0.9:
                        # Replace with random token
                        inputs[batch_idx, token_idx] = random.randint(
                            0, len(self.tokenizer) - 1
                        )
                    # else: keep original (10%)

            # Set labels for non-masked tokens to -100
            masked_token_indices = set()
            for word_id in words_to_mask:
                masked_token_indices.update(word_to_tokens[word_id])

            for token_idx in range(seq_length):
                if token_idx not in masked_token_indices:
                    labels[batch_idx, token_idx] = -100

        return inputs, labels

    def _get_special_tokens_mask(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Get mask for special tokens."""
        special_tokens = set(self.tokenizer.all_special_ids)
        mask = torch.zeros_like(inputs, dtype=torch.bool)

        for token_id in special_tokens:
            mask |= inputs.eq(token_id)

        return mask

    def _get_word_ids(self, input_ids: List[int]) -> List[Optional[int]]:
        """Get word ID for each token.

        Tokens from the same word share the same word ID.
        Special tokens get None.
        """
        # Decode tokens to detect word boundaries
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        word_ids: List[Optional[int]] = []
        current_word = 0

        for token in tokens:
            if token in self.tokenizer.all_special_tokens:
                word_ids.append(None)
            elif token.startswith("##") or token.startswith("Ġ"):
                # Continuation of previous word (BERT/RoBERTa style)
                word_ids.append(current_word)
            else:
                # New word
                current_word += 1
                word_ids.append(current_word)

        return word_ids


@dataclass
class SpanMLMCollator:
    """Data collator for Span Masked Language Modeling.

    Implements span masking with Geometric distribution for span lengths,
    specifically designed for TAPT to reduce overfitting on small corpora.

    Unlike random token masking, span masking:
    - Masks contiguous spans of tokens
    - Uses Geometric(1/mean_span_length) distribution for span lengths
    - Encourages learning of longer-range dependencies
    - Reduces overfitting by making the task harder

    Reference: SpanBERT (Joshi et al., 2020)
    """

    tokenizer: Any
    mlm_probability: float = 0.15
    mean_span_length: float = 3.0
    max_span_length: int = 10
    pad_to_multiple_of: Optional[int] = None
    # Track masking statistics
    _span_lengths: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a mask token for Span MLM")

        if self.mean_span_length < 1.0:
            raise ValueError("mean_span_length must be >= 1.0")

        # Geometric distribution parameter: p = 1/μ
        self._geom_p = 1.0 / self.mean_span_length
        self._span_lengths = []

    def _sample_span_length(self) -> int:
        """Sample span length from Geometric distribution.

        Geometric distribution: P(X=k) = (1-p)^(k-1) * p
        where p = 1/mean_span_length

        Returns:
            Span length >= 1, capped at max_span_length.
        """
        # Sample from Geometric distribution (shifted by 1 since we want min length 1)
        span_length = np.random.geometric(self._geom_p)
        return min(span_length, self.max_span_length)

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Collate examples and apply span masking.

        Args:
            examples: List of tokenized examples.

        Returns:
            Batch with input_ids, attention_mask, and labels.
        """
        # Stack tensors
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }

        # Create labels (copy of input_ids)
        labels = batch["input_ids"].clone()

        # Apply span masking
        masked_inputs, labels = self._span_mask(
            batch["input_ids"],
            labels,
        )

        batch["input_ids"] = masked_inputs
        batch["labels"] = labels

        return batch

    def _span_mask(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply span masking with Geometric distribution span lengths.

        Algorithm:
        1. Calculate target number of tokens to mask (mlm_probability * seq_len)
        2. Randomly select span start positions
        3. For each span, sample length from Geometric(1/mean_span_length)
        4. Apply 80/10/10 masking strategy to each span

        Args:
            inputs: Input token IDs [batch_size, seq_length].
            labels: Label token IDs [batch_size, seq_length].

        Returns:
            Tuple of (masked_inputs, labels).
        """
        batch_size, seq_length = inputs.shape

        for batch_idx in range(batch_size):
            input_ids = inputs[batch_idx]

            # Get special token mask
            special_mask = self._get_special_tokens_mask(input_ids)
            padding_mask = input_ids.eq(self.tokenizer.pad_token_id)

            # Get valid positions (non-special, non-padding)
            valid_positions = ~(special_mask | padding_mask)
            valid_indices = torch.where(valid_positions)[0].tolist()

            if len(valid_indices) == 0:
                labels[batch_idx, :] = -100
                continue

            # Calculate target number of tokens to mask
            num_to_mask = max(1, int(len(valid_indices) * self.mlm_probability))

            # Select spans
            masked_positions = set()
            span_lengths_batch = []

            while len(masked_positions) < num_to_mask and valid_indices:
                # Sample span start position
                start_idx = random.choice(valid_indices)

                # Sample span length
                span_length = self._sample_span_length()
                span_lengths_batch.append(span_length)

                # Mask the span
                for offset in range(span_length):
                    pos = start_idx + offset
                    if pos < seq_length and valid_positions[pos]:
                        masked_positions.add(pos)

                    if len(masked_positions) >= num_to_mask:
                        break

                # Remove start position from valid indices
                if start_idx in valid_indices:
                    valid_indices.remove(start_idx)

            # Track span statistics
            self._span_lengths.extend(span_lengths_batch)

            # Apply masking to selected positions
            for pos in masked_positions:
                prob = random.random()
                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    inputs[batch_idx, pos] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10%: Replace with random token
                    inputs[batch_idx, pos] = random.randint(0, len(self.tokenizer) - 1)
                # 10%: Keep original

            # Set labels to -100 for non-masked positions
            for pos in range(seq_length):
                if pos not in masked_positions:
                    labels[batch_idx, pos] = -100

        return inputs, labels

    def _get_special_tokens_mask(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get mask for special tokens."""
        special_tokens = set(self.tokenizer.all_special_ids)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for token_id in special_tokens:
            mask |= input_ids.eq(token_id)

        return mask

    def get_span_statistics(self) -> Dict[str, float]:
        """Get statistics about span lengths used during masking.

        Returns:
            Dictionary with mean, std, min, max span lengths.
        """
        if not self._span_lengths:
            return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "count": 0}

        lengths = np.array(self._span_lengths)
        return {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "count": len(lengths),
        }

    def reset_statistics(self) -> None:
        """Reset span length statistics."""
        self._span_lengths = []


@dataclass
class CLMDataCollator:
    """Data collator for Causal Language Modeling.

    For decoder-only models (GPT, Llama, etc.).
    Labels are shifted input_ids for next-token prediction.
    """

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Collate examples for CLM.

        Args:
            examples: List of tokenized examples.

        Returns:
            Batch with input_ids, attention_mask, and labels.
        """
        # Stack tensors
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }

        # Labels are input_ids (shifted internally by the model)
        labels = batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch


@dataclass
class SpanCorruptionCollator:
    """Data collator for T5-style span corruption.

    Corrupts spans of text and replaces with sentinel tokens.
    The model learns to predict the corrupted spans.
    """

    tokenizer: Any
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0

    def __post_init__(self):
        """Validate tokenizer has sentinel tokens."""
        # T5 uses <extra_id_0>, <extra_id_1>, etc.
        if not hasattr(self.tokenizer, "additional_special_tokens"):
            logger.warning("Tokenizer may not have sentinel tokens for span corruption")

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Collate examples with span corruption.

        Args:
            examples: List of tokenized examples.

        Returns:
            Batch with corrupted inputs and target spans.
        """
        batch_inputs = []
        batch_labels = []

        for example in examples:
            input_ids = example["input_ids"]
            corrupted, labels = self._corrupt_spans(input_ids)
            batch_inputs.append(corrupted)
            batch_labels.append(labels)

        # Pad to same length
        max_input_len = max(len(x) for x in batch_inputs)
        max_label_len = max(len(x) for x in batch_labels)

        padded_inputs = []
        padded_labels = []
        attention_masks = []

        for inp, lab in zip(batch_inputs, batch_labels):
            # Pad inputs
            pad_len = max_input_len - len(inp)
            padded_inp = torch.cat([
                inp,
                torch.full((pad_len,), self.tokenizer.pad_token_id),
            ])
            padded_inputs.append(padded_inp)

            # Attention mask
            mask = torch.cat([
                torch.ones(len(inp)),
                torch.zeros(pad_len),
            ])
            attention_masks.append(mask)

            # Pad labels
            pad_len = max_label_len - len(lab)
            padded_lab = torch.cat([
                lab,
                torch.full((pad_len,), -100),
            ])
            padded_labels.append(padded_lab)

        return {
            "input_ids": torch.stack(padded_inputs),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(padded_labels),
        }

    def _corrupt_spans(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrupt random spans in input.

        Args:
            input_ids: Original input token IDs.

        Returns:
            Tuple of (corrupted_input, target_spans).
        """
        length = len(input_ids)

        # Calculate number of spans to corrupt
        num_noise_tokens = int(length * self.noise_density)
        num_spans = max(1, int(num_noise_tokens / self.mean_noise_span_length))

        # Generate span boundaries
        span_starts = sorted(random.sample(range(length), min(num_spans, length)))

        # Create mask for corrupted positions
        mask = torch.zeros(length, dtype=torch.bool)
        sentinel_tokens = []

        for i, start in enumerate(span_starts):
            span_length = int(random.expovariate(1.0 / self.mean_noise_span_length))
            span_length = max(1, min(span_length, length - start))

            end = start + span_length
            mask[start:end] = True
            sentinel_tokens.append((start, i))

        # Build corrupted input (replace spans with sentinels)
        corrupted = []
        labels = []
        current_sentinel = 0

        i = 0
        while i < length:
            if not mask[i]:
                corrupted.append(input_ids[i].item())
                i += 1
            else:
                # Add sentinel token
                sentinel_id = self._get_sentinel_id(current_sentinel)
                corrupted.append(sentinel_id)
                labels.append(sentinel_id)

                # Add original tokens to labels
                while i < length and mask[i]:
                    labels.append(input_ids[i].item())
                    i += 1

                current_sentinel += 1

        # Add final sentinel to labels
        labels.append(self.tokenizer.eos_token_id)

        return (
            torch.tensor(corrupted, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def _get_sentinel_id(self, idx: int) -> int:
        """Get sentinel token ID for index."""
        sentinel_token = f"<extra_id_{idx}>"
        if sentinel_token in self.tokenizer.additional_special_tokens:
            return self.tokenizer.convert_tokens_to_ids(sentinel_token)
        else:
            # Fallback: use unused tokens
            return self.tokenizer.unk_token_id


def create_data_collator(
    tokenizer: Any,
    objective: str = "mlm",
    mlm_probability: float = 0.15,
    whole_word_masking: bool = True,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    max_span_length: int = 10,
) -> Union[MLMDataCollator, SpanMLMCollator, CLMDataCollator, SpanCorruptionCollator]:
    """Create appropriate data collator based on objective.

    Args:
        tokenizer: HuggingFace tokenizer.
        objective: Pretraining objective (mlm, span_mlm, clm, span).
            - mlm: Standard Masked Language Modeling (random token masking)
            - span_mlm: Span MLM with Geometric distribution (for TAPT)
            - clm: Causal Language Modeling (GPT-style)
            - span: T5-style span corruption (encoder-decoder)
        mlm_probability: Masking probability for MLM/Span MLM.
        whole_word_masking: Use whole-word masking for MLM.
        noise_density: Noise density for span corruption.
        mean_noise_span_length: Mean span length for span_mlm or span corruption.
        max_span_length: Maximum span length for span_mlm.

    Returns:
        Appropriate data collator.
    """
    if objective == "mlm":
        return MLMDataCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            whole_word_masking=whole_word_masking,
        )
    elif objective == "span_mlm":
        return SpanMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            mean_span_length=mean_noise_span_length,
            max_span_length=max_span_length,
        )
    elif objective == "clm":
        return CLMDataCollator(tokenizer=tokenizer)
    elif objective == "span":
        return SpanCorruptionCollator(
            tokenizer=tokenizer,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
        )
    else:
        raise ValueError(f"Unknown objective: {objective}. Valid options: mlm, span_mlm, clm, span")
