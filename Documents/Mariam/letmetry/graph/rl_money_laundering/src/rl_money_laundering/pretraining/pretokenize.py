"""
Pre-tokenization script for efficient training.

Pre-tokenizes the corpus offline and saves as memory-mapped numpy arrays
for efficient streaming during training. This eliminates tokenization
bottleneck and enables fast random access to pre-tokenized data.

Usage:
    from rl_money_laundering.pretraining.pretokenize import (
        pretokenize_corpus,
        PreTokenizedDataset,
        PackedDataset,
    )

    # Pre-tokenize corpus
    pretokenize_corpus(
        input_path="./data/corpus.jsonl",
        output_dir="./data/pretokenized/",
        tokenizer_name="roberta-base",
        max_length=512,
    )

    # Load pre-tokenized dataset
    dataset = PreTokenizedDataset("./data/pretokenized/")

    # Or use packed dataset for efficiency
    packed = PackedDataset("./data/pretokenized/", pack_length=512)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PreTokenizeConfig:
    """Configuration for pre-tokenization."""

    # Input/Output
    input_path: Path
    output_dir: Path

    # Tokenizer settings
    tokenizer_name: str = "roberta-base"
    max_length: int = 512

    # Processing settings
    text_column: str = "text"
    add_special_tokens: bool = True
    num_workers: int = 4
    chunk_size: int = 10000  # Samples per chunk

    # Output format
    dtype: str = "int32"  # numpy dtype for token IDs

    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)


@dataclass
class PreTokenizeStats:
    """Statistics from pre-tokenization."""

    total_samples: int = 0
    total_tokens: int = 0
    avg_length: float = 0.0
    truncated_samples: int = 0
    empty_samples: int = 0
    chunks_written: int = 0
    output_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "avg_length": self.avg_length,
            "truncated_samples": self.truncated_samples,
            "empty_samples": self.empty_samples,
            "chunks_written": self.chunks_written,
            "output_size_mb": self.output_size_mb,
        }


class PreTokenizer:
    """Pre-tokenizes a corpus and saves to disk.

    Saves data in the following format:
    - input_ids.npy: Memory-mapped array of shape (N, max_length)
    - attention_mask.npy: Memory-mapped array of shape (N, max_length)
    - metadata.json: Tokenizer info and statistics
    """

    def __init__(self, config: PreTokenizeConfig) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for pre-tokenization")

        self.config = config
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            use_fast=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(self) -> PreTokenizeStats:
        """Run pre-tokenization pipeline.

        Returns:
            Statistics about the pre-tokenization.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        stats = PreTokenizeStats()

        # First pass: count samples and get token lengths
        logger.info("First pass: counting samples...")
        sample_count, token_lengths = self._count_samples()
        stats.total_samples = sample_count

        if sample_count == 0:
            logger.warning("No samples found in input file")
            return stats

        # Calculate statistics
        stats.avg_length = np.mean(token_lengths)
        stats.truncated_samples = sum(
            1 for l in token_lengths if l > self.config.max_length
        )

        logger.info(
            f"Found {sample_count} samples, avg length {stats.avg_length:.1f} tokens"
        )
        if stats.truncated_samples > 0:
            logger.warning(
                f"{stats.truncated_samples} samples will be truncated "
                f"({100 * stats.truncated_samples / sample_count:.1f}%)"
            )

        # Create memory-mapped arrays
        dtype = np.dtype(self.config.dtype)
        input_ids_path = self.config.output_dir / "input_ids.npy"
        attention_mask_path = self.config.output_dir / "attention_mask.npy"

        # Create empty memory-mapped files
        input_ids = np.memmap(
            input_ids_path,
            dtype=dtype,
            mode="w+",
            shape=(sample_count, self.config.max_length),
        )
        attention_mask = np.memmap(
            attention_mask_path,
            dtype=np.uint8,  # Binary mask, use uint8
            mode="w+",
            shape=(sample_count, self.config.max_length),
        )

        # Second pass: tokenize and write
        logger.info("Second pass: tokenizing and writing...")
        idx = 0
        total_tokens = 0

        with open(self.config.input_path, "r", encoding="utf-8") as f:
            batch_texts = []

            for line in tqdm(f, total=sample_count, desc="Tokenizing"):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    text = data.get(self.config.text_column, "")
                    if not text:
                        stats.empty_samples += 1
                        continue

                    batch_texts.append(text)

                    # Process in batches for efficiency
                    if len(batch_texts) >= self.config.chunk_size:
                        written, tokens = self._write_batch(
                            batch_texts, input_ids, attention_mask, idx
                        )
                        idx += written
                        total_tokens += tokens
                        batch_texts = []

                except json.JSONDecodeError:
                    continue

            # Write remaining samples
            if batch_texts:
                written, tokens = self._write_batch(
                    batch_texts, input_ids, attention_mask, idx
                )
                idx += written
                total_tokens += tokens

        # Flush memory-mapped files
        input_ids.flush()
        attention_mask.flush()
        del input_ids
        del attention_mask

        # Save metadata
        stats.total_tokens = total_tokens
        stats.chunks_written = 1  # Single file format
        stats.output_size_mb = (
            input_ids_path.stat().st_size + attention_mask_path.stat().st_size
        ) / (1024 ** 2)

        metadata = {
            "tokenizer_name": self.config.tokenizer_name,
            "max_length": self.config.max_length,
            "vocab_size": len(self.tokenizer),
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_samples": sample_count,
            "dtype": self.config.dtype,
            "stats": stats.to_dict(),
        }

        with open(self.config.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Pre-tokenization complete: {sample_count} samples, "
            f"{stats.output_size_mb:.1f} MB"
        )

        return stats

    def _count_samples(self) -> Tuple[int, List[int]]:
        """Count samples and estimate token lengths."""
        count = 0
        lengths = []

        with open(self.config.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    text = data.get(self.config.text_column, "")
                    if text:
                        count += 1
                        # Estimate token length (rough approximation)
                        # More accurate would be to tokenize, but slower
                        estimated_tokens = len(text.split()) * 1.3
                        lengths.append(int(estimated_tokens))
                except json.JSONDecodeError:
                    continue

        return count, lengths

    def _write_batch(
        self,
        texts: List[str],
        input_ids: np.memmap,
        attention_mask: np.memmap,
        start_idx: int,
    ) -> Tuple[int, int]:
        """Tokenize and write a batch of texts.

        Returns:
            Tuple of (samples_written, total_tokens).
        """
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=self.config.add_special_tokens,
            return_tensors="np",
        )

        batch_size = len(texts)
        end_idx = start_idx + batch_size

        input_ids[start_idx:end_idx] = encodings["input_ids"]
        attention_mask[start_idx:end_idx] = encodings["attention_mask"]

        # Count non-padding tokens
        total_tokens = int(encodings["attention_mask"].sum())

        return batch_size, total_tokens


class PreTokenizedDataset(Dataset if HAS_TORCH else object):
    """Dataset that reads from pre-tokenized memory-mapped files.

    Provides fast random access to pre-tokenized samples without
    tokenization overhead during training.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize pre-tokenized dataset.

        Args:
            data_dir: Directory containing pre-tokenized files.
            max_samples: Optional limit on number of samples.
        """
        if not HAS_TORCH:
            raise ImportError("torch required for PreTokenizedDataset")

        self.data_dir = Path(data_dir)

        # Load metadata
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.num_samples = self.metadata["num_samples"]
        self.max_length = self.metadata["max_length"]
        dtype = np.dtype(self.metadata["dtype"])

        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)

        # Memory-map the arrays (read-only)
        self.input_ids = np.memmap(
            self.data_dir / "input_ids.npy",
            dtype=dtype,
            mode="r",
            shape=(self.metadata["num_samples"], self.max_length),
        )
        self.attention_mask = np.memmap(
            self.data_dir / "attention_mask.npy",
            dtype=np.uint8,
            mode="r",
            shape=(self.metadata["num_samples"], self.max_length),
        )

        logger.info(
            f"Loaded pre-tokenized dataset: {self.num_samples} samples, "
            f"max_length={self.max_length}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Get a single sample."""
        import torch

        return {
            "input_ids": torch.from_numpy(self.input_ids[idx].copy()).long(),
            "attention_mask": torch.from_numpy(
                self.attention_mask[idx].copy()
            ).long(),
        }


class PackedDataset(Dataset if HAS_TORCH else object):
    """Dataset that packs multiple short documents into single sequences.

    Reduces padding waste by concatenating short documents to fill
    the context window. Documents are separated by [SEP] tokens.

    Example:
        [CLS] Doc1 [SEP] Doc2 [SEP] Doc3 [SEP] [PAD] [PAD]

    This can significantly reduce training time by eliminating
    padding tokens that don't contribute to learning.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        pack_length: int = 512,
        separator_token_id: Optional[int] = None,
        shuffle_docs: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize packed dataset.

        Args:
            data_dir: Directory containing pre-tokenized files.
            pack_length: Target sequence length for packing.
            separator_token_id: Token ID to separate documents.
            shuffle_docs: Whether to shuffle document order.
            seed: Random seed for shuffling.
            max_samples: Maximum packed samples to generate.
        """
        if not HAS_TORCH:
            raise ImportError("torch required for PackedDataset")

        self.data_dir = Path(data_dir)
        self.pack_length = pack_length
        self.shuffle_docs = shuffle_docs
        self.seed = seed
        self.max_samples = max_samples

        # Load metadata
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.pad_token_id = self.metadata["pad_token_id"]
        self.sep_token_id = separator_token_id or 2  # Default [SEP] for BERT/RoBERTa

        # Load pre-tokenized data
        dtype = np.dtype(self.metadata["dtype"])
        num_samples = self.metadata["num_samples"]
        max_length = self.metadata["max_length"]

        self.input_ids = np.memmap(
            self.data_dir / "input_ids.npy",
            dtype=dtype,
            mode="r",
            shape=(num_samples, max_length),
        )
        self.attention_mask = np.memmap(
            self.data_dir / "attention_mask.npy",
            dtype=np.uint8,
            mode="r",
            shape=(num_samples, max_length),
        )

        # Build packed samples
        self._build_packed_samples()

    def _build_packed_samples(self) -> None:
        """Pre-compute packed sample indices."""
        logger.info("Building packed samples...")

        # Get document lengths (excluding padding)
        doc_lengths = np.sum(self.attention_mask, axis=1)

        # Create document index order
        doc_indices = np.arange(len(doc_lengths))
        if self.shuffle_docs:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(doc_indices)

        # Pack documents into sequences
        self.packed_samples: List[List[int]] = []
        current_pack: List[int] = []
        current_length = 0

        for doc_idx in doc_indices:
            doc_len = int(doc_lengths[doc_idx])

            # Skip empty documents
            if doc_len <= 2:  # Only [CLS] and [SEP]
                continue

            # Check if document fits in current pack
            # +1 for separator token between documents
            space_needed = doc_len + (1 if current_pack else 0)

            if current_length + space_needed <= self.pack_length:
                # Add to current pack
                current_pack.append(doc_idx)
                current_length += space_needed
            else:
                # Save current pack and start new one
                if current_pack:
                    self.packed_samples.append(current_pack)
                    if (
                        self.max_samples is not None
                        and len(self.packed_samples) >= self.max_samples
                    ):
                        break

                # Start new pack with current document
                if doc_len <= self.pack_length:
                    current_pack = [doc_idx]
                    current_length = doc_len
                else:
                    # Document too long, truncate it as its own pack
                    current_pack = [doc_idx]
                    current_length = self.pack_length

        # Don't forget last pack
        if current_pack and (
            self.max_samples is None or len(self.packed_samples) < self.max_samples
        ):
            self.packed_samples.append(current_pack)

        logger.info(
            f"Created {len(self.packed_samples)} packed samples from "
            f"{self.metadata['num_samples']} documents"
        )

        # Calculate packing efficiency
        total_tokens = sum(doc_lengths[i] for pack in self.packed_samples for i in pack)
        total_capacity = len(self.packed_samples) * self.pack_length
        efficiency = total_tokens / total_capacity if total_capacity > 0 else 0
        logger.info(f"Packing efficiency: {efficiency:.1%}")

    def __len__(self) -> int:
        return len(self.packed_samples)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Get a packed sample."""
        import torch

        doc_indices = self.packed_samples[idx]

        # Build packed sequence
        packed_ids = []
        packed_mask = []

        for i, doc_idx in enumerate(doc_indices):
            # Get document tokens (excluding padding)
            doc_ids = self.input_ids[doc_idx]
            doc_mask = self.attention_mask[doc_idx]
            doc_len = int(np.sum(doc_mask))

            # Add separator between documents (not before first)
            if i > 0:
                packed_ids.append(self.sep_token_id)
                packed_mask.append(1)

            # Add document tokens
            packed_ids.extend(doc_ids[:doc_len].tolist())
            packed_mask.extend([1] * doc_len)

            # Stop if we've filled the pack
            if len(packed_ids) >= self.pack_length:
                packed_ids = packed_ids[: self.pack_length]
                packed_mask = packed_mask[: self.pack_length]
                break

        # Pad to pack_length
        pad_length = self.pack_length - len(packed_ids)
        if pad_length > 0:
            packed_ids.extend([self.pad_token_id] * pad_length)
            packed_mask.extend([0] * pad_length)

        return {
            "input_ids": torch.tensor(packed_ids, dtype=torch.long),
            "attention_mask": torch.tensor(packed_mask, dtype=torch.long),
        }

    def get_packing_stats(self) -> Dict[str, Any]:
        """Get statistics about packing efficiency.

        Returns:
            Dictionary with packing statistics.
        """
        doc_lengths = np.sum(self.attention_mask, axis=1)

        docs_per_pack = [len(pack) for pack in self.packed_samples]
        tokens_per_pack = [
            sum(doc_lengths[i] for i in pack) for pack in self.packed_samples
        ]

        return {
            "num_packed_samples": len(self.packed_samples),
            "num_original_docs": self.metadata["num_samples"],
            "avg_docs_per_pack": np.mean(docs_per_pack),
            "avg_tokens_per_pack": np.mean(tokens_per_pack),
            "pack_length": self.pack_length,
            "packing_efficiency": np.mean(tokens_per_pack) / self.pack_length,
        }


class StreamingPreTokenizedDataset(IterableDataset if HAS_TORCH else object):
    """Streaming dataset for very large pre-tokenized corpora.

    Reads data in chunks with a shuffle buffer for randomization.
    Memory-efficient for corpora too large to fit in RAM.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            data_dir: Directory containing pre-tokenized files.
            shuffle_buffer_size: Size of shuffle buffer.
            seed: Random seed for shuffling.
            num_workers: Number of data loading workers.
        """
        if not HAS_TORCH:
            raise ImportError("torch required for StreamingPreTokenizedDataset")

        self.data_dir = Path(data_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.num_workers = num_workers

        # Load metadata
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.num_samples = self.metadata["num_samples"]
        self.max_length = self.metadata["max_length"]

    def __iter__(self) -> Iterator[Dict[str, "torch.Tensor"]]:
        """Iterate through samples with shuffling."""
        import torch

        # Load memory-mapped arrays
        dtype = np.dtype(self.metadata["dtype"])
        input_ids = np.memmap(
            self.data_dir / "input_ids.npy",
            dtype=dtype,
            mode="r",
            shape=(self.num_samples, self.max_length),
        )
        attention_mask = np.memmap(
            self.data_dir / "attention_mask.npy",
            dtype=np.uint8,
            mode="r",
            shape=(self.num_samples, self.max_length),
        )

        # Create shuffled indices
        rng = np.random.default_rng(self.seed)
        indices = np.arange(self.num_samples)
        rng.shuffle(indices)

        # Shuffle buffer
        buffer: List[Dict[str, "torch.Tensor"]] = []

        for idx in indices:
            sample = {
                "input_ids": torch.from_numpy(input_ids[idx].copy()).long(),
                "attention_mask": torch.from_numpy(attention_mask[idx].copy()).long(),
            }
            buffer.append(sample)

            if len(buffer) >= self.shuffle_buffer_size:
                rng.shuffle(buffer)
                # Yield half the buffer
                for item in buffer[: self.shuffle_buffer_size // 2]:
                    yield item
                buffer = buffer[self.shuffle_buffer_size // 2 :]

        # Flush remaining buffer
        rng.shuffle(buffer)
        for item in buffer:
            yield item


def pretokenize_corpus(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer_name: str = "roberta-base",
    max_length: int = 512,
    text_column: str = "text",
    num_workers: int = 4,
) -> PreTokenizeStats:
    """Pre-tokenize a corpus and save to disk.

    Convenience function for common use case.

    Args:
        input_path: Path to JSONL input file.
        output_dir: Directory for output files.
        tokenizer_name: Name or path of tokenizer.
        max_length: Maximum sequence length.
        text_column: Name of text column in JSON.
        num_workers: Number of worker processes.

    Returns:
        Pre-tokenization statistics.
    """
    config = PreTokenizeConfig(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        text_column=text_column,
        num_workers=num_workers,
    )

    pretokenizer = PreTokenizer(config)
    return pretokenizer.run()


def main():
    """CLI entry point for pre-tokenization."""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize corpus for training")
    parser.add_argument("input_path", type=str, help="Path to JSONL input file")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="roberta-base",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    stats = pretokenize_corpus(
        input_path=args.input_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        text_column=args.text_column,
    )

    print("\nPre-tokenization Statistics:")
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
