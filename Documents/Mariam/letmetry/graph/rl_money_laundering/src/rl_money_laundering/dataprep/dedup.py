"""
Deduplication pipeline for AML corpus.

Provides exact and near-duplicate detection to clean training data:
- ExactDedup: SHA256 hash-based exact matching
- MinHashDedup: Locality-sensitive hashing for near-duplicates
- DedupPipeline: Orchestrates the full deduplication workflow

Near-duplicate detection uses MinHash LSH with configurable similarity threshold.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import TextSample, DatasetSource

logger = logging.getLogger(__name__)

# Try to import datasketch for MinHash LSH
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    logger.warning(
        "datasketch not available. Install with: pip install datasketch. "
        "Near-duplicate detection will use fallback implementation."
    )


@dataclass
class DedupStats:
    """Statistics from deduplication process."""

    total_input: int = 0
    total_output: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    too_short_removed: int = 0
    too_long_removed: int = 0

    # Per-source breakdown
    removed_by_source: Dict[str, int] = field(default_factory=dict)
    kept_by_source: Dict[str, int] = field(default_factory=dict)

    @property
    def total_removed(self) -> int:
        return self.total_input - self.total_output

    @property
    def dedup_ratio(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_removed / self.total_input

    def __str__(self) -> str:
        lines = [
            "Deduplication Statistics:",
            f"  Input samples: {self.total_input:,}",
            f"  Output samples: {self.total_output:,}",
            f"  Total removed: {self.total_removed:,} ({self.dedup_ratio:.1%})",
            f"    - Exact duplicates: {self.exact_duplicates_removed:,}",
            f"    - Near duplicates: {self.near_duplicates_removed:,}",
            f"    - Too short: {self.too_short_removed:,}",
            f"    - Too long: {self.too_long_removed:,}",
        ]
        if self.removed_by_source:
            lines.append("  Removed by source:")
            for source, count in sorted(self.removed_by_source.items()):
                lines.append(f"    {source}: {count:,}")
        return "\n".join(lines)


def normalize_text(text: str) -> str:
    """Normalize text for deduplication comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove punctuation variations
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def compute_hash(text: str, normalize: bool = True) -> str:
    """Compute SHA256 hash of text.

    Args:
        text: Input text.
        normalize: Whether to normalize text before hashing.

    Returns:
        Hex digest of SHA256 hash.
    """
    if normalize:
        text = normalize_text(text)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """Get k-shingles (character n-grams) from text.

    Args:
        text: Input text.
        k: Shingle size (default: 5 characters).

    Returns:
        Set of shingles.
    """
    text = normalize_text(text)
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def get_word_shingles(text: str, k: int = 3) -> Set[str]:
    """Get word-level k-shingles from text.

    Args:
        text: Input text.
        k: Number of words per shingle.

    Returns:
        Set of word shingles.
    """
    words = normalize_text(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


class BaseDeduplicator(ABC):
    """Abstract base class for deduplication strategies."""

    @abstractmethod
    def add(self, sample: TextSample) -> bool:
        """Add a sample and check if it's a duplicate.

        Args:
            sample: Text sample to add.

        Returns:
            True if sample was added (not a duplicate), False if duplicate.
        """
        pass

    @abstractmethod
    def is_duplicate(self, sample: TextSample) -> bool:
        """Check if a sample is a duplicate without adding it.

        Args:
            sample: Text sample to check.

        Returns:
            True if sample is a duplicate.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored hashes/signatures."""
        pass


class ExactDedup(BaseDeduplicator):
    """Exact duplicate detection using SHA256 hashes."""

    def __init__(self, normalize: bool = True) -> None:
        """Initialize exact deduplicator.

        Args:
            normalize: Whether to normalize text before hashing.
        """
        self.normalize = normalize
        self._seen_hashes: Set[str] = set()

    def add(self, sample: TextSample) -> bool:
        text_hash = compute_hash(sample.text, normalize=self.normalize)
        if text_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(text_hash)
        return True

    def is_duplicate(self, sample: TextSample) -> bool:
        text_hash = compute_hash(sample.text, normalize=self.normalize)
        return text_hash in self._seen_hashes

    def clear(self) -> None:
        self._seen_hashes.clear()

    @property
    def size(self) -> int:
        return len(self._seen_hashes)


class MinHashDedup(BaseDeduplicator):
    """Near-duplicate detection using MinHash LSH.

    Uses Locality-Sensitive Hashing to efficiently find documents
    with high Jaccard similarity.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        shingle_size: int = 5,
        use_word_shingles: bool = False,
    ) -> None:
        """Initialize MinHash deduplicator.

        Args:
            threshold: Jaccard similarity threshold for near-duplicates.
            num_perm: Number of permutations for MinHash.
            shingle_size: Size of shingles (characters or words).
            use_word_shingles: Use word-level shingles instead of character.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.use_word_shingles = use_word_shingles

        if DATASKETCH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._minhashes: Dict[str, MinHash] = {}
        else:
            # Fallback: simple hash-based bucketing
            self._buckets: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)

        self._sample_count = 0

    def _get_shingles(self, text: str) -> Set[str]:
        """Get shingles from text."""
        if self.use_word_shingles:
            return get_word_shingles(text, self.shingle_size)
        return get_shingles(text, self.shingle_size)

    def _create_minhash(self, shingles: Set[str]) -> "MinHash":
        """Create MinHash signature from shingles."""
        m = MinHash(num_perm=self.num_perm)
        for s in shingles:
            m.update(s.encode('utf-8'))
        return m

    def _compute_bucket_key(self, shingles: Set[str]) -> str:
        """Compute bucket key for fallback implementation."""
        # Use first few shingles sorted as bucket key
        sorted_shingles = sorted(shingles)[:10]
        return hashlib.md5("".join(sorted_shingles).encode()).hexdigest()[:8]

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def add(self, sample: TextSample) -> bool:
        shingles = self._get_shingles(sample.text)
        if not shingles:
            return True  # Empty text, not a duplicate

        sample_key = f"sample_{self._sample_count}"

        if DATASKETCH_AVAILABLE:
            minhash = self._create_minhash(shingles)

            # Check for near-duplicates
            result = self._lsh.query(minhash)
            if result:
                return False  # Found near-duplicate

            # Add to index
            self._lsh.insert(sample_key, minhash)
            self._minhashes[sample_key] = minhash
        else:
            # Fallback implementation
            bucket_key = self._compute_bucket_key(shingles)

            # Check existing items in bucket
            for existing_key, existing_shingles in self._buckets[bucket_key]:
                similarity = self._jaccard_similarity(shingles, existing_shingles)
                if similarity >= self.threshold:
                    return False  # Found near-duplicate

            # Add to bucket
            self._buckets[bucket_key].append((sample_key, shingles))

        self._sample_count += 1
        return True

    def is_duplicate(self, sample: TextSample) -> bool:
        shingles = self._get_shingles(sample.text)
        if not shingles:
            return False

        if DATASKETCH_AVAILABLE:
            minhash = self._create_minhash(shingles)
            result = self._lsh.query(minhash)
            return bool(result)
        else:
            bucket_key = self._compute_bucket_key(shingles)
            for _, existing_shingles in self._buckets[bucket_key]:
                similarity = self._jaccard_similarity(shingles, existing_shingles)
                if similarity >= self.threshold:
                    return True
            return False

    def clear(self) -> None:
        if DATASKETCH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            self._minhashes.clear()
        else:
            self._buckets.clear()
        self._sample_count = 0

    @property
    def size(self) -> int:
        return self._sample_count


@dataclass
class DedupConfig:
    """Configuration for deduplication pipeline."""

    # Length filters
    min_tokens: int = 50
    max_tokens: int = 10000
    min_chars: int = 100
    max_chars: int = 100000

    # Exact dedup
    enable_exact_dedup: bool = True
    normalize_for_exact: bool = True

    # Near-duplicate detection
    enable_near_dedup: bool = True
    similarity_threshold: float = 0.85
    num_perm: int = 128
    shingle_size: int = 5
    use_word_shingles: bool = False

    # Processing
    batch_size: int = 10000
    verbose: bool = True


class DedupPipeline:
    """Full deduplication pipeline combining multiple strategies.

    Applies deduplication in order:
    1. Length filtering (too short / too long)
    2. Exact duplicate removal
    3. Near-duplicate removal (optional)
    """

    def __init__(self, config: Optional[DedupConfig] = None) -> None:
        """Initialize deduplication pipeline.

        Args:
            config: Deduplication configuration.
        """
        self.config = config or DedupConfig()
        self.stats = DedupStats()

        # Initialize deduplicators
        self._exact_dedup: Optional[ExactDedup] = None
        self._near_dedup: Optional[MinHashDedup] = None

        if self.config.enable_exact_dedup:
            self._exact_dedup = ExactDedup(normalize=self.config.normalize_for_exact)

        if self.config.enable_near_dedup:
            self._near_dedup = MinHashDedup(
                threshold=self.config.similarity_threshold,
                num_perm=self.config.num_perm,
                shingle_size=self.config.shingle_size,
                use_word_shingles=self.config.use_word_shingles,
            )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Approximate: ~4 chars per token for English
        return len(text) // 4

    def _passes_length_filter(self, sample: TextSample) -> Tuple[bool, str]:
        """Check if sample passes length filters.

        Returns:
            Tuple of (passes, reason_if_failed).
        """
        char_count = len(sample.text)
        token_count = sample.token_count or self._estimate_tokens(sample.text)

        if char_count < self.config.min_chars or token_count < self.config.min_tokens:
            return False, "too_short"
        if char_count > self.config.max_chars or token_count > self.config.max_tokens:
            return False, "too_long"
        return True, ""

    def process(
        self,
        samples: Iterable[TextSample],
    ) -> Generator[TextSample, None, None]:
        """Process samples through the deduplication pipeline.

        Args:
            samples: Input samples to deduplicate.

        Yields:
            Deduplicated samples.
        """
        self.stats = DedupStats()

        for sample in samples:
            self.stats.total_input += 1
            source_key = sample.source.value

            # Length filter
            passes, reason = self._passes_length_filter(sample)
            if not passes:
                if reason == "too_short":
                    self.stats.too_short_removed += 1
                elif reason == "too_long":
                    self.stats.too_long_removed += 1
                self.stats.removed_by_source[source_key] = \
                    self.stats.removed_by_source.get(source_key, 0) + 1
                continue

            # Exact dedup
            if self._exact_dedup is not None:
                if not self._exact_dedup.add(sample):
                    self.stats.exact_duplicates_removed += 1
                    self.stats.removed_by_source[source_key] = \
                        self.stats.removed_by_source.get(source_key, 0) + 1
                    continue

            # Near-duplicate detection
            if self._near_dedup is not None:
                if not self._near_dedup.add(sample):
                    self.stats.near_duplicates_removed += 1
                    self.stats.removed_by_source[source_key] = \
                        self.stats.removed_by_source.get(source_key, 0) + 1
                    continue

            # Sample passed all filters
            self.stats.total_output += 1
            self.stats.kept_by_source[source_key] = \
                self.stats.kept_by_source.get(source_key, 0) + 1
            yield sample

            # Periodic logging
            if self.config.verbose and self.stats.total_input % 10000 == 0:
                logger.info(
                    f"Dedup progress: {self.stats.total_input:,} processed, "
                    f"{self.stats.total_output:,} kept ({1 - self.stats.dedup_ratio:.1%})"
                )

        if self.config.verbose:
            logger.info(str(self.stats))

    def process_batch(self, samples: List[TextSample]) -> List[TextSample]:
        """Process a batch of samples.

        Args:
            samples: List of samples to deduplicate.

        Returns:
            List of deduplicated samples.
        """
        return list(self.process(samples))

    def clear(self) -> None:
        """Clear all stored state and reset stats."""
        if self._exact_dedup is not None:
            self._exact_dedup.clear()
        if self._near_dedup is not None:
            self._near_dedup.clear()
        self.stats = DedupStats()

    def get_stats(self) -> DedupStats:
        """Get deduplication statistics."""
        return self.stats


def deduplicate_corpus(
    samples: Iterable[TextSample],
    min_tokens: int = 50,
    max_tokens: int = 10000,
    similarity_threshold: float = 0.85,
    enable_near_dedup: bool = True,
) -> Generator[TextSample, None, None]:
    """Convenience function to deduplicate a corpus.

    Args:
        samples: Input samples.
        min_tokens: Minimum token count.
        max_tokens: Maximum token count.
        similarity_threshold: Jaccard similarity threshold for near-duplicates.
        enable_near_dedup: Whether to enable near-duplicate detection.

    Yields:
        Deduplicated samples.

    Example:
        ```python
        from rl_money_laundering.dataprep.dedup import deduplicate_corpus

        # Simple dedup with defaults
        clean_samples = list(deduplicate_corpus(raw_samples))

        # Stricter dedup
        clean_samples = list(deduplicate_corpus(
            raw_samples,
            similarity_threshold=0.75,  # Catch more near-duplicates
            min_tokens=100,  # Require longer documents
        ))
        ```
    """
    config = DedupConfig(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        similarity_threshold=similarity_threshold,
        enable_near_dedup=enable_near_dedup,
    )
    pipeline = DedupPipeline(config)
    yield from pipeline.process(samples)
