"""
Unified corpus builder for continued pretraining.

Combines text from multiple AML-related sources into a unified corpus
suitable for domain-adaptive pretraining (DAPT) and task-adaptive
pretraining (TAPT).

Supports:
- Weighted mixing of sources
- Stratified sampling by category/label
- Chunk-based processing for long documents
- Export to HuggingFace datasets format
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

from .base import (
    BaseTextDataset,
    CorpusStats,
    DatasetConfig,
    DatasetSource,
    FraudLabel,
    TextCategory,
    TextChunker,
    TextSample,
)

logger = logging.getLogger(__name__)


@dataclass
class MixingConfig:
    """Configuration for corpus mixing."""

    # Source weights (higher = more samples)
    source_weights: Dict[DatasetSource, float] = field(default_factory=dict)

    # Category weights
    category_weights: Dict[TextCategory, float] = field(default_factory=dict)

    # Balancing options
    balance_labels: bool = False
    balance_categories: bool = False
    balance_sources: bool = False

    # Sampling
    max_samples_per_source: Optional[int] = None
    max_total_samples: Optional[int] = None

    # Deduplication
    deduplicate: bool = True
    dedup_threshold: float = 0.9  # Similarity threshold

    # Random seed
    random_seed: int = 42

    @classmethod
    def default_aml_weights(cls) -> "MixingConfig":
        """Default weights prioritizing AML-specific sources."""
        return cls(
            source_weights={
                DatasetSource.FINCEN_FILES: 2.0,  # Highest priority
                DatasetSource.FATF_TYPOLOGIES: 1.5,
                DatasetSource.SEC_EDGAR: 1.0,
                DatasetSource.OPENSANCTIONS: 1.0,
                DatasetSource.FRAUDNLP: 1.0,
                DatasetSource.VENMO_NOTES: 0.5,
                DatasetSource.GDELT: 0.3,
            },
            category_weights={
                TextCategory.SAR_NARRATIVE: 2.0,
                TextCategory.TYPOLOGY_CASE: 1.5,
                TextCategory.RED_FLAG_INDICATOR: 1.5,
                TextCategory.MD_AND_A: 1.0,
                TextCategory.ENTITY_DESCRIPTION: 1.0,
                TextCategory.ACTION_SEQUENCE: 1.0,
                TextCategory.PAYMENT_MEMO: 0.5,
                TextCategory.NEWS_ARTICLE: 0.3,
            },
        )

    @classmethod
    def balanced(cls) -> "MixingConfig":
        """Equal weights for all sources."""
        return cls(
            balance_sources=True,
            balance_categories=True,
        )


class UnifiedCorpusBuilder:
    """Builds a unified corpus from multiple text datasets.

    This builder combines text from various AML-related sources,
    applying configurable mixing strategies to create a balanced
    corpus for pretraining.

    Example usage:
        ```python
        builder = UnifiedCorpusBuilder(mixing_config=MixingConfig.default_aml_weights())

        builder.add_dataset(fincen_loader)
        builder.add_dataset(fatf_loader)
        builder.add_dataset(sec_loader)

        corpus = builder.build()
        builder.export_to_jsonl(Path("corpus.jsonl"))
        ```
    """

    def __init__(
        self,
        mixing_config: Optional[MixingConfig] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        """Initialize corpus builder.

        Args:
            mixing_config: Configuration for source mixing.
            chunk_size: Target chunk size for long documents.
            chunk_overlap: Overlap between chunks.
        """
        self.mixing_config = mixing_config or MixingConfig()
        self._datasets: List[BaseTextDataset] = []
        self._samples: List[TextSample] = []
        self._stats: Optional[CorpusStats] = None

        self._chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._seen_hashes: Set[str] = set()

    def add_dataset(self, dataset: BaseTextDataset) -> "UnifiedCorpusBuilder":
        """Add a dataset to the corpus.

        Args:
            dataset: Dataset loader to add.

        Returns:
            Self for chaining.
        """
        self._datasets.append(dataset)
        self._samples = []  # Reset built corpus
        self._stats = None
        return self

    def add_samples(self, samples: List[TextSample]) -> "UnifiedCorpusBuilder":
        """Add raw samples directly.

        Args:
            samples: List of TextSample objects.

        Returns:
            Self for chaining.
        """
        self._samples.extend(samples)
        self._stats = None
        return self

    def build(self) -> List[TextSample]:
        """Build the unified corpus.

        Applies mixing configuration to combine samples from all sources.

        Returns:
            List of TextSample objects forming the corpus.
        """
        random.seed(self.mixing_config.random_seed)

        # Collect all samples
        all_samples: List[TextSample] = list(self._samples)

        for dataset in self._datasets:
            try:
                dataset_samples = dataset.get_samples()
                logger.info(
                    f"Collected {len(dataset_samples)} samples from {dataset.name}"
                )
                all_samples.extend(dataset_samples)
            except Exception as e:
                logger.error(f"Error loading {dataset.name}: {e}")

        logger.info(f"Total samples before processing: {len(all_samples)}")

        # Apply source limit
        if self.mixing_config.max_samples_per_source:
            all_samples = self._limit_per_source(all_samples)

        # Deduplicate
        if self.mixing_config.deduplicate:
            all_samples = self._deduplicate(all_samples)
            logger.info(f"Samples after deduplication: {len(all_samples)}")

        # Apply weights
        if self.mixing_config.source_weights or self.mixing_config.category_weights:
            all_samples = self._apply_weights(all_samples)

        # Balance if requested
        if self.mixing_config.balance_sources:
            all_samples = self._balance_by_source(all_samples)
        if self.mixing_config.balance_categories:
            all_samples = self._balance_by_category(all_samples)
        if self.mixing_config.balance_labels:
            all_samples = self._balance_by_label(all_samples)

        # Apply total limit
        if self.mixing_config.max_total_samples:
            if len(all_samples) > self.mixing_config.max_total_samples:
                all_samples = random.sample(
                    all_samples,
                    self.mixing_config.max_total_samples,
                )

        # Shuffle
        random.shuffle(all_samples)

        self._samples = all_samples
        self._stats = None  # Reset stats

        logger.info(f"Final corpus size: {len(self._samples)}")
        return self._samples

    def _limit_per_source(self, samples: List[TextSample]) -> List[TextSample]:
        """Limit samples per source."""
        by_source: Dict[DatasetSource, List[TextSample]] = defaultdict(list)
        for s in samples:
            by_source[s.source].append(s)

        limited = []
        max_per = self.mixing_config.max_samples_per_source
        for source, source_samples in by_source.items():
            if len(source_samples) > max_per:
                source_samples = random.sample(source_samples, max_per)
            limited.extend(source_samples)

        return limited

    def _deduplicate(self, samples: List[TextSample]) -> List[TextSample]:
        """Remove duplicate samples based on text hash."""
        unique = []
        seen = set()

        for sample in samples:
            # Simple hash-based dedup
            text_hash = hash(sample.text[:500])  # Use first 500 chars
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(sample)

        return unique

    def _apply_weights(self, samples: List[TextSample]) -> List[TextSample]:
        """Apply source and category weights via resampling."""
        weighted = []

        for sample in samples:
            # Get weight multiplier
            source_weight = self.mixing_config.source_weights.get(sample.source, 1.0)
            category_weight = self.mixing_config.category_weights.get(sample.category, 1.0)

            combined_weight = source_weight * category_weight

            # Probabilistic resampling
            if combined_weight >= 1.0:
                # Include at least once
                weighted.append(sample)
                # Maybe include additional copies
                extra_copies = int(combined_weight) - 1
                extra_prob = combined_weight - int(combined_weight)
                for _ in range(extra_copies):
                    weighted.append(sample)
                if random.random() < extra_prob:
                    weighted.append(sample)
            else:
                # Include with probability
                if random.random() < combined_weight:
                    weighted.append(sample)

        return weighted

    def _balance_by_source(self, samples: List[TextSample]) -> List[TextSample]:
        """Balance samples across sources."""
        by_source: Dict[DatasetSource, List[TextSample]] = defaultdict(list)
        for s in samples:
            by_source[s.source].append(s)

        # Find minimum count
        min_count = min(len(v) for v in by_source.values()) if by_source else 0

        balanced = []
        for source_samples in by_source.values():
            if len(source_samples) > min_count:
                source_samples = random.sample(source_samples, min_count)
            balanced.extend(source_samples)

        return balanced

    def _balance_by_category(self, samples: List[TextSample]) -> List[TextSample]:
        """Balance samples across categories."""
        by_category: Dict[TextCategory, List[TextSample]] = defaultdict(list)
        for s in samples:
            by_category[s.category].append(s)

        min_count = min(len(v) for v in by_category.values()) if by_category else 0

        balanced = []
        for cat_samples in by_category.values():
            if len(cat_samples) > min_count:
                cat_samples = random.sample(cat_samples, min_count)
            balanced.extend(cat_samples)

        return balanced

    def _balance_by_label(self, samples: List[TextSample]) -> List[TextSample]:
        """Balance samples across labels."""
        by_label: Dict[FraudLabel, List[TextSample]] = defaultdict(list)
        for s in samples:
            by_label[s.label].append(s)

        # Exclude UNKNOWN from balancing
        labeled_samples = {
            k: v for k, v in by_label.items()
            if k != FraudLabel.UNKNOWN
        }

        if not labeled_samples:
            return samples

        min_count = min(len(v) for v in labeled_samples.values())

        balanced = list(by_label[FraudLabel.UNKNOWN])  # Keep all unknown
        for label_samples in labeled_samples.values():
            if len(label_samples) > min_count:
                label_samples = random.sample(label_samples, min_count)
            balanced.extend(label_samples)

        return balanced

    def iter_chunks(self) -> Generator[TextSample, None, None]:
        """Iterate over chunked samples.

        Long documents are split into training-sized chunks.

        Yields:
            TextSample objects.
        """
        for sample in self._samples:
            for chunk in self._chunker.chunk_sample(sample):
                yield chunk

    def get_corpus(self) -> List[TextSample]:
        """Get the built corpus."""
        if not self._samples:
            return self.build()
        return self._samples

    @property
    def stats(self) -> CorpusStats:
        """Get corpus statistics."""
        if self._stats is None:
            self._stats = CorpusStats()
            for sample in self.get_corpus():
                self._stats.update(sample)
            self._stats.finalize()
        return self._stats

    def export_to_jsonl(
        self,
        output_path: Path,
        chunk_documents: bool = True,
    ) -> int:
        """Export corpus to JSONL format.

        Args:
            output_path: Output file path.
            chunk_documents: Whether to chunk long documents.

        Returns:
            Number of samples written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            if chunk_documents:
                for sample in self.iter_chunks():
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
            else:
                for sample in self.get_corpus():
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    count += 1

        logger.info(f"Exported {count} samples to {output_path}")
        return count

    def export_to_hf_dataset(
        self,
        output_dir: Optional[Path] = None,
        chunk_documents: bool = True,
    ) -> Any:
        """Export to HuggingFace Dataset format.

        Args:
            output_dir: Optional directory to save dataset.
            chunk_documents: Whether to chunk long documents.

        Returns:
            HuggingFace Dataset object.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        if chunk_documents:
            data = [s.to_training_format() for s in self.iter_chunks()]
        else:
            data = [s.to_training_format() for s in self.get_corpus()]

        dataset = Dataset.from_list(data)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(output_dir))
            logger.info(f"Saved dataset to {output_dir}")

        return dataset

    def export_for_mlm(
        self,
        output_path: Path,
        text_only: bool = True,
    ) -> int:
        """Export corpus for masked language model pretraining.

        Args:
            output_path: Output file path.
            text_only: If True, export just text (one per line).

        Returns:
            Number of samples written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self.iter_chunks():
                if text_only:
                    f.write(sample.text + "\n")
                else:
                    f.write(json.dumps({"text": sample.text}, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"Exported {count} samples for MLM to {output_path}")
        return count

    def get_split(
        self,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ) -> Tuple[List[TextSample], List[TextSample], List[TextSample]]:
        """Split corpus into train/val/test sets.

        Args:
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
            test_ratio: Proportion for testing.

        Returns:
            Tuple of (train, val, test) sample lists.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

        samples = self.get_corpus()
        n = len(samples)

        # Shuffle with seed for reproducibility
        random.seed(self.mixing_config.random_seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def print_summary(self) -> None:
        """Print corpus summary."""
        print(self.stats)
        print()

        # Source distribution
        print("Source distribution:")
        for source, count in sorted(
            self.stats.samples_by_source.items(),
            key=lambda x: -x[1],
        ):
            pct = 100 * count / self.stats.total_samples
            print(f"  {source}: {count:,} ({pct:.1f}%)")

        print()

        # Category distribution
        print("Category distribution:")
        for category, count in sorted(
            self.stats.samples_by_category.items(),
            key=lambda x: -x[1],
        ):
            pct = 100 * count / self.stats.total_samples
            print(f"  {category}: {count:,} ({pct:.1f}%)")


def build_aml_pretraining_corpus(
    data_dir: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> UnifiedCorpusBuilder:
    """Convenience function to build an AML pretraining corpus.

    Args:
        data_dir: Directory containing raw data.
        output_dir: Directory for output files.
        max_samples: Optional sample limit.

    Returns:
        Configured UnifiedCorpusBuilder.
    """
    from .fincen_loader import FinCENLoader
    from .sec_fraud_loader import SECFraudLoader
    from .fatf_loader import FATFTypologyLoader
    from .opensanctions_loader import OpenSanctionsLoader
    from .fraudnlp_loader import FraudNLPLoader

    config = DatasetConfig(
        data_dir=data_dir,
        cache_dir=data_dir / ".cache",
        output_dir=output_dir,
        max_samples=max_samples,
        auto_download=True,
    )

    builder = UnifiedCorpusBuilder(
        mixing_config=MixingConfig.default_aml_weights(),
    )

    # Add available datasets
    try:
        builder.add_dataset(FinCENLoader(config))
    except Exception as e:
        logger.warning(f"Could not load FinCEN: {e}")

    try:
        builder.add_dataset(SECFraudLoader(config))
    except Exception as e:
        logger.warning(f"Could not load SEC: {e}")

    try:
        builder.add_dataset(FATFTypologyLoader(config))
    except Exception as e:
        logger.warning(f"Could not load FATF: {e}")

    try:
        builder.add_dataset(OpenSanctionsLoader(config))
    except Exception as e:
        logger.warning(f"Could not load OpenSanctions: {e}")

    try:
        builder.add_dataset(FraudNLPLoader(config))
    except Exception as e:
        logger.warning(f"Could not load FraudNLP: {e}")

    return builder
