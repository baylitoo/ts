"""
Base classes and protocols for text-rich AML dataset loading.

Provides unified interfaces for all dataset loaders to ensure
consistent data format for downstream pretraining and fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


class ComplianceMode(str, Enum):
    """Compliance mode for data governance.

    Controls which data sources are allowed based on legal/regulatory constraints.

    - RESEARCH: All sources enabled (for academic research only)
    - STRICT: Disables leaked/restricted sources (bank-safe)
    - PUBLIC_ONLY: Only publicly available, clearly licensed data
    """

    RESEARCH = "research"
    STRICT = "strict"
    PUBLIC_ONLY = "public_only"


class DatasetSource(str, Enum):
    """Enumeration of supported dataset sources."""

    FINCEN_FILES = "fincen_files"
    SEC_EDGAR = "sec_edgar"
    FRAUDNLP = "fraudnlp"
    FATF_TYPOLOGIES = "fatf_typologies"
    OPENSANCTIONS = "opensanctions"
    VENMO_NOTES = "venmo_notes"
    GDELT = "gdelt"
    ICIJ_OFFSHORE = "icij_offshore"
    FINCEN_ADVISORIES = "fincen_advisories"  # Public advisories (not leaked SARs)
    DOJ_ENFORCEMENT = "doj_enforcement"  # DOJ press releases and actions
    SEC_ENFORCEMENT = "sec_enforcement"  # SEC enforcement actions
    CUSTOM = "custom"


# Data governance: sources that should be disabled in strict/production mode
RESTRICTED_SOURCES = frozenset({
    DatasetSource.FINCEN_FILES,  # Leaked SARs - legal risk
    DatasetSource.VENMO_NOTES,   # Privacy concerns, not AML text
    DatasetSource.ICIJ_OFFSHORE, # Leaked documents
})

# Sources safe for public/production use (clear licensing)
PUBLIC_SOURCES = frozenset({
    DatasetSource.SEC_EDGAR,
    DatasetSource.FATF_TYPOLOGIES,
    DatasetSource.OPENSANCTIONS,  # CC-BY licensed
    DatasetSource.FINCEN_ADVISORIES,
    DatasetSource.DOJ_ENFORCEMENT,
    DatasetSource.SEC_ENFORCEMENT,
})


def get_allowed_sources(compliance_mode: ComplianceMode) -> frozenset:
    """Get allowed data sources for a given compliance mode.

    Args:
        compliance_mode: The compliance mode to check.

    Returns:
        Frozenset of allowed DatasetSource values.
    """
    all_sources = frozenset(DatasetSource) - {DatasetSource.CUSTOM}

    if compliance_mode == ComplianceMode.RESEARCH:
        return all_sources
    elif compliance_mode == ComplianceMode.STRICT:
        return all_sources - RESTRICTED_SOURCES
    elif compliance_mode == ComplianceMode.PUBLIC_ONLY:
        return PUBLIC_SOURCES
    else:
        return all_sources


class TextCategory(str, Enum):
    """Categories of text for training purposes."""

    # High-signal AML text
    SAR_NARRATIVE = "sar_narrative"
    TYPOLOGY_CASE = "typology_case"
    RED_FLAG_INDICATOR = "red_flag_indicator"
    INVESTIGATION_REPORT = "investigation_report"

    # Financial/legal text
    MD_AND_A = "md_and_a"
    FINANCIAL_STATEMENT = "financial_statement"
    RISK_DISCLOSURE = "risk_disclosure"
    LEGAL_FILING = "legal_filing"

    # Entity/sanctions text
    ENTITY_DESCRIPTION = "entity_description"
    SANCTIONS_REASON = "sanctions_reason"
    PEP_PROFILE = "pep_profile"

    # Transactional text
    PAYMENT_MEMO = "payment_memo"
    TRANSACTION_PURPOSE = "transaction_purpose"
    ACTION_SEQUENCE = "action_sequence"

    # News/media text
    ADVERSE_MEDIA = "adverse_media"
    NEWS_ARTICLE = "news_article"

    # Generic
    UNKNOWN = "unknown"


class FraudLabel(str, Enum):
    """Fraud/AML labels for supervised training."""

    FRAUD = "fraud"
    NON_FRAUD = "non_fraud"
    SUSPICIOUS = "suspicious"
    ILLICIT = "illicit"
    LICIT = "licit"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk level classification for samples."""

    CRITICAL = "critical"  # Known fraud/AML cases
    HIGH = "high"  # Strong indicators
    MEDIUM = "medium"  # Some risk signals
    LOW = "low"  # Minimal risk
    UNKNOWN = "unknown"


class LicenseType(str, Enum):
    """License types for data provenance tracking."""

    PUBLIC_DOMAIN = "public_domain"
    CC_BY = "cc_by_4_0"
    CC_BY_SA = "cc_by_sa_4_0"
    CC_BY_NC = "cc_by_nc_4_0"
    GOVERNMENT = "government"  # US gov public domain
    RESTRICTED = "restricted"  # Leaked/unclear license
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


# License mapping for known sources
SOURCE_LICENSES: Dict[DatasetSource, LicenseType] = {
    DatasetSource.SEC_EDGAR: LicenseType.PUBLIC_DOMAIN,
    DatasetSource.FATF_TYPOLOGIES: LicenseType.GOVERNMENT,
    DatasetSource.OPENSANCTIONS: LicenseType.CC_BY,
    DatasetSource.FINCEN_ADVISORIES: LicenseType.GOVERNMENT,
    DatasetSource.DOJ_ENFORCEMENT: LicenseType.GOVERNMENT,
    DatasetSource.SEC_ENFORCEMENT: LicenseType.GOVERNMENT,
    DatasetSource.GDELT: LicenseType.CC_BY,
    DatasetSource.FINCEN_FILES: LicenseType.RESTRICTED,
    DatasetSource.VENMO_NOTES: LicenseType.RESTRICTED,
    DatasetSource.ICIJ_OFFSHORE: LicenseType.RESTRICTED,
    DatasetSource.FRAUDNLP: LicenseType.CC_BY,
}


@dataclass
class Provenance:
    """Provenance tracking for data governance and audit trails.

    Tracks where data came from, its license, and when it was accessed.
    Essential for compliance audits and reproducibility.
    """

    # Source URL or identifier
    url: Optional[str] = None

    # License information
    license: LicenseType = LicenseType.UNKNOWN

    # When the data was accessed/downloaded
    access_date: Optional[str] = None  # ISO format: YYYY-MM-DD

    # Hash of original content (for verification)
    content_hash: Optional[str] = None  # Format: "sha256:abc123..."

    # Original filename if applicable
    original_filename: Optional[str] = None

    # Dataset version or release date
    dataset_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "license": self.license.value if self.license else None,
            "access_date": self.access_date,
            "content_hash": self.content_hash,
            "original_filename": self.original_filename,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Provenance":
        """Create from dictionary."""
        data = data.copy()
        if data.get("license"):
            data["license"] = LicenseType(data["license"])
        return cls(**data)

    @classmethod
    def for_source(
        cls,
        source: DatasetSource,
        url: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> "Provenance":
        """Create provenance for a known source with default license."""
        return cls(
            url=url,
            license=SOURCE_LICENSES.get(source, LicenseType.UNKNOWN),
            access_date=datetime.now().strftime("%Y-%m-%d"),
            content_hash=content_hash,
        )


@dataclass
class ProcessingInfo:
    """Tracks processing steps applied to a sample.

    Used to prevent double-processing and for audit trails.
    """

    # Deduplication hash (for tracking duplicates across runs)
    dedup_hash: Optional[str] = None  # Format: "minhash:..." or "sha256:..."

    # PII scrubbing status
    pii_scrubbed: bool = False
    pii_scrub_method: Optional[str] = None  # e.g., "presidio", "regex", "spacy"

    # Boilerplate removal status
    boilerplate_removed: bool = False
    boilerplate_method: Optional[str] = None  # e.g., "sec_edgar", "news", "general"

    # Text normalization
    normalized: bool = False
    normalization_steps: List[str] = field(default_factory=list)

    # Quality filtering
    quality_score: Optional[float] = None  # 0.0 to 1.0
    passed_quality_filter: bool = True

    # Processing timestamp
    processed_at: Optional[str] = None  # ISO format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dedup_hash": self.dedup_hash,
            "pii_scrubbed": self.pii_scrubbed,
            "pii_scrub_method": self.pii_scrub_method,
            "boilerplate_removed": self.boilerplate_removed,
            "boilerplate_method": self.boilerplate_method,
            "normalized": self.normalized,
            "normalization_steps": self.normalization_steps,
            "quality_score": self.quality_score,
            "passed_quality_filter": self.passed_quality_filter,
            "processed_at": self.processed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingInfo":
        """Create from dictionary."""
        return cls(**data)

    def mark_processed(self) -> None:
        """Update processed timestamp."""
        self.processed_at = datetime.now().isoformat()


@dataclass
class EnhancedLabels:
    """Rich labeling information for supervised training.

    Extends basic fraud labels with typology and entity information.
    """

    # Primary fraud/AML label
    primary_label: FraudLabel = FraudLabel.UNKNOWN

    # Risk level assessment
    risk_level: RiskLevel = RiskLevel.UNKNOWN

    # FATF typology codes (e.g., "shell_company", "structuring")
    typology_codes: List[str] = field(default_factory=list)

    # Red flag indicators detected
    red_flags: List[str] = field(default_factory=list)

    # Named entities extracted
    entities_mentioned: List[Dict[str, str]] = field(default_factory=list)
    # Format: [{"name": "...", "type": "person|org|location", "risk": "high|medium|low"}]

    # Confidence in labels (if from model prediction)
    label_confidence: Optional[float] = None

    # Source of labels (manual, model, rule-based)
    label_source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_label": self.primary_label.value,
            "risk_level": self.risk_level.value,
            "typology_codes": self.typology_codes,
            "red_flags": self.red_flags,
            "entities_mentioned": self.entities_mentioned,
            "label_confidence": self.label_confidence,
            "label_source": self.label_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedLabels":
        """Create from dictionary."""
        data = data.copy()
        if data.get("primary_label"):
            data["primary_label"] = FraudLabel(data["primary_label"])
        if data.get("risk_level"):
            data["risk_level"] = RiskLevel(data["risk_level"])
        return cls(**data)


@dataclass
class TextSample:
    """A single text sample with metadata for training.

    This is the unified format that all dataset loaders produce.
    Designed to support both pretraining (MLM) and supervised tasks.

    Enhanced with provenance tracking, processing info, and rich labels
    for data governance and audit compliance.
    """

    # Core text content
    text: str

    # Source identification
    source: DatasetSource
    sample_id: str

    # Classification metadata
    category: TextCategory = TextCategory.UNKNOWN
    label: FraudLabel = FraudLabel.UNKNOWN

    # Optional structured fields (legacy, kept for backward compatibility)
    entities: List[str] = field(default_factory=list)
    typologies: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)

    # Temporal information
    timestamp: Optional[datetime] = None
    time_period: Optional[str] = None

    # Additional metadata (flexible key-value store)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing state
    token_count: Optional[int] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None

    # === NEW: Enhanced metadata schema ===

    # Data provenance for governance and audit trails
    provenance: Optional[Provenance] = None

    # Processing pipeline tracking
    processing: Optional[ProcessingInfo] = None

    # Enhanced labeling with typologies and risk levels
    enhanced_labels: Optional[EnhancedLabels] = None

    def __post_init__(self) -> None:
        """Compute sample ID hash if not provided."""
        if not self.sample_id:
            content_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]
            self.sample_id = f"{self.source.value}_{content_hash}"

        # Initialize provenance with source license if not provided
        if self.provenance is None:
            self.provenance = Provenance.for_source(self.source)

        # Initialize processing info if not provided
        if self.processing is None:
            self.processing = ProcessingInfo()

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Character count."""
        return len(self.text)

    @property
    def content_hash(self) -> str:
        """SHA256 hash of the text content."""
        return f"sha256:{hashlib.sha256(self.text.encode()).hexdigest()}"

    @property
    def risk_level(self) -> RiskLevel:
        """Get risk level from enhanced labels or infer from label."""
        if self.enhanced_labels and self.enhanced_labels.risk_level != RiskLevel.UNKNOWN:
            return self.enhanced_labels.risk_level
        # Infer from primary label
        if self.label in (FraudLabel.FRAUD, FraudLabel.ILLICIT):
            return RiskLevel.HIGH
        elif self.label == FraudLabel.SUSPICIOUS:
            return RiskLevel.MEDIUM
        elif self.label in (FraudLabel.NON_FRAUD, FraudLabel.LICIT):
            return RiskLevel.LOW
        return RiskLevel.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "text": self.text,
            "source": self.source.value,
            "sample_id": self.sample_id,
            "category": self.category.value,
            "label": self.label.value,
            "entities": self.entities,
            "typologies": self.typologies,
            "indicators": self.indicators,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "time_period": self.time_period,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            # Enhanced metadata
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "processing": self.processing.to_dict() if self.processing else None,
            "enhanced_labels": self.enhanced_labels.to_dict() if self.enhanced_labels else None,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextSample":
        """Create from dictionary."""
        data = data.copy()
        data["source"] = DatasetSource(data["source"])
        data["category"] = TextCategory(data["category"])
        data["label"] = FraudLabel(data["label"])
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        # Parse enhanced metadata
        if data.get("provenance"):
            data["provenance"] = Provenance.from_dict(data["provenance"])
        if data.get("processing"):
            data["processing"] = ProcessingInfo.from_dict(data["processing"])
        if data.get("enhanced_labels"):
            data["enhanced_labels"] = EnhancedLabels.from_dict(data["enhanced_labels"])
        return cls(**data)

    def to_training_format(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert to format suitable for HuggingFace datasets."""
        result = {
            "text": self.text,
            "label": self.label.value,
            "category": self.category.value,
            "source": self.source.value,
        }
        if include_metadata:
            result["entities"] = self.entities
            result["typologies"] = self.typologies
            result["sample_id"] = self.sample_id
            result["risk_level"] = self.risk_level.value
            if self.enhanced_labels:
                result["typology_codes"] = self.enhanced_labels.typology_codes
                result["red_flags"] = self.enhanced_labels.red_flags
        return result

    def to_jsonl_v2(self) -> Dict[str, Any]:
        """Export to enhanced JSONL format for corpus files.

        This format is optimized for downstream processing and includes
        full provenance and processing information.
        """
        return {
            "id": self.sample_id,
            "text": self.text,
            "source": self.source.value,
            "category": self.category.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "processing": self.processing.to_dict() if self.processing else None,
            "labels": {
                "primary": self.label.value,
                "risk_level": self.risk_level.value,
                "typologies": self.typologies or (
                    self.enhanced_labels.typology_codes if self.enhanced_labels else []
                ),
                "indicators": self.indicators or (
                    self.enhanced_labels.red_flags if self.enhanced_labels else []
                ),
            },
            "entities": self.entities or (
                [e.get("name", "") for e in self.enhanced_labels.entities_mentioned]
                if self.enhanced_labels else []
            ),
            "metadata": {
                "token_count": self.token_count,
                "word_count": self.word_count,
                "char_count": self.char_count,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                **self.metadata,
            },
        }

    def mark_pii_scrubbed(self, method: str = "unknown") -> "TextSample":
        """Mark this sample as having PII scrubbed.

        Returns self for method chaining.
        """
        if self.processing is None:
            self.processing = ProcessingInfo()
        self.processing.pii_scrubbed = True
        self.processing.pii_scrub_method = method
        self.processing.mark_processed()
        return self

    def mark_boilerplate_removed(self, method: str = "unknown") -> "TextSample":
        """Mark this sample as having boilerplate removed.

        Returns self for method chaining.
        """
        if self.processing is None:
            self.processing = ProcessingInfo()
        self.processing.boilerplate_removed = True
        self.processing.boilerplate_method = method
        self.processing.mark_processed()
        return self

    def set_dedup_hash(self, dedup_hash: str) -> "TextSample":
        """Set the deduplication hash.

        Returns self for method chaining.
        """
        if self.processing is None:
            self.processing = ProcessingInfo()
        self.processing.dedup_hash = dedup_hash
        return self


@dataclass
class CorpusStats:
    """Statistics about a corpus or dataset."""

    total_samples: int = 0
    total_tokens: int = 0
    total_words: int = 0
    total_chars: int = 0

    samples_by_source: Dict[str, int] = field(default_factory=dict)
    samples_by_category: Dict[str, int] = field(default_factory=dict)
    samples_by_label: Dict[str, int] = field(default_factory=dict)

    avg_tokens_per_sample: float = 0.0
    avg_words_per_sample: float = 0.0

    unique_entities: int = 0
    unique_typologies: int = 0

    def update(self, sample: TextSample) -> None:
        """Update stats with a new sample."""
        self.total_samples += 1
        self.total_words += sample.word_count
        self.total_chars += sample.char_count
        if sample.token_count:
            self.total_tokens += sample.token_count

        source_key = sample.source.value
        self.samples_by_source[source_key] = self.samples_by_source.get(source_key, 0) + 1

        cat_key = sample.category.value
        self.samples_by_category[cat_key] = self.samples_by_category.get(cat_key, 0) + 1

        label_key = sample.label.value
        self.samples_by_label[label_key] = self.samples_by_label.get(label_key, 0) + 1

    def finalize(self) -> None:
        """Compute derived statistics."""
        if self.total_samples > 0:
            self.avg_words_per_sample = self.total_words / self.total_samples
            if self.total_tokens > 0:
                self.avg_tokens_per_sample = self.total_tokens / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Corpus Statistics:",
            f"  Total samples: {self.total_samples:,}",
            f"  Total words: {self.total_words:,}",
            f"  Total chars: {self.total_chars:,}",
            f"  Avg words/sample: {self.avg_words_per_sample:.1f}",
        ]
        if self.samples_by_source:
            lines.append("  By source:")
            for src, count in sorted(self.samples_by_source.items()):
                lines.append(f"    {src}: {count:,}")
        if self.samples_by_label:
            lines.append("  By label:")
            for label, count in sorted(self.samples_by_label.items()):
                lines.append(f"    {label}: {count:,}")
        return "\n".join(lines)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    # Data paths
    data_dir: Path
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Processing options
    max_samples: Optional[int] = None
    min_text_length: int = 10
    max_text_length: int = 50000

    # Chunking for long documents
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64  # tokens

    # Filtering
    required_categories: Optional[List[TextCategory]] = None
    required_labels: Optional[List[FraudLabel]] = None
    exclude_categories: Optional[List[TextCategory]] = None

    # Sampling
    balance_labels: bool = False
    random_seed: int = 42

    # Download options
    auto_download: bool = True
    verify_checksums: bool = True

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def default(cls, data_dir: Union[str, Path]) -> "DatasetConfig":
        """Create default configuration."""
        data_path = Path(data_dir)
        return cls(
            data_dir=data_path,
            cache_dir=data_path / ".cache",
            output_dir=data_path / "processed",
        )


class TextChunker:
    """Utility for chunking long documents into training-sized pieces."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens (or words if no tokenizer).
            chunk_overlap: Overlap between consecutive chunks.
            tokenizer: Optional HuggingFace tokenizer for accurate token counts.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks.
        """
        if self.tokenizer:
            return self._chunk_by_tokens(text)
        return self._chunk_by_words(text)

    def _chunk_by_words(self, text: str) -> List[str]:
        """Chunk by word count (approximate)."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - self.chunk_overlap
            if start >= len(words) - self.chunk_overlap:
                break

        return chunks

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk by actual token count using tokenizer."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - self.chunk_overlap
            if start >= len(tokens) - self.chunk_overlap:
                break

        return chunks

    def chunk_sample(self, sample: TextSample) -> List[TextSample]:
        """Chunk a TextSample into multiple samples.

        Args:
            sample: Input sample to chunk.

        Returns:
            List of chunked samples with updated metadata.
        """
        chunks = self.chunk_text(sample.text)

        if len(chunks) == 1:
            return [sample]

        chunked_samples = []
        for i, chunk_text in enumerate(chunks):
            chunked = TextSample(
                text=chunk_text,
                source=sample.source,
                sample_id=f"{sample.sample_id}_chunk{i}",
                category=sample.category,
                label=sample.label,
                entities=sample.entities,
                typologies=sample.typologies,
                indicators=sample.indicators,
                timestamp=sample.timestamp,
                time_period=sample.time_period,
                metadata={**sample.metadata, "original_id": sample.sample_id},
                chunk_index=i,
                total_chunks=len(chunks),
            )
            chunked_samples.append(chunked)

        return chunked_samples


class BaseTextDataset(ABC):
    """Abstract base class for all text dataset loaders.

    Subclasses must implement:
    - download(): Fetch raw data if not present
    - load(): Parse raw data into TextSample objects
    - __iter__(): Iterate over samples
    """

    source: DatasetSource = DatasetSource.CUSTOM

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize dataset loader.

        Args:
            config: Dataset configuration.
        """
        self.config = config
        self._samples: Optional[List[TextSample]] = None
        self._stats: Optional[CorpusStats] = None
        self._chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    @property
    def name(self) -> str:
        """Dataset name."""
        return self.source.value

    @property
    def stats(self) -> CorpusStats:
        """Get corpus statistics (loads data if needed)."""
        if self._stats is None:
            self._compute_stats()
        return self._stats

    @abstractmethod
    def download(self) -> bool:
        """Download raw data if not present.

        Returns:
            True if download successful or data exists.
        """
        pass

    @abstractmethod
    def load(self) -> List[TextSample]:
        """Load and parse the dataset.

        Returns:
            List of TextSample objects.
        """
        pass

    def get_samples(self, reload: bool = False) -> List[TextSample]:
        """Get all samples, loading if needed.

        Args:
            reload: Force reload from source.

        Returns:
            List of TextSample objects.
        """
        if self._samples is None or reload:
            self._samples = self.load()
            self._apply_filters()
            self._stats = None  # Reset stats
        return self._samples

    def _apply_filters(self) -> None:
        """Apply configured filters to loaded samples."""
        if self._samples is None:
            return

        original_count = len(self._samples)

        # Filter by text length
        self._samples = [
            s for s in self._samples
            if self.config.min_text_length <= len(s.text) <= self.config.max_text_length
        ]

        # Filter by category
        if self.config.required_categories:
            self._samples = [
                s for s in self._samples
                if s.category in self.config.required_categories
            ]

        if self.config.exclude_categories:
            self._samples = [
                s for s in self._samples
                if s.category not in self.config.exclude_categories
            ]

        # Filter by label
        if self.config.required_labels:
            self._samples = [
                s for s in self._samples
                if s.label in self.config.required_labels
            ]

        # Limit samples
        if self.config.max_samples and len(self._samples) > self.config.max_samples:
            import random
            random.seed(self.config.random_seed)
            self._samples = random.sample(self._samples, self.config.max_samples)

        filtered_count = len(self._samples)
        if filtered_count < original_count:
            logger.info(
                f"Filtered {self.name}: {original_count} -> {filtered_count} samples"
            )

    def _compute_stats(self) -> None:
        """Compute statistics over loaded samples."""
        self._stats = CorpusStats()
        for sample in self.get_samples():
            self._stats.update(sample)
        self._stats.finalize()

    def iter_chunks(self) -> Generator[TextSample, None, None]:
        """Iterate over chunked samples for long documents.

        Yields:
            TextSample objects, with long documents split into chunks.
        """
        for sample in self.get_samples():
            for chunk in self._chunker.chunk_sample(sample):
                yield chunk

    def __iter__(self) -> Iterator[TextSample]:
        """Iterate over samples."""
        return iter(self.get_samples())

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.get_samples())

    def __getitem__(self, idx: int) -> TextSample:
        """Get sample by index."""
        return self.get_samples()[idx]

    def to_jsonl(self, output_path: Path) -> int:
        """Export dataset to JSONL format.

        Args:
            output_path: Output file path.

        Returns:
            Number of samples written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self.get_samples():
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        logger.info(f"Exported {count} samples to {output_path}")
        return count

    def to_hf_dataset(self, include_metadata: bool = False) -> Any:
        """Convert to HuggingFace Dataset.

        Args:
            include_metadata: Include full metadata in dataset.

        Returns:
            HuggingFace Dataset object.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        data = [s.to_training_format(include_metadata) for s in self.get_samples()]
        return Dataset.from_list(data)


class DatasetRegistry:
    """Registry for available dataset loaders."""

    _loaders: Dict[DatasetSource, type] = {}

    @classmethod
    def register(cls, source: DatasetSource):
        """Decorator to register a dataset loader."""
        def decorator(loader_cls: type) -> type:
            cls._loaders[source] = loader_cls
            return loader_cls
        return decorator

    @classmethod
    def get_loader(cls, source: DatasetSource) -> type:
        """Get loader class for a source."""
        if source not in cls._loaders:
            raise ValueError(f"No loader registered for {source}")
        return cls._loaders[source]

    @classmethod
    def available_sources(cls) -> List[DatasetSource]:
        """List available dataset sources."""
        return list(cls._loaders.keys())

    @classmethod
    def create_loader(
        cls,
        source: DatasetSource,
        config: DatasetConfig,
    ) -> BaseTextDataset:
        """Create a loader instance."""
        loader_cls = cls.get_loader(source)
        return loader_cls(config)
