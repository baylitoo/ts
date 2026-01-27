"""
Data preparation module for text-rich AML datasets.

This module provides loaders and processors for various text-rich datasets
that can be used to train an LLM-based judge for AML detection:

Datasets supported:
- FinCEN Files: SAR narratives from ICIJ leak
- SEC/EDGAR: Financial fraud filings with MD&A text
- FraudNLP: Action sequences as NLP classification
- FATF Typologies: Case studies and red flag indicators
- OpenSanctions: Entity descriptions and sanctions data
- Venmo Notes: Payment memo text
- GDELT: News articles for adverse media

Usage:
    from rl_money_laundering.dataprep import (
        FinCENLoader,
        SECFraudLoader,
        FraudNLPLoader,
        FATFTypologyLoader,
        OpenSanctionsLoader,
        VenmoNotesLoader,
        GDELTLoader,
        UnifiedCorpusBuilder,
    )
"""

from .base import (
    BaseTextDataset,
    ComplianceMode,
    DatasetConfig,
    DatasetSource,
    CorpusStats,
    TextSample,
    RESTRICTED_SOURCES,
    PUBLIC_SOURCES,
    get_allowed_sources,
    # Enhanced metadata schema
    RiskLevel,
    LicenseType,
    SOURCE_LICENSES,
    Provenance,
    ProcessingInfo,
    EnhancedLabels,
)
from .fincen_loader import FinCENLoader
from .sec_fraud_loader import SECFraudLoader
from .fraudnlp_loader import FraudNLPLoader
from .fatf_loader import FATFTypologyLoader
from .opensanctions_loader import OpenSanctionsLoader
from .venmo_loader import VenmoNotesLoader
from .gdelt_loader import GDELTLoader
from .corpus_builder import UnifiedCorpusBuilder, MixingConfig
from .dedup import (
    DedupConfig,
    DedupPipeline,
    DedupStats,
    ExactDedup,
    MinHashDedup,
    deduplicate_corpus,
)
from .cleaning import (
    BoilerplateStripper,
    CleaningConfig,
    TextNormalizer,
    clean_corpus,
)
from .pii_scrubber import (
    PIIType,
    RedactionStyle,
    PIIMatch,
    ScrubConfig,
    ScrubStats,
    RegexScrubber,
    PresidioScrubber,
    SpacyScrubber,
    PIIScrubPipeline,
    scrub_pii,
)
from .pipeline import (
    DataPrepPipeline,
    PipelineConfig,
    PipelineResult,
    run_default_pipeline,
    quick_test_pipeline,
)

__all__ = [
    # Base classes
    "BaseTextDataset",
    "ComplianceMode",
    "DatasetConfig",
    "DatasetSource",
    "CorpusStats",
    "TextSample",
    "RESTRICTED_SOURCES",
    "PUBLIC_SOURCES",
    "get_allowed_sources",
    # Enhanced metadata schema
    "RiskLevel",
    "LicenseType",
    "SOURCE_LICENSES",
    "Provenance",
    "ProcessingInfo",
    "EnhancedLabels",
    # Loaders
    "FinCENLoader",
    "SECFraudLoader",
    "FraudNLPLoader",
    "FATFTypologyLoader",
    "OpenSanctionsLoader",
    "VenmoNotesLoader",
    "GDELTLoader",
    # Builders
    "UnifiedCorpusBuilder",
    "MixingConfig",
    # Deduplication
    "DedupConfig",
    "DedupPipeline",
    "DedupStats",
    "ExactDedup",
    "MinHashDedup",
    "deduplicate_corpus",
    # Cleaning
    "BoilerplateStripper",
    "CleaningConfig",
    "TextNormalizer",
    "clean_corpus",
    # PII Scrubbing
    "PIIType",
    "RedactionStyle",
    "PIIMatch",
    "ScrubConfig",
    "ScrubStats",
    "RegexScrubber",
    "PresidioScrubber",
    "SpacyScrubber",
    "PIIScrubPipeline",
    "scrub_pii",
    # Pipeline
    "DataPrepPipeline",
    "PipelineConfig",
    "PipelineResult",
    "run_default_pipeline",
    "quick_test_pipeline",
]
