"""
PII (Personally Identifiable Information) scrubbing module.

Provides configurable PII detection and redaction for AML corpus data:
- RegexScrubber: Pattern-based detection for common PII types
- PresidioScrubber: Microsoft Presidio integration for production use
- SpacyScrubber: spaCy NER-based detection
- PIIScrubPipeline: Orchestrates multiple scrubbers

PII types handled:
- Names (person, organization)
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- Bank account numbers (IBAN, routing)
- IP addresses
- Physical addresses
- Dates of birth
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
)

from .base import TextSample, DatasetSource

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected and scrubbed."""

    # Identity
    PERSON_NAME = "person_name"
    ORGANIZATION = "organization"

    # Contact
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"

    # Financial
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"
    ROUTING_NUMBER = "routing_number"

    # Technical
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"

    # Dates
    DATE_OF_BIRTH = "date_of_birth"
    DATE = "date"

    # Other
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    TAX_ID = "tax_id"


class RedactionStyle(str, Enum):
    """How to redact detected PII."""

    MASK = "mask"  # Replace with [PII_TYPE]
    HASH = "hash"  # Replace with hash of original
    PLACEHOLDER = "placeholder"  # Replace with generic placeholder
    REMOVE = "remove"  # Remove entirely


# Common regex patterns for PII detection
PII_PATTERNS: Dict[PIIType, List[Pattern]] = {
    PIIType.EMAIL: [
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
    ],
    PIIType.PHONE: [
        # US phone formats
        re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        # International with country code
        re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
    ],
    PIIType.SSN: [
        re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    ],
    PIIType.CREDIT_CARD: [
        # Visa, Mastercard, Amex, Discover
        re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
        # With separators
        re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    ],
    PIIType.IBAN: [
        re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b'),
    ],
    PIIType.ROUTING_NUMBER: [
        # US ABA routing number (9 digits)
        re.compile(r'\b\d{9}\b'),
    ],
    PIIType.BANK_ACCOUNT: [
        # Generic account numbers (8-17 digits)
        re.compile(r'\b\d{8,17}\b'),
    ],
    PIIType.IP_ADDRESS: [
        # IPv4
        re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
        # IPv6 (simplified)
        re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
    ],
    PIIType.MAC_ADDRESS: [
        re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'),
    ],
    PIIType.DATE_OF_BIRTH: [
        # MM/DD/YYYY or DD/MM/YYYY
        re.compile(r'\b(?:0[1-9]|1[0-2])[/.-](?:0[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}\b'),
        # YYYY-MM-DD
        re.compile(r'\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b'),
    ],
    PIIType.PASSPORT: [
        # Generic passport format
        re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
    ],
}

# Patterns that need context to avoid false positives
CONTEXT_PATTERNS: Dict[PIIType, List[Tuple[Pattern, Pattern]]] = {
    # (context_pattern, value_pattern)
    PIIType.SSN: [
        (re.compile(r'(?:ssn|social\s*security)', re.IGNORECASE), PII_PATTERNS[PIIType.SSN][0]),
    ],
    PIIType.DATE_OF_BIRTH: [
        (re.compile(r'(?:dob|date\s*of\s*birth|born|birthday)', re.IGNORECASE), PII_PATTERNS[PIIType.DATE_OF_BIRTH][0]),
    ],
}


@dataclass
class PIIMatch:
    """A detected PII instance."""

    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float = 1.0
    context: Optional[str] = None

    def __repr__(self) -> str:
        return f"PIIMatch({self.pii_type.value}, '{self.text[:20]}...', conf={self.confidence:.2f})"


@dataclass
class ScrubStats:
    """Statistics from PII scrubbing."""

    total_samples: int = 0
    samples_with_pii: int = 0
    total_pii_found: int = 0
    pii_by_type: Dict[str, int] = field(default_factory=dict)
    samples_by_source: Dict[str, int] = field(default_factory=dict)

    def update(self, matches: List[PIIMatch], source: DatasetSource) -> None:
        """Update stats with detected PII."""
        self.total_samples += 1
        if matches:
            self.samples_with_pii += 1
            self.total_pii_found += len(matches)
            for match in matches:
                key = match.pii_type.value
                self.pii_by_type[key] = self.pii_by_type.get(key, 0) + 1

        source_key = source.value
        self.samples_by_source[source_key] = self.samples_by_source.get(source_key, 0) + 1

    def __str__(self) -> str:
        lines = [
            "PII Scrubbing Statistics:",
            f"  Total samples: {self.total_samples:,}",
            f"  Samples with PII: {self.samples_with_pii:,} ({self.samples_with_pii/max(1, self.total_samples):.1%})",
            f"  Total PII found: {self.total_pii_found:,}",
        ]
        if self.pii_by_type:
            lines.append("  PII by type:")
            for pii_type, count in sorted(self.pii_by_type.items(), key=lambda x: -x[1]):
                lines.append(f"    {pii_type}: {count:,}")
        return "\n".join(lines)


@dataclass
class ScrubConfig:
    """Configuration for PII scrubbing."""

    # Which PII types to detect
    enabled_types: Set[PIIType] = field(default_factory=lambda: set(PIIType))

    # Redaction style
    redaction_style: RedactionStyle = RedactionStyle.MASK

    # Minimum confidence threshold
    min_confidence: float = 0.7

    # Whether to use context patterns for higher accuracy
    use_context_patterns: bool = True

    # Whether to keep a log of detected PII (for audit)
    audit_log: bool = False

    # Custom redaction strings per type
    custom_redactions: Dict[PIIType, str] = field(default_factory=dict)

    # Sources to skip (e.g., already sanitized)
    skip_sources: Set[DatasetSource] = field(default_factory=set)

    # Verbose logging
    verbose: bool = True

    def get_redaction(self, pii_type: PIIType, original: str) -> str:
        """Get the redaction string for a PII type."""
        if pii_type in self.custom_redactions:
            return self.custom_redactions[pii_type]

        if self.redaction_style == RedactionStyle.MASK:
            return f"[{pii_type.value.upper()}]"
        elif self.redaction_style == RedactionStyle.HASH:
            hash_val = hashlib.md5(original.encode()).hexdigest()[:8]
            return f"[{pii_type.value.upper()}_{hash_val}]"
        elif self.redaction_style == RedactionStyle.PLACEHOLDER:
            return "XXXXX"
        elif self.redaction_style == RedactionStyle.REMOVE:
            return ""
        return f"[{pii_type.value.upper()}]"


class BaseScrubber(ABC):
    """Abstract base class for PII scrubbers."""

    @abstractmethod
    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII in text.

        Args:
            text: Input text to scan.

        Returns:
            List of detected PII matches.
        """
        pass

    def scrub(self, text: str, config: ScrubConfig) -> Tuple[str, List[PIIMatch]]:
        """Detect and redact PII in text.

        Args:
            text: Input text to scrub.
            config: Scrubbing configuration.

        Returns:
            Tuple of (scrubbed_text, detected_matches).
        """
        matches = self.detect(text)

        # Filter by enabled types and confidence
        matches = [
            m for m in matches
            if m.pii_type in config.enabled_types and m.confidence >= config.min_confidence
        ]

        if not matches:
            return text, []

        # Sort by position (descending) to replace from end
        matches.sort(key=lambda m: m.start, reverse=True)

        # Apply redactions
        scrubbed = text
        for match in matches:
            redaction = config.get_redaction(match.pii_type, match.text)
            scrubbed = scrubbed[:match.start] + redaction + scrubbed[match.end:]

        return scrubbed, matches


class RegexScrubber(BaseScrubber):
    """Regex-based PII detection.

    Fast but may have false positives. Best for common patterns like
    emails, phone numbers, SSNs, credit cards.
    """

    def __init__(
        self,
        enabled_types: Optional[Set[PIIType]] = None,
        use_context: bool = True,
    ) -> None:
        """Initialize regex scrubber.

        Args:
            enabled_types: PII types to detect (None for all).
            use_context: Use context patterns for better accuracy.
        """
        self.enabled_types = enabled_types or set(PII_PATTERNS.keys())
        self.use_context = use_context

    def detect(self, text: str) -> List[PIIMatch]:
        matches = []

        for pii_type in self.enabled_types:
            if pii_type not in PII_PATTERNS:
                continue

            for pattern in PII_PATTERNS[pii_type]:
                for match in pattern.finditer(text):
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # Regex matches have moderate confidence
                    )
                    matches.append(pii_match)

        # Apply context patterns for higher confidence
        if self.use_context:
            for pii_type, context_patterns in CONTEXT_PATTERNS.items():
                if pii_type not in self.enabled_types:
                    continue

                for context_pattern, value_pattern in context_patterns:
                    # Find context mentions
                    for context_match in context_pattern.finditer(text):
                        # Search for value near context (within 100 chars)
                        search_start = context_match.end()
                        search_end = min(len(text), search_start + 100)
                        search_region = text[search_start:search_end]

                        for value_match in value_pattern.finditer(search_region):
                            actual_start = search_start + value_match.start()
                            actual_end = search_start + value_match.end()

                            pii_match = PIIMatch(
                                pii_type=pii_type,
                                text=value_match.group(),
                                start=actual_start,
                                end=actual_end,
                                confidence=0.95,  # Context-validated = higher confidence
                                context=context_match.group(),
                            )
                            matches.append(pii_match)

        # Deduplicate overlapping matches (keep highest confidence)
        matches = self._deduplicate_matches(matches)

        return matches

    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []

        # Sort by start position, then by confidence (descending)
        matches.sort(key=lambda m: (m.start, -m.confidence))

        deduped = []
        last_end = -1

        for match in matches:
            if match.start >= last_end:
                deduped.append(match)
                last_end = match.end
            elif match.confidence > deduped[-1].confidence:
                # Higher confidence match overlaps - replace
                deduped[-1] = match
                last_end = match.end

        return deduped


class PresidioScrubber(BaseScrubber):
    """Microsoft Presidio-based PII detection.

    More accurate than regex, uses NLP models. Requires presidio-analyzer.

    Install: pip install presidio-analyzer presidio-anonymizer
    """

    def __init__(
        self,
        language: str = "en",
        entities: Optional[List[str]] = None,
    ) -> None:
        """Initialize Presidio scrubber.

        Args:
            language: Text language code.
            entities: Presidio entity types to detect.
        """
        try:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
            self._available = True
        except ImportError:
            logger.warning(
                "Presidio not available. Install with: "
                "pip install presidio-analyzer presidio-anonymizer"
            )
            self._available = False
            self._analyzer = None

        self.language = language
        self.entities = entities or [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "IBAN_CODE", "US_SSN", "US_BANK_NUMBER", "IP_ADDRESS",
            "LOCATION", "DATE_TIME", "US_PASSPORT", "US_DRIVER_LICENSE",
        ]

        # Map Presidio entity types to our PIIType
        self._type_map = {
            "PERSON": PIIType.PERSON_NAME,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "IBAN_CODE": PIIType.IBAN,
            "US_SSN": PIIType.SSN,
            "US_BANK_NUMBER": PIIType.BANK_ACCOUNT,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "LOCATION": PIIType.ADDRESS,
            "DATE_TIME": PIIType.DATE,
            "US_PASSPORT": PIIType.PASSPORT,
            "US_DRIVER_LICENSE": PIIType.DRIVERS_LICENSE,
            "ORGANIZATION": PIIType.ORGANIZATION,
        }

    def detect(self, text: str) -> List[PIIMatch]:
        if not self._available or self._analyzer is None:
            logger.warning("Presidio not available, returning empty matches")
            return []

        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
        )

        matches = []
        for result in results:
            pii_type = self._type_map.get(result.entity_type)
            if pii_type is None:
                continue

            match = PIIMatch(
                pii_type=pii_type,
                text=text[result.start:result.end],
                start=result.start,
                end=result.end,
                confidence=result.score,
            )
            matches.append(match)

        return matches


class SpacyScrubber(BaseScrubber):
    """spaCy NER-based PII detection.

    Uses spaCy's named entity recognition for person/org/location.
    Best for name detection, combine with RegexScrubber for other types.

    Install: pip install spacy && python -m spacy download en_core_web_sm
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        entity_types: Optional[List[str]] = None,
    ) -> None:
        """Initialize spaCy scrubber.

        Args:
            model_name: spaCy model to use.
            entity_types: NER entity types to detect.
        """
        try:
            import spacy
            self._nlp = spacy.load(model_name)
            self._available = True
        except (ImportError, OSError) as e:
            logger.warning(
                f"spaCy not available ({e}). Install with: "
                f"pip install spacy && python -m spacy download {model_name}"
            )
            self._available = False
            self._nlp = None

        self.entity_types = entity_types or ["PERSON", "ORG", "GPE", "LOC"]

        # Map spaCy entity types to our PIIType
        self._type_map = {
            "PERSON": PIIType.PERSON_NAME,
            "ORG": PIIType.ORGANIZATION,
            "GPE": PIIType.ADDRESS,
            "LOC": PIIType.ADDRESS,
            "DATE": PIIType.DATE,
        }

    def detect(self, text: str) -> List[PIIMatch]:
        if not self._available or self._nlp is None:
            return []

        doc = self._nlp(text)
        matches = []

        for ent in doc.ents:
            if ent.label_ not in self.entity_types:
                continue

            pii_type = self._type_map.get(ent.label_)
            if pii_type is None:
                continue

            match = PIIMatch(
                pii_type=pii_type,
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.85,  # spaCy NER has good accuracy
            )
            matches.append(match)

        return matches


class PIIScrubPipeline:
    """Full PII scrubbing pipeline combining multiple scrubbers.

    Orchestrates regex, Presidio, and spaCy scrubbers with configurable
    fallbacks and combination strategies.
    """

    def __init__(
        self,
        config: Optional[ScrubConfig] = None,
        use_presidio: bool = False,
        use_spacy: bool = False,
    ) -> None:
        """Initialize PII scrubbing pipeline.

        Args:
            config: Scrubbing configuration.
            use_presidio: Enable Presidio scrubber (requires install).
            use_spacy: Enable spaCy scrubber (requires install).
        """
        self.config = config or ScrubConfig()
        self.stats = ScrubStats()

        # Initialize scrubbers
        self._scrubbers: List[BaseScrubber] = []

        # Always include regex scrubber as baseline
        regex_types = {
            PIIType.EMAIL, PIIType.PHONE, PIIType.SSN, PIIType.CREDIT_CARD,
            PIIType.IBAN, PIIType.IP_ADDRESS, PIIType.MAC_ADDRESS,
            PIIType.DATE_OF_BIRTH, PIIType.PASSPORT,
        }
        self._scrubbers.append(RegexScrubber(
            enabled_types=regex_types & self.config.enabled_types,
            use_context=self.config.use_context_patterns,
        ))

        # Add Presidio if requested and available
        if use_presidio:
            presidio = PresidioScrubber()
            if presidio._available:
                self._scrubbers.append(presidio)
            else:
                logger.info("Presidio requested but not available, using regex only")

        # Add spaCy if requested and available
        if use_spacy:
            spacy_scrubber = SpacyScrubber()
            if spacy_scrubber._available:
                self._scrubbers.append(spacy_scrubber)
            else:
                logger.info("spaCy requested but not available, using regex only")

        logger.info(f"PII pipeline initialized with {len(self._scrubbers)} scrubber(s)")

    def scrub_text(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """Scrub PII from text.

        Args:
            text: Input text to scrub.

        Returns:
            Tuple of (scrubbed_text, all_matches).
        """
        all_matches: List[PIIMatch] = []

        # Collect matches from all scrubbers
        for scrubber in self._scrubbers:
            matches = scrubber.detect(text)
            all_matches.extend(matches)

        # Deduplicate and merge overlapping matches
        all_matches = self._merge_matches(all_matches)

        # Filter by config
        all_matches = [
            m for m in all_matches
            if m.pii_type in self.config.enabled_types
            and m.confidence >= self.config.min_confidence
        ]

        if not all_matches:
            return text, []

        # Sort by position (descending) for replacement
        all_matches.sort(key=lambda m: m.start, reverse=True)

        # Apply redactions
        scrubbed = text
        for match in all_matches:
            redaction = self.config.get_redaction(match.pii_type, match.text)
            scrubbed = scrubbed[:match.start] + redaction + scrubbed[match.end:]

        return scrubbed, all_matches

    def _merge_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Merge overlapping matches, keeping highest confidence."""
        if not matches:
            return []

        # Sort by start position
        matches.sort(key=lambda m: m.start)

        merged = []
        current = matches[0]

        for match in matches[1:]:
            if match.start <= current.end:
                # Overlapping - keep higher confidence
                if match.confidence > current.confidence:
                    current = match
                elif match.end > current.end:
                    # Extend current match if same confidence
                    current = PIIMatch(
                        pii_type=current.pii_type,
                        text=current.text,
                        start=current.start,
                        end=match.end,
                        confidence=max(current.confidence, match.confidence),
                    )
            else:
                merged.append(current)
                current = match

        merged.append(current)
        return merged

    def scrub_sample(self, sample: TextSample) -> TextSample:
        """Scrub PII from a TextSample.

        Args:
            sample: Input sample to scrub.

        Returns:
            New TextSample with PII removed.
        """
        # Skip if source is in skip list
        if sample.source in self.config.skip_sources:
            return sample

        scrubbed_text, matches = self.scrub_text(sample.text)

        # Update stats
        self.stats.update(matches, sample.source)

        if not matches:
            return sample

        # Create new sample with scrubbed text
        new_sample = TextSample(
            text=scrubbed_text,
            source=sample.source,
            sample_id=sample.sample_id,
            category=sample.category,
            label=sample.label,
            entities=sample.entities,
            typologies=sample.typologies,
            indicators=sample.indicators,
            timestamp=sample.timestamp,
            time_period=sample.time_period,
            metadata={
                **sample.metadata,
                "pii_scrubbed": True,
                "pii_count": len(matches),
                "pii_types": list({m.pii_type.value for m in matches}),
            },
            token_count=sample.token_count,
            chunk_index=sample.chunk_index,
            total_chunks=sample.total_chunks,
            provenance=sample.provenance,
            processing=sample.processing,
            enhanced_labels=sample.enhanced_labels,
        )

        # Mark as PII scrubbed using the method
        new_sample.mark_pii_scrubbed(method="pipeline")

        return new_sample

    def process(
        self,
        samples: Iterable[TextSample],
    ) -> Generator[TextSample, None, None]:
        """Process samples through the PII scrubbing pipeline.

        Args:
            samples: Input samples to scrub.

        Yields:
            Scrubbed samples.
        """
        self.stats = ScrubStats()

        for sample in samples:
            yield self.scrub_sample(sample)

            # Periodic logging
            if self.config.verbose and self.stats.total_samples % 5000 == 0:
                logger.info(
                    f"PII scrub progress: {self.stats.total_samples:,} processed, "
                    f"{self.stats.samples_with_pii:,} with PII ({self.stats.samples_with_pii/max(1, self.stats.total_samples):.1%})"
                )

        if self.config.verbose:
            logger.info(str(self.stats))

    def get_stats(self) -> ScrubStats:
        """Get scrubbing statistics."""
        return self.stats


def scrub_pii(
    samples: Iterable[TextSample],
    redaction_style: RedactionStyle = RedactionStyle.MASK,
    use_presidio: bool = False,
    use_spacy: bool = False,
    min_confidence: float = 0.7,
) -> Generator[TextSample, None, None]:
    """Convenience function to scrub PII from a corpus.

    Args:
        samples: Input samples.
        redaction_style: How to redact detected PII.
        use_presidio: Enable Presidio (requires install).
        use_spacy: Enable spaCy (requires install).
        min_confidence: Minimum confidence threshold.

    Yields:
        Scrubbed samples.

    Example:
        ```python
        from rl_money_laundering.dataprep.pii_scrubber import scrub_pii

        # Simple scrubbing with defaults
        clean_samples = list(scrub_pii(raw_samples))

        # With Presidio for better accuracy
        clean_samples = list(scrub_pii(
            raw_samples,
            use_presidio=True,
            redaction_style=RedactionStyle.HASH,
        ))
        ```
    """
    config = ScrubConfig(
        redaction_style=redaction_style,
        min_confidence=min_confidence,
    )
    pipeline = PIIScrubPipeline(
        config=config,
        use_presidio=use_presidio,
        use_spacy=use_spacy,
    )
    yield from pipeline.process(samples)
