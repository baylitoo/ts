"""
Text cleaning and boilerplate removal for AML corpus.

Provides utilities to strip boilerplate, normalize text, and clean
source-specific noise from documents:
- SEC EDGAR: filing headers, HTML junk, exhibit references
- GDELT/News: newsletter CTAs, social media buttons, cookie consent
- General: email signatures, legal disclaimers, whitespace normalization
"""

from __future__ import annotations

import html
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
)

from .base import TextSample, DatasetSource

logger = logging.getLogger(__name__)


# =============================================================================
# Regex Patterns for Boilerplate Detection
# =============================================================================

# SEC EDGAR patterns
SEC_HEADER_PATTERNS = [
    r'ACCESSION\s+NUMBER[:\s]+[\d\-]+',
    r'CONFORMED\s+SUBMISSION\s+TYPE[:\s]+\S+',
    r'FILED\s+AS\s+OF\s+DATE[:\s]+\d+',
    r'DATE\s+AS\s+OF\s+CHANGE[:\s]+\d+',
    r'CENTRAL\s+INDEX\s+KEY[:\s]+\d+',
    r'STANDARD\s+INDUSTRIAL\s+CLASSIFICATION[:\s]+.+',
    r'IRS\s+NUMBER[:\s]+[\d\-]+',
    r'STATE\s+OF\s+INCORPORATION[:\s]+\w+',
    r'FISCAL\s+YEAR\s+END[:\s]+\d+',
    r'FILER[:\s]*$',
    r'COMPANY\s+CONFORMED\s+NAME[:\s]+.+',
    r'<SEC-DOCUMENT>.*?</SEC-DOCUMENT>',
    r'<SEC-HEADER>.*?</SEC-HEADER>',
    r'-----BEGIN\s+PRIVACY-ENHANCED\s+MESSAGE-----.*?-----END\s+PRIVACY-ENHANCED\s+MESSAGE-----',
]

SEC_EXHIBIT_PATTERNS = [
    r'EXHIBIT\s+\d+[\.\d]*',
    r'EX-\d+[\.\d]*',
    r'See\s+Exhibit\s+\d+',
    r'Incorporated\s+by\s+reference\s+to\s+Exhibit',
    r'\*+\s*Filed\s+herewith',
]

SEC_TOC_PATTERNS = [
    r'TABLE\s+OF\s+CONTENTS',
    r'INDEX\s+TO\s+FINANCIAL\s+STATEMENTS',
    r'\.{3,}\s*\d+\s*$',  # Dotted lines to page numbers
    r'Page\s+\d+\s+of\s+\d+',
]

# News/GDELT patterns
NEWS_CTA_PATTERNS = [
    r'Subscribe\s+to\s+(?:our\s+)?newsletter',
    r'Sign\s+up\s+for\s+(?:our\s+)?(?:daily|weekly|free)\s+(?:email|newsletter)',
    r'Get\s+the\s+latest\s+(?:news|updates)',
    r'Follow\s+us\s+on\s+(?:Twitter|Facebook|Instagram|LinkedIn)',
    r'Share\s+this\s+(?:article|story|post)',
    r'Like\s+us\s+on\s+Facebook',
    r'Join\s+our\s+(?:community|mailing\s+list)',
    r'Click\s+here\s+to\s+(?:subscribe|sign\s+up|learn\s+more)',
    r'Read\s+more\s+(?:articles|stories)\s+like\s+this',
    r'Recommended\s+for\s+you',
    r'You\s+may\s+also\s+like',
    r'Related\s+(?:articles|stories|posts)',
]

NEWS_COOKIE_PATTERNS = [
    r'(?:We\s+use\s+)?cookies\s+to\s+(?:improve|enhance)',
    r'By\s+(?:continuing|using)\s+(?:this\s+site|to\s+browse)',
    r'Accept\s+(?:all\s+)?cookies',
    r'Cookie\s+(?:policy|settings|preferences)',
    r'Privacy\s+(?:policy|notice|settings)',
    r'GDPR\s+(?:compliance|notice)',
    r'(?:Your|Our)\s+privacy\s+(?:matters|is\s+important)',
]

NEWS_SOCIAL_PATTERNS = [
    r'\[?\s*(?:Share|Tweet|Pin|Email|Print)\s*\]?',
    r'(?:Facebook|Twitter|LinkedIn|Pinterest|Reddit|WhatsApp)\s*(?:Share|Icon)?',
    r'\d+\s+(?:shares?|likes?|comments?|views?)',
    r'(?:Share|Tweet|Post)\s+this\s+(?:article|story)',
]

# General patterns
EMAIL_SIGNATURE_PATTERNS = [
    r'(?:Best|Kind|Warm)\s+(?:regards|wishes)',
    r'Sincerely(?:\s+yours)?',
    r'(?:Yours\s+)?(?:truly|faithfully)',
    r'Sent\s+from\s+my\s+(?:iPhone|iPad|Android|BlackBerry)',
    r'Get\s+Outlook\s+for\s+(?:iOS|Android)',
    r'This\s+email\s+and\s+any\s+attachments?\s+(?:are|is)\s+confidential',
    r'CONFIDENTIALITY\s+NOTICE',
    r'DISCLAIMER[:\s]',
    r'This\s+message\s+(?:is\s+)?intended\s+(?:only\s+)?for',
    r'If\s+you\s+(?:are\s+not|have\s+received)\s+(?:the\s+intended|this)',
]

LEGAL_DISCLAIMER_PATTERNS = [
    r'(?:All\s+)?rights?\s+reserved',
    r'Copyright\s*[©\(c\)]\s*\d{4}',
    r'Terms\s+(?:of\s+(?:service|use)|and\s+conditions)',
    r'(?:Privacy|Cookie)\s+policy',
    r'No\s+(?:part\s+of\s+this|reproduction)',
    r'(?:Written|Prior)\s+permission\s+(?:of\s+the\s+)?(?:author|publisher)',
    r'The\s+(?:views|opinions)\s+expressed\s+(?:herein|in\s+this)',
    r'(?:This|The)\s+(?:information|content)\s+(?:is\s+)?provided\s+(?:as\s+is|for\s+informational)',
    r'(?:We|The\s+company)\s+(?:make|makes)\s+no\s+(?:warranties?|representations?)',
]

# HTML/formatting patterns
HTML_PATTERNS = [
    r'<[^>]+>',  # HTML tags
    r'&[a-zA-Z]+;',  # Named HTML entities
    r'&\#\d+;',  # Numeric HTML entities
    r'<!\-\-.*?\-\->',  # HTML comments
    r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
    r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>',
]

# Whitespace patterns
WHITESPACE_PATTERNS = [
    (r'\n{3,}', '\n\n'),  # Multiple newlines -> double
    (r' {2,}', ' '),  # Multiple spaces -> single
    (r'\t+', ' '),  # Tabs -> space
    (r'^\s+', ''),  # Leading whitespace
    (r'\s+$', ''),  # Trailing whitespace
]


def compile_patterns(patterns: List[str], flags: int = re.IGNORECASE) -> List[Pattern]:
    """Compile regex patterns with error handling."""
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, flags))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{p}': {e}")
    return compiled


# Pre-compiled patterns for performance
_SEC_HEADER_RE = compile_patterns(SEC_HEADER_PATTERNS, re.IGNORECASE | re.MULTILINE)
_SEC_EXHIBIT_RE = compile_patterns(SEC_EXHIBIT_PATTERNS, re.IGNORECASE)
_SEC_TOC_RE = compile_patterns(SEC_TOC_PATTERNS, re.IGNORECASE | re.MULTILINE)
_NEWS_CTA_RE = compile_patterns(NEWS_CTA_PATTERNS, re.IGNORECASE)
_NEWS_COOKIE_RE = compile_patterns(NEWS_COOKIE_PATTERNS, re.IGNORECASE)
_NEWS_SOCIAL_RE = compile_patterns(NEWS_SOCIAL_PATTERNS, re.IGNORECASE)
_EMAIL_SIG_RE = compile_patterns(EMAIL_SIGNATURE_PATTERNS, re.IGNORECASE)
_LEGAL_DISCLAIMER_RE = compile_patterns(LEGAL_DISCLAIMER_PATTERNS, re.IGNORECASE)
_HTML_RE = compile_patterns(HTML_PATTERNS, re.IGNORECASE | re.DOTALL)


# =============================================================================
# Cleaning Functions
# =============================================================================

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove HTML tags
    for pattern in _HTML_RE:
        text = pattern.sub(' ', text)

    # Decode HTML entities
    text = html.unescape(text)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    for pattern, replacement in WHITESPACE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text.strip()


def strip_patterns(text: str, patterns: List[Pattern], replacement: str = '') -> str:
    """Remove all matches of compiled patterns from text."""
    for pattern in patterns:
        text = pattern.sub(replacement, text)
    return text


def strip_lines_matching(text: str, patterns: List[Pattern]) -> str:
    """Remove entire lines that match any pattern."""
    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        should_keep = True
        for pattern in patterns:
            if pattern.search(line):
                should_keep = False
                break
        if should_keep:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


# =============================================================================
# Source-Specific Cleaners
# =============================================================================

class BaseCleaner(ABC):
    """Abstract base class for text cleaners."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean the input text.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text.
        """
        pass

    def __call__(self, text: str) -> str:
        return self.clean(text)


class SECEdgarCleaner(BaseCleaner):
    """Cleaner for SEC EDGAR filings."""

    def __init__(
        self,
        strip_headers: bool = True,
        strip_exhibits: bool = True,
        strip_toc: bool = True,
        strip_html: bool = True,
    ) -> None:
        self.strip_headers = strip_headers
        self.strip_exhibits = strip_exhibits
        self.strip_toc = strip_toc
        self._strip_html = strip_html

    def clean(self, text: str) -> str:
        # Strip HTML first
        if self._strip_html:
            text = strip_html(text)

        # Strip filing headers
        if self.strip_headers:
            text = strip_patterns(text, _SEC_HEADER_RE, '')

        # Strip exhibit references
        if self.strip_exhibits:
            text = strip_lines_matching(text, _SEC_EXHIBIT_RE)

        # Strip table of contents
        if self.strip_toc:
            text = strip_lines_matching(text, _SEC_TOC_RE)

        return normalize_whitespace(text)


class NewsCleaner(BaseCleaner):
    """Cleaner for news articles (GDELT, etc.)."""

    def __init__(
        self,
        strip_cta: bool = True,
        strip_cookie_notices: bool = True,
        strip_social: bool = True,
        strip_html: bool = True,
    ) -> None:
        self.strip_cta = strip_cta
        self.strip_cookie_notices = strip_cookie_notices
        self.strip_social = strip_social
        self._strip_html = strip_html

    def clean(self, text: str) -> str:
        if self._strip_html:
            text = strip_html(text)

        if self.strip_cta:
            text = strip_lines_matching(text, _NEWS_CTA_RE)

        if self.strip_cookie_notices:
            text = strip_patterns(text, _NEWS_COOKIE_RE, '')

        if self.strip_social:
            text = strip_patterns(text, _NEWS_SOCIAL_RE, '')

        return normalize_whitespace(text)


class EmailCleaner(BaseCleaner):
    """Cleaner for email-style text."""

    def __init__(
        self,
        strip_signatures: bool = True,
        strip_disclaimers: bool = True,
    ) -> None:
        self.strip_signatures = strip_signatures
        self.strip_disclaimers = strip_disclaimers

    def clean(self, text: str) -> str:
        if self.strip_signatures:
            text = strip_lines_matching(text, _EMAIL_SIG_RE)

        if self.strip_disclaimers:
            text = strip_lines_matching(text, _LEGAL_DISCLAIMER_RE)

        return normalize_whitespace(text)


class GeneralCleaner(BaseCleaner):
    """General-purpose text cleaner."""

    def __init__(
        self,
        strip_html: bool = True,
        strip_legal: bool = False,  # Keep by default for some sources
        normalize_whitespace: bool = True,
    ) -> None:
        self._strip_html = strip_html
        self.strip_legal = strip_legal
        self._normalize_whitespace = normalize_whitespace

    def clean(self, text: str) -> str:
        if self._strip_html:
            text = strip_html(text)

        if self.strip_legal:
            text = strip_lines_matching(text, _LEGAL_DISCLAIMER_RE)

        if self._normalize_whitespace:
            text = normalize_whitespace(text)

        return text


# =============================================================================
# Boilerplate Stripping Pipeline
# =============================================================================

@dataclass
class CleaningConfig:
    """Configuration for text cleaning."""

    # Source-specific settings
    sec_strip_headers: bool = True
    sec_strip_exhibits: bool = True
    sec_strip_toc: bool = True

    news_strip_cta: bool = True
    news_strip_cookie: bool = True
    news_strip_social: bool = True

    email_strip_signatures: bool = True
    email_strip_disclaimers: bool = True

    # General settings
    strip_html: bool = True
    strip_legal_disclaimers: bool = False  # Keep for some legal contexts
    normalize_whitespace: bool = True

    # Minimum length after cleaning
    min_length_after_clean: int = 50


class BoilerplateStripper:
    """Main class for stripping boilerplate from various sources.

    Automatically selects appropriate cleaner based on source type.
    """

    def __init__(self, config: Optional[CleaningConfig] = None) -> None:
        """Initialize boilerplate stripper.

        Args:
            config: Cleaning configuration.
        """
        self.config = config or CleaningConfig()
        self._cleaners: Dict[DatasetSource, BaseCleaner] = self._build_cleaners()
        self._general_cleaner = GeneralCleaner(
            strip_html=self.config.strip_html,
            strip_legal=self.config.strip_legal_disclaimers,
            normalize_whitespace=self.config.normalize_whitespace,
        )

    def _build_cleaners(self) -> Dict[DatasetSource, BaseCleaner]:
        """Build source-specific cleaners."""
        return {
            DatasetSource.SEC_EDGAR: SECEdgarCleaner(
                strip_headers=self.config.sec_strip_headers,
                strip_exhibits=self.config.sec_strip_exhibits,
                strip_toc=self.config.sec_strip_toc,
                strip_html=self.config.strip_html,
            ),
            DatasetSource.GDELT: NewsCleaner(
                strip_cta=self.config.news_strip_cta,
                strip_cookie_notices=self.config.news_strip_cookie,
                strip_social=self.config.news_strip_social,
                strip_html=self.config.strip_html,
            ),
            DatasetSource.FINCEN_FILES: EmailCleaner(
                strip_signatures=self.config.email_strip_signatures,
                strip_disclaimers=self.config.email_strip_disclaimers,
            ),
            DatasetSource.VENMO_NOTES: GeneralCleaner(
                strip_html=False,  # Venmo notes are plain text
                normalize_whitespace=True,
            ),
        }

    def clean(self, sample: TextSample) -> TextSample:
        """Clean a text sample.

        Args:
            sample: Input sample to clean.

        Returns:
            New TextSample with cleaned text.
        """
        # Get appropriate cleaner
        cleaner = self._cleaners.get(sample.source, self._general_cleaner)

        # Clean the text
        cleaned_text = cleaner.clean(sample.text)

        # Create new sample with cleaned text
        return TextSample(
            text=cleaned_text,
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
                "boilerplate_removed": True,
                "original_length": len(sample.text),
                "cleaned_length": len(cleaned_text),
            },
            token_count=sample.token_count,
            chunk_index=sample.chunk_index,
            total_chunks=sample.total_chunks,
        )

    def clean_batch(self, samples: List[TextSample]) -> List[TextSample]:
        """Clean a batch of samples.

        Args:
            samples: List of samples to clean.

        Returns:
            List of cleaned samples (filtered for minimum length).
        """
        cleaned = []
        for sample in samples:
            cleaned_sample = self.clean(sample)
            if len(cleaned_sample.text) >= self.config.min_length_after_clean:
                cleaned.append(cleaned_sample)
        return cleaned


class TextNormalizer:
    """Utility class for text normalization.

    Provides common normalization operations without source-specific logic.
    """

    def __init__(
        self,
        lowercase: bool = False,
        strip_accents: bool = False,
        normalize_unicode: bool = True,
        fix_encoding: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.normalize_unicode = normalize_unicode
        self.fix_encoding = fix_encoding

    def normalize(self, text: str) -> str:
        """Normalize text.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        # Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Normalize unicode
        if self.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)

        # Strip accents
        if self.strip_accents:
            text = self._strip_accents(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        return normalize_whitespace(text)

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Replace common mojibake patterns
        replacements = [
            ('â€™', "'"),
            ('â€œ', '"'),
            ('â€', '"'),
            ('â€"', '—'),
            ('â€"', '–'),
            ('Â', ''),
            ('\x00', ''),  # Null bytes
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def _strip_accents(self, text: str) -> str:
        """Remove accents from text."""
        import unicodedata
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )


def clean_corpus(
    samples: List[TextSample],
    config: Optional[CleaningConfig] = None,
) -> List[TextSample]:
    """Convenience function to clean a corpus.

    Args:
        samples: Input samples.
        config: Cleaning configuration.

    Returns:
        List of cleaned samples.

    Example:
        ```python
        from rl_money_laundering.dataprep.cleaning import clean_corpus

        # Clean with defaults
        cleaned = clean_corpus(raw_samples)

        # Custom config
        from rl_money_laundering.dataprep.cleaning import CleaningConfig
        config = CleaningConfig(
            strip_legal_disclaimers=True,
            min_length_after_clean=100,
        )
        cleaned = clean_corpus(raw_samples, config=config)
        ```
    """
    stripper = BoilerplateStripper(config)
    return stripper.clean_batch(samples)
