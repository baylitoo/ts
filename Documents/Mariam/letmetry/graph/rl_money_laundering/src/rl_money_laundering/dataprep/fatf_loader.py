"""
FATF Typologies PDF loader.

Extracts money laundering case studies, typology descriptions, and red flag
indicators from FATF (Financial Action Task Force) reports.

FATF reports are gold-standard AML content containing:
- Detailed case studies of ML schemes
- Typology descriptions (layering, structuring, TBML, etc.)
- Red flag indicators for compliance
- Investigative methodologies

Data sources:
- FATF: https://www.fatf-gafi.org/en/topics/methods-and-trends.html
- APG: Asia/Pacific Group typology reports
- FinCEN advisories
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from .base import (
    BaseTextDataset,
    DatasetConfig,
    DatasetRegistry,
    DatasetSource,
    FraudLabel,
    TextCategory,
    TextSample,
)

logger = logging.getLogger(__name__)


# Known FATF report URLs (direct PDF links)
FATF_REPORT_URLS = {
    "typologies_2000_2001": "https://www.fincen.gov/system/files/shared/fatftypologie.pdf",
    "professional_ml_2018": "https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Professional-Money-Laundering.pdf",
    "tbml_2006": "https://home.treasury.gov/system/files/246/Trade-based-ML-062006.pdf",
    "legal_professionals": "https://www.fatf-gafi.org/content/dam/fatf/documents/reports/ML%20and%20TF%20vulnerabilities%20legal%20professionals.pdf",
}

# Typology keywords for classification
TYPOLOGY_KEYWORDS = {
    "layering": ["layering", "layers", "multiple transfers", "chain of transactions", "complex web"],
    "structuring": ["structuring", "smurfing", "split transactions", "below threshold", "multiple deposits"],
    "trade_based": ["trade-based", "trade based", "tbml", "over-invoicing", "under-invoicing", "phantom shipments"],
    "shell_company": ["shell company", "shell companies", "nominee", "front company", "paper company"],
    "cash_intensive": ["cash-intensive", "cash intensive", "currency exchange", "money service business"],
    "real_estate": ["real estate", "property", "real property", "land transactions"],
    "cryptocurrency": ["cryptocurrency", "virtual currency", "bitcoin", "crypto assets", "digital assets"],
    "correspondent_banking": ["correspondent banking", "nested accounts", "payable-through"],
    "hawala": ["hawala", "informal value transfer", "ivts", "alternative remittance"],
    "pep": ["politically exposed", "pep", "government official", "public official"],
    "terrorist_financing": ["terrorist financing", "terrorism", "tf", "terrorist organization"],
    "sanctions_evasion": ["sanctions", "embargo", "ofac", "sanctions evasion"],
}

# Red flag patterns
RED_FLAG_PATTERNS = [
    r"red\s+flag[s]?",
    r"indicator[s]?\s+(?:of|for)",
    r"warning\s+sign[s]?",
    r"suspicious\s+(?:activity|transaction|behavior)",
    r"unusual\s+(?:pattern|activity|transaction)",
    r"inconsistent\s+with",
    r"no\s+(?:apparent|legitimate|economic)\s+(?:purpose|reason)",
    r"lack[s]?\s+(?:of\s+)?(?:documentation|transparency)",
    r"complex\s+(?:structure|arrangement)",
    r"rapid\s+(?:movement|transfer)",
]


@dataclass
class ExtractedSection:
    """A section extracted from a FATF report."""

    title: str
    content: str
    page_number: Optional[int] = None
    section_type: str = "general"
    typologies: List[str] = None
    is_case_study: bool = False

    def __post_init__(self):
        if self.typologies is None:
            self.typologies = []


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using available libraries.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Extracted text content.
    """
    text = ""

    # Try PyPDF2 first
    try:
        import PyPDF2

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        if text.strip():
            return text
    except ImportError:
        logger.debug("PyPDF2 not available")
    except Exception as e:
        logger.debug(f"PyPDF2 extraction failed: {e}")

    # Try pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        if text.strip():
            return text
    except ImportError:
        logger.debug("pdfplumber not available")
    except Exception as e:
        logger.debug(f"pdfplumber extraction failed: {e}")

    # Try pymupdf (fitz)
    try:
        import fitz

        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
        if text.strip():
            return text
    except ImportError:
        logger.debug("pymupdf not available")
    except Exception as e:
        logger.debug(f"pymupdf extraction failed: {e}")

    logger.warning(
        f"Could not extract text from {pdf_path}. "
        "Install a PDF library: pip install PyPDF2 pdfplumber pymupdf"
    )
    return text


def detect_typologies(text: str) -> List[str]:
    """Detect mentioned typologies in text."""
    text_lower = text.lower()
    found = []
    for typology, keywords in TYPOLOGY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                found.append(typology)
                break
    return found


def detect_red_flags(text: str) -> List[str]:
    """Extract red flag indicators from text."""
    flags = []
    text_lower = text.lower()

    for pattern in RED_FLAG_PATTERNS:
        matches = re.findall(pattern, text_lower)
        flags.extend(matches)

    return list(set(flags))


def is_case_study(text: str) -> bool:
    """Check if text appears to be a case study."""
    indicators = [
        r"case\s+(?:study|example|no\.?\s*\d+)",
        r"example\s+\d+",
        r"scenario\s+\d+",
        r"(?:the|a)\s+(?:following|this)\s+case",
        r"in\s+this\s+(?:case|example)",
    ]

    text_lower = text.lower()
    for pattern in indicators:
        if re.search(pattern, text_lower):
            return True
    return False


def split_into_sections(text: str) -> List[ExtractedSection]:
    """Split document text into logical sections.

    Args:
        text: Full document text.

    Returns:
        List of extracted sections.
    """
    sections = []

    # Common section header patterns
    header_patterns = [
        r"^(?:CHAPTER|Chapter)\s+\d+[:\.]?\s*(.+)$",
        r"^(?:SECTION|Section)\s+\d+[:\.]?\s*(.+)$",
        r"^(?:\d+\.)+\s+(.+)$",  # Numbered sections like "1.2.3 Title"
        r"^(?:ANNEX|Annex)\s+\w+[:\.]?\s*(.+)$",
        r"^(?:APPENDIX|Appendix)\s+\w+[:\.]?\s*(.+)$",
        r"^Case\s+(?:Study|Example)\s*\d*[:\.]?\s*(.*)$",
        r"^(?:Box|Figure|Table)\s+\d+[:\.]?\s*(.+)$",
    ]

    # Split by common section markers
    lines = text.split("\n")
    current_section = ExtractedSection(title="Introduction", content="")
    current_content = []

    for line in lines:
        line = line.strip()
        if not line:
            current_content.append("")
            continue

        # Check if this line is a section header
        is_header = False
        header_title = None

        for pattern in header_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                is_header = True
                header_title = match.group(1) if match.lastindex else line
                break

        # Also check for ALL CAPS lines (often headers)
        if not is_header and line.isupper() and len(line) > 10 and len(line) < 100:
            is_header = True
            header_title = line

        if is_header:
            # Save current section
            if current_content:
                current_section.content = "\n".join(current_content).strip()
                if len(current_section.content) > 100:  # Minimum content
                    current_section.typologies = detect_typologies(current_section.content)
                    current_section.is_case_study = is_case_study(current_section.content)
                    sections.append(current_section)

            # Start new section
            current_section = ExtractedSection(
                title=header_title or line,
                content="",
            )
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_content:
        current_section.content = "\n".join(current_content).strip()
        if len(current_section.content) > 100:
            current_section.typologies = detect_typologies(current_section.content)
            current_section.is_case_study = is_case_study(current_section.content)
            sections.append(current_section)

    return sections


@DatasetRegistry.register(DatasetSource.FATF_TYPOLOGIES)
class FATFTypologyLoader(BaseTextDataset):
    """Loader for FATF typology reports.

    Extracts and structures content from FATF PDF reports including:
    - Case studies with detailed ML schemes
    - Typology descriptions
    - Red flag indicators
    - Investigative methodologies

    The extracted text is ideal for training an LLM judge to understand
    AML reasoning and typology patterns.
    """

    source = DatasetSource.FATF_TYPOLOGIES

    def __init__(
        self,
        config: DatasetConfig,
        report_urls: Optional[Dict[str, str]] = None,
        extract_case_studies: bool = True,
        min_section_length: int = 200,
    ) -> None:
        """Initialize FATF loader.

        Args:
            config: Dataset configuration.
            report_urls: Custom report URLs to download.
            extract_case_studies: Prioritize case study extraction.
            min_section_length: Minimum section length to include.
        """
        super().__init__(config)
        self.report_urls = report_urls or FATF_REPORT_URLS
        self.extract_case_studies = extract_case_studies
        self.min_section_length = min_section_length

    def download(self) -> bool:
        """Download FATF reports.

        Returns:
            True if reports are available.
        """
        # Check for existing PDFs
        pdf_files = list(self.config.data_dir.glob("*.pdf"))
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} FATF PDFs")
            return True

        # Download reports
        if self.config.auto_download:
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0

            for name, url in self.report_urls.items():
                try:
                    local_path = self.config.data_dir / f"{name}.pdf"
                    if local_path.exists():
                        continue

                    logger.info(f"Downloading {name} from {url}")
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()

                    local_path.write_bytes(response.content)
                    downloaded += 1
                    logger.info(f"Saved to {local_path}")

                except Exception as e:
                    logger.warning(f"Failed to download {name}: {e}")

            if downloaded > 0:
                return True

        logger.warning(
            f"FATF reports not found at {self.config.data_dir}\n"
            f"Download manually from: https://www.fatf-gafi.org/en/topics/methods-and-trends.html"
        )
        return False

    def load(self) -> List[TextSample]:
        """Load and parse FATF reports.

        Returns:
            List of TextSample objects from report sections.
        """
        if not self.download():
            return []

        samples = []

        for pdf_path in self.config.data_dir.glob("*.pdf"):
            try:
                report_samples = self._process_pdf(pdf_path)
                samples.extend(report_samples)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")

        # Also process any text files (pre-extracted)
        for txt_path in self.config.data_dir.glob("*.txt"):
            try:
                text = txt_path.read_text(encoding="utf-8")
                txt_samples = self._process_text(text, txt_path.stem)
                samples.extend(txt_samples)
            except Exception as e:
                logger.error(f"Error processing {txt_path}: {e}")

        logger.info(f"Loaded {len(samples)} FATF samples")
        return samples

    def _process_pdf(self, pdf_path: Path) -> List[TextSample]:
        """Process a single PDF file."""
        samples = []

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return samples

        # Split into sections
        sections = split_into_sections(text)

        report_name = pdf_path.stem
        for idx, section in enumerate(sections):
            if len(section.content) < self.min_section_length:
                continue

            # Determine category
            if section.is_case_study:
                category = TextCategory.TYPOLOGY_CASE
            elif detect_red_flags(section.content):
                category = TextCategory.RED_FLAG_INDICATOR
            else:
                category = TextCategory.INVESTIGATION_REPORT

            # Build sample
            sample = TextSample(
                text=section.content,
                source=DatasetSource.FATF_TYPOLOGIES,
                sample_id=f"fatf_{report_name}_{idx}",
                category=category,
                label=FraudLabel.UNKNOWN,  # Reference material, not labeled
                typologies=section.typologies,
                indicators=detect_red_flags(section.content),
                metadata={
                    "report": report_name,
                    "section_title": section.title,
                    "is_case_study": section.is_case_study,
                    "page_number": section.page_number,
                },
            )
            samples.append(sample)

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        return samples

    def _process_text(self, text: str, source_name: str) -> List[TextSample]:
        """Process pre-extracted text."""
        samples = []

        sections = split_into_sections(text)

        for idx, section in enumerate(sections):
            if len(section.content) < self.min_section_length:
                continue

            category = TextCategory.TYPOLOGY_CASE if section.is_case_study else TextCategory.INVESTIGATION_REPORT

            sample = TextSample(
                text=section.content,
                source=DatasetSource.FATF_TYPOLOGIES,
                sample_id=f"fatf_txt_{source_name}_{idx}",
                category=category,
                label=FraudLabel.UNKNOWN,
                typologies=section.typologies,
                indicators=detect_red_flags(section.content),
                metadata={
                    "source_file": source_name,
                    "section_title": section.title,
                },
            )
            samples.append(sample)

        return samples

    def get_case_studies(self) -> List[TextSample]:
        """Get only case study samples."""
        return [
            s for s in self.get_samples()
            if s.metadata.get("is_case_study", False) or s.category == TextCategory.TYPOLOGY_CASE
        ]

    def get_by_typology(self, typology: str) -> List[TextSample]:
        """Get samples mentioning a specific typology.

        Args:
            typology: Typology name (e.g., 'layering', 'structuring').

        Returns:
            Filtered samples.
        """
        return [s for s in self.get_samples() if typology in s.typologies]

    def get_red_flag_sections(self) -> List[TextSample]:
        """Get sections containing red flag indicators."""
        return [s for s in self.get_samples() if s.indicators]

    def get_typology_distribution(self) -> Dict[str, int]:
        """Get distribution of typologies across samples."""
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            for typology in sample.typologies:
                counts[typology] = counts.get(typology, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def add_custom_report(self, pdf_path: Path) -> int:
        """Add a custom FATF report to the dataset.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Number of samples added.
        """
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return 0

        new_samples = self._process_pdf(pdf_path)
        if self._samples is None:
            self._samples = []
        self._samples.extend(new_samples)

        logger.info(f"Added {len(new_samples)} samples from {pdf_path}")
        return len(new_samples)
