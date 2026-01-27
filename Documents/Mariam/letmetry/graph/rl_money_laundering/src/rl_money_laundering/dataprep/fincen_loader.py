"""
FinCEN Files dataset loader.

Loads SAR (Suspicious Activity Report) narratives from the ICIJ FinCEN Files leak.
This is the highest-signal text data for AML as it contains real compliance
officer reasoning about suspicious transactions.

Data source: https://www.icij.org/investigations/fincen-files/download-fincen-files-transaction-data/
"""

from __future__ import annotations

import csv
import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ICIJ download URLs
FINCEN_DOWNLOAD_URL = "https://offshoreleaks.icij.org/pages/database"
FINCEN_DATA_URL = "https://www.icij.org/investigations/fincen-files/download-fincen-files-transaction-data/"

# Known SAR filing reasons (from FinCEN categories)
SAR_FILING_REASONS = {
    "money_laundering": "Suspicion of money laundering operations",
    "fraud": "Suspicion of fraud",
    "structuring": "Structuring transactions to avoid reporting",
    "financial_instruments": "Financial instruments/monetary contracts",
    "terrorist_financing": "Terrorist financing",
    "bsa_violation": "BSA/Structuring/Money Laundering",
    "other": "Other suspicious activity",
}

# Typology patterns to extract from narratives
TYPOLOGY_PATTERNS = {
    "layering": r"\b(layer(?:ing)?|multiple\s+transfers?|chain\s+of\s+transactions?)\b",
    "structuring": r"\b(structur(?:ing|ed)?|smurfing|split(?:ting)?\s+transactions?)\b",
    "shell_company": r"\b(shell\s+compan(?:y|ies)|offshore\s+entit(?:y|ies)|nominee)\b",
    "round_tripping": r"\b(round[- ]?trip(?:ping)?|circular\s+flow)\b",
    "trade_based": r"\b(trade[- ]?based|over[- ]?invoic(?:ing|ed)|under[- ]?invoic(?:ing|ed))\b",
    "cash_intensive": r"\b(cash[- ]?intensive|currency\s+exchange|money\s+service)\b",
    "high_risk_jurisdiction": r"\b(high[- ]?risk\s+(?:jurisdiction|countr(?:y|ies))|offshore)\b",
    "rapid_movement": r"\b(rapid(?:ly)?\s+(?:mov(?:ed?|ing)|transfer(?:red)?)|quick\s+succession)\b",
    "correspondent_banking": r"\b(correspondent\s+bank(?:ing)?|nested\s+account)\b",
    "pep": r"\b(p\.?e\.?p\.?|politically\s+exposed|government\s+official)\b",
}


def extract_typologies(text: str) -> List[str]:
    """Extract AML typology mentions from narrative text."""
    text_lower = text.lower()
    found = []
    for typology, pattern in TYPOLOGY_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(typology)
    return found


def extract_entities(text: str) -> List[str]:
    """Extract entity names from narrative (basic extraction)."""
    # Look for patterns like "Bank X", "Company Y", quoted names
    entities = []

    # Quoted names
    quoted = re.findall(r'"([^"]+)"', text)
    entities.extend(quoted)

    # Bank names (often followed by specific patterns)
    banks = re.findall(r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Bank\b)", text)
    entities.extend(banks)

    # Company patterns
    companies = re.findall(
        r"(\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:LLC|Ltd|Inc|Corp|Limited)\b)",
        text,
    )
    entities.extend(companies)

    return list(set(entities))[:20]  # Limit to top 20


def extract_indicators(text: str) -> List[str]:
    """Extract red flag indicators from narrative."""
    indicators = []
    indicator_phrases = [
        "no apparent business purpose",
        "unusual pattern",
        "suspicious activity",
        "inconsistent with",
        "no legitimate",
        "potential money laundering",
        "high-risk",
        "lacks economic purpose",
        "rapid movement",
        "unknown beneficial owner",
        "anonymous",
        "complex structure",
        "shell company",
        "nominee",
    ]

    text_lower = text.lower()
    for phrase in indicator_phrases:
        if phrase in text_lower:
            indicators.append(phrase)

    return indicators


@DatasetRegistry.register(DatasetSource.FINCEN_FILES)
class FinCENLoader(BaseTextDataset):
    """Loader for FinCEN Files SAR narratives.

    The FinCEN Files contain ~2,100 SARs with narrative descriptions
    explaining why compliance officers flagged transactions as suspicious.

    Data structure:
    - Transaction data (CSV): amounts, dates, banks, countries
    - SAR narratives: free-text descriptions of suspicious activity
    - Filing reasons: categorized suspicion types

    This loader focuses on extracting and structuring the narrative text
    for use in training an LLM judge.
    """

    source = DatasetSource.FINCEN_FILES

    # Expected files
    TRANSACTIONS_FILE = "fincen_files_transactions.csv"
    NARRATIVES_DIR = "narratives"  # If narratives are in separate files

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize FinCEN loader."""
        super().__init__(config)
        self._transactions_path = config.data_dir / self.TRANSACTIONS_FILE
        self._narratives_path = config.data_dir / self.NARRATIVES_DIR

    def download(self) -> bool:
        """Check for data and provide download instructions.

        The FinCEN Files require manual download from ICIJ due to
        sensitivity. This method checks if data exists and guides
        the user if not.

        Returns:
            True if data is available.
        """
        if self._transactions_path.exists():
            logger.info(f"FinCEN data found at {self._transactions_path}")
            return True

        # Check for any CSV files that might be the data
        csv_files = list(self.config.data_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Found CSV files: {csv_files}")
            # Try to identify the transactions file
            for csv_file in csv_files:
                if "transaction" in csv_file.name.lower() or "fincen" in csv_file.name.lower():
                    self._transactions_path = csv_file
                    return True

        logger.warning(
            f"FinCEN Files not found at {self.config.data_dir}\n"
            f"Please download manually from:\n"
            f"  {FINCEN_DATA_URL}\n"
            f"And place the CSV file in: {self.config.data_dir}"
        )
        return False

    def load(self) -> List[TextSample]:
        """Load FinCEN Files data into TextSample objects.

        Returns:
            List of TextSample objects containing SAR narratives.
        """
        if not self.download():
            logger.warning("FinCEN data not available, returning empty list")
            return []

        samples = []

        # Try loading from transactions CSV
        if self._transactions_path.exists():
            samples.extend(self._load_transactions_csv())

        # Try loading from separate narrative files
        if self._narratives_path.exists() and self._narratives_path.is_dir():
            samples.extend(self._load_narrative_files())

        # Also try loading any JSON exports
        json_files = list(self.config.data_dir.glob("*.json"))
        for json_file in json_files:
            try:
                samples.extend(self._load_json_export(json_file))
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(samples)} FinCEN samples")
        return samples

    def _load_transactions_csv(self) -> List[TextSample]:
        """Load from ICIJ transactions CSV format."""
        samples = []

        try:
            with open(self._transactions_path, "r", encoding="utf-8") as f:
                # Try to detect CSV format
                sample_line = f.readline()
                f.seek(0)

                # Determine delimiter
                delimiter = "," if "," in sample_line else "\t"

                reader = csv.DictReader(f, delimiter=delimiter)

                for row_idx, row in enumerate(reader):
                    sample = self._parse_transaction_row(row, row_idx)
                    if sample:
                        samples.append(sample)

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except Exception as e:
            logger.error(f"Error loading transactions CSV: {e}")

        return samples

    def _parse_transaction_row(
        self,
        row: Dict[str, str],
        row_idx: int,
    ) -> Optional[TextSample]:
        """Parse a single transaction row into a TextSample."""
        # Look for narrative/description fields
        narrative_fields = [
            "narrative",
            "description",
            "sar_narrative",
            "suspicious_activity_description",
            "notes",
            "comment",
            "reason",
            "filing_reason",
        ]

        narrative_text = ""
        for field in narrative_fields:
            # Try exact match and case-insensitive
            for key in row.keys():
                if key.lower() == field.lower():
                    if row[key] and row[key].strip():
                        narrative_text = row[key].strip()
                        break
            if narrative_text:
                break

        # If no narrative found, construct from transaction details
        if not narrative_text:
            narrative_text = self._construct_narrative_from_fields(row)

        if not narrative_text or len(narrative_text) < self.config.min_text_length:
            return None

        # Extract metadata
        amount = self._extract_amount(row)
        date = self._extract_date(row)
        filing_reason = self._extract_filing_reason(row)

        # Extract AML-relevant features from text
        typologies = extract_typologies(narrative_text)
        entities = extract_entities(narrative_text)
        indicators = extract_indicators(narrative_text)

        # All FinCEN Files are suspicious by definition
        sample = TextSample(
            text=narrative_text,
            source=DatasetSource.FINCEN_FILES,
            sample_id=f"fincen_{row_idx}",
            category=TextCategory.SAR_NARRATIVE,
            label=FraudLabel.SUSPICIOUS,
            entities=entities,
            typologies=typologies,
            indicators=indicators,
            timestamp=date,
            metadata={
                "amount": amount,
                "filing_reason": filing_reason,
                "source_row": row_idx,
                "raw_fields": {k: v for k, v in row.items() if v},
            },
        )

        return sample

    def _construct_narrative_from_fields(self, row: Dict[str, str]) -> str:
        """Construct a narrative from structured transaction fields."""
        parts = []

        # Bank information
        bank_fields = ["bank", "filer_name", "institution", "reporting_bank"]
        for field in bank_fields:
            for key, value in row.items():
                if field in key.lower() and value:
                    parts.append(f"Filing institution: {value}")
                    break

        # Transaction details
        if "amount" in str(row.keys()).lower():
            for key, value in row.items():
                if "amount" in key.lower() and value:
                    parts.append(f"Transaction amount: {value}")
                    break

        # Beneficiary/originator
        for key, value in row.items():
            if any(x in key.lower() for x in ["beneficiary", "originator", "sender", "receiver"]):
                if value:
                    parts.append(f"{key}: {value}")

        # Countries
        for key, value in row.items():
            if "country" in key.lower() and value:
                parts.append(f"{key}: {value}")

        # Filing reason
        for key, value in row.items():
            if "reason" in key.lower() and value:
                parts.append(f"Filing reason: {value}")

        return " | ".join(parts) if parts else ""

    def _extract_amount(self, row: Dict[str, str]) -> Optional[float]:
        """Extract transaction amount from row."""
        for key, value in row.items():
            if "amount" in key.lower() and value:
                try:
                    # Remove currency symbols and commas
                    clean = re.sub(r"[^\d.]", "", str(value))
                    return float(clean) if clean else None
                except ValueError:
                    pass
        return None

    def _extract_date(self, row: Dict[str, str]) -> Optional[datetime]:
        """Extract date from row."""
        date_fields = ["date", "filing_date", "transaction_date", "begin_date"]
        for field in date_fields:
            for key, value in row.items():
                if field in key.lower() and value:
                    try:
                        # Try common date formats
                        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]:
                            try:
                                return datetime.strptime(value.strip(), fmt)
                            except ValueError:
                                continue
                    except Exception:
                        pass
        return None

    def _extract_filing_reason(self, row: Dict[str, str]) -> Optional[str]:
        """Extract SAR filing reason from row."""
        for key, value in row.items():
            if any(x in key.lower() for x in ["reason", "type", "category", "violation"]):
                if value:
                    return value.strip()
        return None

    def _load_narrative_files(self) -> List[TextSample]:
        """Load narratives from separate text files."""
        samples = []

        for txt_file in self._narratives_path.glob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8").strip()
                if len(text) >= self.config.min_text_length:
                    sample = TextSample(
                        text=text,
                        source=DatasetSource.FINCEN_FILES,
                        sample_id=f"fincen_narrative_{txt_file.stem}",
                        category=TextCategory.SAR_NARRATIVE,
                        label=FraudLabel.SUSPICIOUS,
                        typologies=extract_typologies(text),
                        entities=extract_entities(text),
                        indicators=extract_indicators(text),
                    )
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {txt_file}: {e}")

        return samples

    def _load_json_export(self, json_path: Path) -> List[TextSample]:
        """Load from JSON export format."""
        samples = []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("records", data.get("data", [data]))
            else:
                return []

            for idx, record in enumerate(records):
                if isinstance(record, dict):
                    # Look for text content
                    text = record.get("narrative") or record.get("text") or record.get("description")
                    if text and len(text) >= self.config.min_text_length:
                        sample = TextSample(
                            text=text,
                            source=DatasetSource.FINCEN_FILES,
                            sample_id=f"fincen_json_{json_path.stem}_{idx}",
                            category=TextCategory.SAR_NARRATIVE,
                            label=FraudLabel.SUSPICIOUS,
                            typologies=extract_typologies(text),
                            entities=extract_entities(text),
                            indicators=extract_indicators(text),
                            metadata=record,
                        )
                        samples.append(sample)

        except Exception as e:
            logger.warning(f"Failed to parse JSON {json_path}: {e}")

        return samples

    def get_by_typology(self, typology: str) -> List[TextSample]:
        """Get samples that mention a specific typology.

        Args:
            typology: Typology name (e.g., 'layering', 'structuring').

        Returns:
            Filtered list of samples.
        """
        return [s for s in self.get_samples() if typology in s.typologies]

    def get_by_filing_reason(self, reason: str) -> List[TextSample]:
        """Get samples with a specific filing reason.

        Args:
            reason: Filing reason keyword.

        Returns:
            Filtered list of samples.
        """
        reason_lower = reason.lower()
        return [
            s for s in self.get_samples()
            if s.metadata.get("filing_reason", "").lower().find(reason_lower) >= 0
        ]

    def get_high_value_transactions(self, min_amount: float = 1_000_000) -> List[TextSample]:
        """Get samples for high-value transactions.

        Args:
            min_amount: Minimum transaction amount.

        Returns:
            Filtered list of samples.
        """
        return [
            s for s in self.get_samples()
            if (s.metadata.get("amount") or 0) >= min_amount
        ]
