"""
SEC EDGAR financial fraud dataset loader.

Loads MD&A (Management Discussion & Analysis) text from SEC filings
for companies involved in fraud vs non-fraud cases.

Data sources:
- HuggingFace: amitkedia/Financial-Fraud-Dataset (85 fraud + 85 non-fraud)
- EDGAR-CORPUS: eloukas/edgar-corpus (6B+ tokens for pretraining)
- Direct EDGAR: Using edgar-crawler for specific filings
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# HuggingFace dataset URLs
HF_FRAUD_DATASET = "amitkedia/Financial-Fraud-Dataset"
HF_EDGAR_CORPUS = "eloukas/edgar-corpus"

# SEC EDGAR API
SEC_EDGAR_API = "https://data.sec.gov"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# Known fraud indicators in MD&A text
FRAUD_LINGUISTIC_INDICATORS = [
    # Deceptive language patterns
    r"\b(aggressively|significantly|substantially)\s+grow",
    r"\b(extraordinary|exceptional|unprecedented)\s+(?:results|performance|growth)",
    r"\bwe\s+(?:believe|expect|anticipate)\s+(?:strong|significant|substantial)",
    # Vague hedging
    r"\b(may|might|could)\s+(?:not\s+)?(?:be\s+)?(?:able|possible)",
    r"\b(?:subject\s+to|depending\s+on)\s+(?:various|certain|numerous)",
    # Blame shifting
    r"\b(?:due\s+to|because\s+of|as\s+a\s+result\s+of)\s+(?:market|economic|industry)",
    # Complexity indicators
    r"\b(?:complex|complicated|sophisticated)\s+(?:structure|arrangement|transaction)",
    r"\b(?:off[- ]?balance[- ]?sheet|special[- ]?purpose)",
]

# Risk disclosure patterns
RISK_PATTERNS = {
    "litigation": r"\b(litigation|lawsuit|legal\s+(?:action|proceeding)|class\s+action)\b",
    "investigation": r"\b(investigation|inquiry|subpoena|sec\s+(?:inquiry|investigation))\b",
    "restatement": r"\b(restat(?:e|ed|ement)|correct(?:ed|ion)|material\s+weakness)\b",
    "going_concern": r"\b(going\s+concern|substantial\s+doubt|ability\s+to\s+continue)\b",
    "internal_control": r"\b(internal\s+control|material\s+weakness|significant\s+deficiency)\b",
}


def detect_fraud_indicators(text: str) -> List[str]:
    """Detect linguistic fraud indicators in text."""
    found = []
    text_lower = text.lower()
    for pattern in FRAUD_LINGUISTIC_INDICATORS:
        if re.search(pattern, text_lower):
            found.append(pattern)
    return found


def detect_risk_patterns(text: str) -> Dict[str, int]:
    """Count risk disclosure patterns in text."""
    text_lower = text.lower()
    counts = {}
    for name, pattern in RISK_PATTERNS.items():
        matches = re.findall(pattern, text_lower)
        if matches:
            counts[name] = len(matches)
    return counts


@DatasetRegistry.register(DatasetSource.SEC_EDGAR)
class SECFraudLoader(BaseTextDataset):
    """Loader for SEC EDGAR financial fraud datasets.

    Primary data source is the HuggingFace Financial-Fraud-Dataset
    which contains MD&A text from 85 fraud and 85 non-fraud companies.

    Can also load from:
    - Local CSV/JSON exports
    - EDGAR-CORPUS for pretraining
    - Direct SEC EDGAR API

    The fraud labels come from SEC Accounting and Auditing Enforcement
    Releases (AAERs) which identify companies with material misstatements.
    """

    source = DatasetSource.SEC_EDGAR

    # Known fraud companies from AAERs (partial list for reference)
    KNOWN_FRAUD_COMPANIES = {
        "enron",
        "worldcom",
        "tyco",
        "healthsouth",
        "adelphia",
        "global crossing",
        "qwest",
        "rite aid",
        "xerox",
        "sunbeam",
    }

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize SEC fraud loader."""
        super().__init__(config)
        self._hf_dataset = None

    def download(self) -> bool:
        """Download dataset from HuggingFace or check local files.

        Returns:
            True if data is available.
        """
        # Check for local files first
        local_files = list(self.config.data_dir.glob("*.csv")) + \
                     list(self.config.data_dir.glob("*.json")) + \
                     list(self.config.data_dir.glob("*.parquet"))

        if local_files:
            logger.info(f"Found local SEC data files: {[f.name for f in local_files]}")
            return True

        # Try HuggingFace
        if self.config.auto_download:
            return self._download_from_huggingface()

        logger.warning(
            f"SEC fraud data not found at {self.config.data_dir}\n"
            f"Enable auto_download or manually download from:\n"
            f"  HuggingFace: https://huggingface.co/datasets/{HF_FRAUD_DATASET}"
        )
        return False

    def _download_from_huggingface(self) -> bool:
        """Download dataset from HuggingFace."""
        try:
            from datasets import load_dataset

            logger.info(f"Downloading {HF_FRAUD_DATASET} from HuggingFace...")
            self._hf_dataset = load_dataset(HF_FRAUD_DATASET)
            logger.info("Download complete")
            return True

        except ImportError:
            logger.warning("Install datasets library: pip install datasets")
            return False
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            return False

    def load(self) -> List[TextSample]:
        """Load SEC fraud dataset into TextSample objects.

        Returns:
            List of TextSample objects containing MD&A text.
        """
        samples = []

        # Try HuggingFace dataset first
        if self._hf_dataset is not None:
            samples.extend(self._load_from_hf_dataset())
        else:
            # Try to load from HuggingFace
            if self._download_from_huggingface() and self._hf_dataset is not None:
                samples.extend(self._load_from_hf_dataset())

        # Load from local files
        for csv_file in self.config.data_dir.glob("*.csv"):
            samples.extend(self._load_from_csv(csv_file))

        for json_file in self.config.data_dir.glob("*.json"):
            samples.extend(self._load_from_json(json_file))

        for parquet_file in self.config.data_dir.glob("*.parquet"):
            samples.extend(self._load_from_parquet(parquet_file))

        logger.info(f"Loaded {len(samples)} SEC fraud samples")
        return samples

    def _load_from_hf_dataset(self) -> List[TextSample]:
        """Load from HuggingFace dataset object."""
        samples = []

        if self._hf_dataset is None:
            return samples

        # Handle different dataset structures
        if "train" in self._hf_dataset:
            data = self._hf_dataset["train"]
        else:
            data = self._hf_dataset

        for idx, row in enumerate(data):
            sample = self._parse_hf_row(row, idx)
            if sample:
                samples.append(sample)

            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        return samples

    def _parse_hf_row(self, row: Dict[str, Any], idx: int) -> Optional[TextSample]:
        """Parse a row from the HuggingFace dataset."""
        # Get text content - look for MD&A or filings column
        text = None
        for field in ["Fillings", "filings", "MD&A", "mda", "text", "content"]:
            if field in row and row[field]:
                text = str(row[field]).strip()
                break

        if not text or len(text) < self.config.min_text_length:
            return None

        # Get fraud label
        label = FraudLabel.UNKNOWN
        for field in ["Label", "label", "fraud", "is_fraud", "fraudulent"]:
            if field in row:
                label_val = row[field]
                if isinstance(label_val, bool):
                    label = FraudLabel.FRAUD if label_val else FraudLabel.NON_FRAUD
                elif isinstance(label_val, (int, float)):
                    label = FraudLabel.FRAUD if label_val == 1 else FraudLabel.NON_FRAUD
                elif isinstance(label_val, str):
                    label_lower = label_val.lower()
                    if "fraud" in label_lower or label_val == "1":
                        label = FraudLabel.FRAUD
                    elif "non" in label_lower or "legitimate" in label_lower or label_val == "0":
                        label = FraudLabel.NON_FRAUD
                break

        # Get company name
        company = None
        for field in ["Company", "company", "name", "company_name", "ticker"]:
            if field in row and row[field]:
                company = str(row[field])
                break

        # Determine text category
        category = TextCategory.MD_AND_A
        if "risk" in text.lower()[:500]:
            category = TextCategory.RISK_DISCLOSURE
        elif "financial statement" in text.lower()[:500]:
            category = TextCategory.FINANCIAL_STATEMENT

        # Extract fraud indicators
        indicators = detect_fraud_indicators(text)
        risk_patterns = detect_risk_patterns(text)

        sample = TextSample(
            text=text,
            source=DatasetSource.SEC_EDGAR,
            sample_id=f"sec_fraud_{idx}",
            category=category,
            label=label,
            entities=[company] if company else [],
            indicators=indicators,
            metadata={
                "company": company,
                "risk_patterns": risk_patterns,
                "fraud_indicator_count": len(indicators),
                "source_dataset": HF_FRAUD_DATASET,
            },
        )

        return sample

    def _load_from_csv(self, csv_path: Path) -> List[TextSample]:
        """Load from local CSV file."""
        samples = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    # Convert to dict format expected by parser
                    sample = self._parse_hf_row(dict(row), idx)
                    if sample:
                        sample.sample_id = f"sec_csv_{csv_path.stem}_{idx}"
                        samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")

        return samples

    def _load_from_json(self, json_path: Path) -> List[TextSample]:
        """Load from local JSON file."""
        samples = []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("data", data.get("records", [data]))
            else:
                return []

            for idx, row in enumerate(records):
                if isinstance(row, dict):
                    sample = self._parse_hf_row(row, idx)
                    if sample:
                        sample.sample_id = f"sec_json_{json_path.stem}_{idx}"
                        samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")

        return samples

    def _load_from_parquet(self, parquet_path: Path) -> List[TextSample]:
        """Load from Parquet file."""
        samples = []

        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)

            for idx, row in df.iterrows():
                sample = self._parse_hf_row(row.to_dict(), idx)
                if sample:
                    sample.sample_id = f"sec_parquet_{parquet_path.stem}_{idx}"
                    samples.append(sample)

        except ImportError:
            logger.warning("Install pandas and pyarrow: pip install pandas pyarrow")
        except Exception as e:
            logger.error(f"Error loading Parquet {parquet_path}: {e}")

        return samples

    def get_fraud_samples(self) -> List[TextSample]:
        """Get only fraud-labeled samples."""
        return [s for s in self.get_samples() if s.label == FraudLabel.FRAUD]

    def get_non_fraud_samples(self) -> List[TextSample]:
        """Get only non-fraud samples."""
        return [s for s in self.get_samples() if s.label == FraudLabel.NON_FRAUD]

    def get_balanced_samples(self, n_per_class: Optional[int] = None) -> List[TextSample]:
        """Get balanced fraud/non-fraud samples.

        Args:
            n_per_class: Number of samples per class (uses min if None).

        Returns:
            Balanced list of samples.
        """
        import random

        fraud = self.get_fraud_samples()
        non_fraud = self.get_non_fraud_samples()

        if n_per_class is None:
            n_per_class = min(len(fraud), len(non_fraud))

        random.seed(self.config.random_seed)
        fraud_sample = random.sample(fraud, min(n_per_class, len(fraud)))
        non_fraud_sample = random.sample(non_fraud, min(n_per_class, len(non_fraud)))

        combined = fraud_sample + non_fraud_sample
        random.shuffle(combined)

        return combined

    def get_samples_with_risk_pattern(self, pattern: str) -> List[TextSample]:
        """Get samples containing a specific risk pattern.

        Args:
            pattern: Risk pattern name (litigation, investigation, etc.).

        Returns:
            Filtered list of samples.
        """
        return [
            s for s in self.get_samples()
            if pattern in s.metadata.get("risk_patterns", {})
        ]


class EDGARCorpusLoader(BaseTextDataset):
    """Loader for the full EDGAR corpus (6B+ tokens) for pretraining.

    This loader is for domain-adaptive pretraining (DAPT) on financial
    language, not for supervised fraud detection.

    Data source: https://huggingface.co/datasets/eloukas/edgar-corpus
    """

    source = DatasetSource.SEC_EDGAR

    def __init__(
        self,
        config: DatasetConfig,
        years: Optional[List[int]] = None,
        sections: Optional[List[str]] = None,
    ) -> None:
        """Initialize EDGAR corpus loader.

        Args:
            config: Dataset configuration.
            years: Specific years to load (None for all).
            sections: Specific sections to load (e.g., ['item_7', 'item_1a']).
        """
        super().__init__(config)
        self.years = years
        self.sections = sections or ["item_7"]  # MD&A by default
        self._hf_dataset = None

    def download(self) -> bool:
        """Download EDGAR corpus from HuggingFace."""
        if self.config.auto_download:
            try:
                from datasets import load_dataset

                logger.info(f"Loading EDGAR corpus from HuggingFace...")
                # The dataset is large, so we may want to stream
                self._hf_dataset = load_dataset(
                    HF_EDGAR_CORPUS,
                    streaming=True,
                )
                return True

            except ImportError:
                logger.warning("Install datasets: pip install datasets")
            except Exception as e:
                logger.error(f"Failed to load EDGAR corpus: {e}")

        return False

    def load(self) -> List[TextSample]:
        """Load EDGAR corpus samples.

        Note: This dataset is very large. Use max_samples or streaming.
        """
        samples = []

        if not self.download():
            return samples

        if self._hf_dataset is None:
            return samples

        # Stream through the dataset
        count = 0
        for split_name in self._hf_dataset.keys():
            for row in self._hf_dataset[split_name]:
                # Extract text from the specified sections
                text = ""
                for section in self.sections:
                    if section in row and row[section]:
                        text += row[section] + "\n\n"

                text = text.strip()
                if len(text) < self.config.min_text_length:
                    continue

                # Filter by year if specified
                if self.years:
                    year = row.get("year") or row.get("filing_year")
                    if year and int(year) not in self.years:
                        continue

                sample = TextSample(
                    text=text,
                    source=DatasetSource.SEC_EDGAR,
                    sample_id=f"edgar_{count}",
                    category=TextCategory.MD_AND_A,
                    label=FraudLabel.UNKNOWN,  # Unlabeled for pretraining
                    metadata={
                        "cik": row.get("cik"),
                        "company": row.get("company"),
                        "year": row.get("year"),
                        "sections": self.sections,
                    },
                )
                samples.append(sample)
                count += 1

                if self.config.max_samples and count >= self.config.max_samples:
                    break

            if self.config.max_samples and count >= self.config.max_samples:
                break

        logger.info(f"Loaded {len(samples)} EDGAR corpus samples")
        return samples

    def iter_streaming(self):
        """Iterate through corpus in streaming mode (memory efficient).

        Yields:
            TextSample objects one at a time.
        """
        if not self.download() or self._hf_dataset is None:
            return

        count = 0
        for split_name in self._hf_dataset.keys():
            for row in self._hf_dataset[split_name]:
                text = ""
                for section in self.sections:
                    if section in row and row[section]:
                        text += row[section] + "\n\n"

                text = text.strip()
                if len(text) < self.config.min_text_length:
                    continue

                sample = TextSample(
                    text=text,
                    source=DatasetSource.SEC_EDGAR,
                    sample_id=f"edgar_stream_{count}",
                    category=TextCategory.MD_AND_A,
                    label=FraudLabel.UNKNOWN,
                )
                yield sample
                count += 1

                if self.config.max_samples and count >= self.config.max_samples:
                    return
