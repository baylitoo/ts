"""
Venmo public transaction notes loader.

Loads payment memo text from public Venmo transactions.
These are real natural language descriptions of payments.

Data source: https://github.com/sa7mon/venmo-data
Research: "I know what you did on Venmo" (PETS 2022)

Note: This data should be used for research purposes only.
"""

from __future__ import annotations

import bson
import json
import logging
import lzma
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

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


# Venmo dataset info
VENMO_REPO = "https://github.com/sa7mon/venmo-data"

# Payment categories based on common patterns
PAYMENT_CATEGORIES = {
    "rent": [r"\brent\b", r"\broommate\b", r"\bapartment\b", r"\bhousing\b"],
    "food": [r"\bfood\b", r"\bdinner\b", r"\blunch\b", r"\bbreakfast\b", r"\bcoffee\b", r"\bpizza\b", r"\b🍕\b", r"\b🍔\b"],
    "utilities": [r"\butilities?\b", r"\belectric\b", r"\bgas\b", r"\binternet\b", r"\bwifi\b"],
    "transportation": [r"\buber\b", r"\blyft\b", r"\bgas\b", r"\bparking\b", r"\btransit\b"],
    "entertainment": [r"\bconcert\b", r"\bmovie\b", r"\bticket\b", r"\bshow\b", r"\bgame\b"],
    "shopping": [r"\bshopping\b", r"\bgroceries?\b", r"\bclothes\b"],
    "services": [r"\bhaircut\b", r"\bnails\b", r"\bmassage\b", r"\bgym\b"],
    "gifts": [r"\bbirthday\b", r"\bgift\b", r"\bpresent\b", r"\b🎁\b", r"\b🎂\b"],
    "alcohol": [r"\bdrinks?\b", r"\bbar\b", r"\bbeer\b", r"\bwine\b", r"\b🍺\b", r"\b🍷\b"],
    "travel": [r"\btrip\b", r"\bhotel\b", r"\bflight\b", r"\bvacation\b", r"\bairbnb\b"],
}

# Potentially sensitive patterns (for research on privacy)
SENSITIVE_PATTERNS = {
    "health": [r"\bdoctor\b", r"\bmedical\b", r"\bprescription\b", r"\btherapy\b", r"\bmental\b"],
    "legal": [r"\blawyer\b", r"\blegal\b", r"\bcourt\b", r"\bbail\b"],
    "adult": [r"\b18\+", r"\badult\b"],
    "drugs": [r"\b420\b", r"\bweed\b", r"\b🌿\b"],
    "gambling": [r"\bbet\b", r"\bpoker\b", r"\bcasino\b", r"\b🎰\b"],
}


def categorize_note(note: str) -> List[str]:
    """Categorize a payment note based on content.

    Args:
        note: Payment note text.

    Returns:
        List of detected categories.
    """
    note_lower = note.lower()
    categories = []

    for category, patterns in PAYMENT_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, note_lower):
                categories.append(category)
                break

    return categories


def detect_sensitive_content(note: str) -> List[str]:
    """Detect potentially sensitive content categories.

    Args:
        note: Payment note text.

    Returns:
        List of sensitive categories detected.
    """
    note_lower = note.lower()
    sensitive = []

    for category, patterns in SENSITIVE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, note_lower):
                sensitive.append(category)
                break

    return sensitive


def clean_note(note: str) -> str:
    """Clean and normalize a payment note.

    Args:
        note: Raw payment note.

    Returns:
        Cleaned note.
    """
    # Remove excessive whitespace
    note = re.sub(r"\s+", " ", note).strip()

    # Keep emojis but remove other special characters
    # note = re.sub(r'[^\w\s\U0001F300-\U0001F9FF]', '', note)

    return note


@DatasetRegistry.register(DatasetSource.VENMO_NOTES)
class VenmoNotesLoader(BaseTextDataset):
    """Loader for Venmo public transaction notes.

    Venmo requires a "payment note" for each transaction, and by default
    these are public. This dataset contains millions of real payment
    descriptions in natural language.

    Key characteristics:
    - Very short text (93% have ≤5 words)
    - Contains emojis and informal language
    - Real-world payment purposes

    Use cases:
    - Payment purpose classification
    - Informal text understanding
    - Privacy research
    """

    source = DatasetSource.VENMO_NOTES

    def __init__(
        self,
        config: DatasetConfig,
        min_note_length: int = 2,
        max_note_length: int = 500,
        include_sensitive: bool = False,
    ) -> None:
        """Initialize Venmo loader.

        Args:
            config: Dataset configuration.
            min_note_length: Minimum characters in note.
            max_note_length: Maximum characters in note.
            include_sensitive: Include potentially sensitive notes.
        """
        super().__init__(config)
        self.min_note_length = min_note_length
        self.max_note_length = max_note_length
        self.include_sensitive = include_sensitive

    def download(self) -> bool:
        """Check for Venmo data (manual download required).

        Returns:
            True if data is available.
        """
        # Check for any data files
        data_files = (
            list(self.config.data_dir.glob("*.json")) +
            list(self.config.data_dir.glob("*.jsonl")) +
            list(self.config.data_dir.glob("*.bson")) +
            list(self.config.data_dir.glob("*.xz")) +
            list(self.config.data_dir.glob("*.csv"))
        )

        if data_files:
            logger.info(f"Found Venmo data files: {[f.name for f in data_files]}")
            return True

        logger.warning(
            f"Venmo data not found at {self.config.data_dir}\n"
            f"This dataset requires manual download due to its size and nature.\n"
            f"Repository: {VENMO_REPO}\n"
            f"The dataset is in BSON format compressed with xz."
        )
        return False

    def load(self) -> List[TextSample]:
        """Load Venmo notes data.

        Returns:
            List of TextSample objects with payment notes.
        """
        if not self.download():
            # Generate synthetic examples for testing
            if self.config.auto_download:
                logger.warning("Generating synthetic Venmo examples for testing")
                return self._generate_synthetic_examples()
            return []

        samples = []

        # Try different formats
        for bson_file in self.config.data_dir.glob("*.bson"):
            samples.extend(self._load_bson(bson_file))

        for xz_file in self.config.data_dir.glob("*.xz"):
            samples.extend(self._load_xz_bson(xz_file))

        for json_file in self.config.data_dir.glob("*.json"):
            samples.extend(self._load_json(json_file))

        for jsonl_file in self.config.data_dir.glob("*.jsonl"):
            samples.extend(self._load_jsonl(jsonl_file))

        for csv_file in self.config.data_dir.glob("*.csv"):
            samples.extend(self._load_csv(csv_file))

        logger.info(f"Loaded {len(samples)} Venmo samples")
        return samples

    def _load_bson(self, bson_path: Path) -> List[TextSample]:
        """Load from BSON file."""
        samples = []

        try:
            with open(bson_path, "rb") as f:
                for idx, doc in enumerate(bson.decode_iter(f.read())):
                    sample = self._parse_transaction(doc, idx)
                    if sample:
                        samples.append(sample)

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except ImportError:
            logger.warning("Install pymongo for BSON support: pip install pymongo")
        except Exception as e:
            logger.error(f"Error loading BSON {bson_path}: {e}")

        return samples

    def _load_xz_bson(self, xz_path: Path) -> List[TextSample]:
        """Load from xz-compressed BSON file."""
        samples = []

        try:
            with lzma.open(xz_path, "rb") as f:
                content = f.read()
                for idx, doc in enumerate(bson.decode_iter(content)):
                    sample = self._parse_transaction(doc, idx)
                    if sample:
                        samples.append(sample)

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except ImportError:
            logger.warning("Install pymongo for BSON support: pip install pymongo")
        except Exception as e:
            logger.error(f"Error loading XZ BSON {xz_path}: {e}")

        return samples

    def _load_json(self, json_path: Path) -> List[TextSample]:
        """Load from JSON file."""
        samples = []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("transactions", data.get("data", []))
            else:
                return []

            for idx, record in enumerate(records):
                sample = self._parse_transaction(record, idx)
                if sample:
                    samples.append(sample)

                if self.config.max_samples and len(samples) >= self.config.max_samples:
                    break

        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")

        return samples

    def _load_jsonl(self, jsonl_path: Path) -> List[TextSample]:
        """Load from JSONL file."""
        samples = []

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        sample = self._parse_transaction(record, idx)
                        if sample:
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except Exception as e:
            logger.error(f"Error loading JSONL {jsonl_path}: {e}")

        return samples

    def _load_csv(self, csv_path: Path) -> List[TextSample]:
        """Load from CSV file."""
        samples = []

        try:
            import csv

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    record = {
                        "note": row.get("note") or row.get("message") or row.get("description"),
                        "amount": row.get("amount"),
                        "date": row.get("date") or row.get("created_time"),
                    }

                    sample = self._parse_transaction(record, idx)
                    if sample:
                        samples.append(sample)

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")

        return samples

    def _parse_transaction(self, record: Dict[str, Any], idx: int) -> Optional[TextSample]:
        """Parse a transaction record into TextSample."""
        # Extract note
        note = record.get("note") or record.get("message") or record.get("description")
        if not note:
            return None

        note = clean_note(str(note))

        # Length filters
        if len(note) < self.min_note_length or len(note) > self.max_note_length:
            return None

        # Check for sensitive content
        sensitive = detect_sensitive_content(note)
        if sensitive and not self.include_sensitive:
            return None

        # Categorize
        categories = categorize_note(note)

        # Extract metadata
        amount = record.get("amount")
        if amount and isinstance(amount, str):
            try:
                amount = float(re.sub(r"[^\d.]", "", amount))
            except ValueError:
                amount = None

        date = None
        date_str = record.get("date") or record.get("created_time") or record.get("datetime")
        if date_str:
            try:
                # Try common date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]:
                    try:
                        date = datetime.strptime(str(date_str)[:19], fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        sample = TextSample(
            text=note,
            source=DatasetSource.VENMO_NOTES,
            sample_id=f"venmo_{idx}",
            category=TextCategory.PAYMENT_MEMO,
            label=FraudLabel.UNKNOWN,  # No fraud labels in this dataset
            typologies=categories,
            metadata={
                "amount": amount,
                "categories": categories,
                "sensitive_categories": sensitive if self.include_sensitive else [],
                "word_count": len(note.split()),
                "has_emoji": bool(re.search(r"[\U0001F300-\U0001F9FF]", note)),
            },
            timestamp=date,
        )

        return sample

    def _generate_synthetic_examples(self, n_samples: int = 500) -> List[TextSample]:
        """Generate synthetic Venmo-like notes for testing.

        Args:
            n_samples: Number of synthetic samples.

        Returns:
            List of synthetic TextSample objects.
        """
        import random

        random.seed(self.config.random_seed)
        samples = []

        # Realistic payment note templates
        templates = [
            # Rent/Utilities
            ("Rent 🏠", ["rent"]),
            ("Utilities", ["utilities"]),
            ("Electric bill", ["utilities"]),
            ("Internet", ["utilities"]),

            # Food
            ("Dinner 🍕", ["food"]),
            ("Coffee ☕", ["food"]),
            ("Groceries", ["food", "shopping"]),
            ("Pizza night", ["food"]),
            ("Lunch", ["food"]),
            ("Brunch 🥞", ["food"]),

            # Transportation
            ("Uber", ["transportation"]),
            ("Gas ⛽", ["transportation"]),
            ("Lyft", ["transportation"]),
            ("Parking", ["transportation"]),

            # Entertainment
            ("Concert tickets 🎵", ["entertainment"]),
            ("Movie night 🎬", ["entertainment"]),
            ("Game tickets", ["entertainment"]),

            # Social
            ("Drinks 🍺", ["alcohol"]),
            ("Bar tab", ["alcohol"]),
            ("Happy hour", ["alcohol"]),
            ("Birthday gift 🎁", ["gifts"]),
            ("Thanks!", []),

            # Travel
            ("Airbnb", ["travel"]),
            ("Trip expenses", ["travel"]),
            ("Hotel", ["travel"]),

            # Misc
            ("Haircut 💇", ["services"]),
            ("Gym", ["services"]),
            ("❤️", []),
            ("🙏", []),
            ("Thx!", []),
        ]

        for i in range(n_samples):
            note, categories = random.choice(templates)

            # Add some variation
            if random.random() > 0.7:
                note = note.lower()
            if random.random() > 0.8:
                note = note + "!"

            sample = TextSample(
                text=note,
                source=DatasetSource.VENMO_NOTES,
                sample_id=f"venmo_synth_{i}",
                category=TextCategory.PAYMENT_MEMO,
                label=FraudLabel.UNKNOWN,
                typologies=categories,
                metadata={
                    "synthetic": True,
                    "categories": categories,
                    "word_count": len(note.split()),
                },
            )
            samples.append(sample)

        return samples

    def get_by_category(self, category: str) -> List[TextSample]:
        """Get notes in a specific category.

        Args:
            category: Category name (rent, food, etc.).

        Returns:
            Filtered samples.
        """
        return [
            s for s in self.get_samples()
            if category in s.metadata.get("categories", [])
        ]

    def get_with_emoji(self) -> List[TextSample]:
        """Get notes containing emojis."""
        return [
            s for s in self.get_samples()
            if s.metadata.get("has_emoji", False)
        ]

    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of payment categories."""
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            for category in sample.metadata.get("categories", []):
                counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_word_count_distribution(self) -> Dict[int, int]:
        """Get distribution of word counts."""
        counts: Dict[int, int] = {}
        for sample in self.get_samples():
            wc = sample.metadata.get("word_count", 0)
            counts[wc] = counts.get(wc, 0) + 1
        return dict(sorted(counts.items()))
