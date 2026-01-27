"""
FraudNLP dataset loader.

Loads the FraudNLP dataset which treats user action sequences as
"language" for NLP-based fraud detection.

This is a unique approach where behavioral sequences (login, balance_check,
transfer, logout) are treated as sentences that can be processed by
language models.

Data source: https://github.com/pboulieris/FraudNLP
Paper: "Fraud Detection with NLP" (Springer Machine Learning)
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


# GitHub repository
FRAUDNLP_REPO = "https://github.com/pboulieris/FraudNLP"
FRAUDNLP_RAW_URL = "https://raw.githubusercontent.com/pboulieris/FraudNLP/main"

# Action vocabulary (common actions in online banking)
ACTION_VOCABULARY = {
    # Authentication actions
    "login": "User authenticated and logged into the system",
    "logout": "User logged out of the session",
    "login_failed": "Authentication attempt failed",
    "password_reset": "User requested password reset",
    "mfa_verify": "Multi-factor authentication verified",
    "mfa_failed": "Multi-factor authentication failed",

    # Account inquiry actions
    "balance_check": "User checked account balance",
    "view_statement": "User viewed account statement",
    "view_transactions": "User viewed transaction history",
    "view_profile": "User viewed profile information",

    # Transfer actions
    "transfer_internal": "Internal fund transfer initiated",
    "transfer_external": "External fund transfer initiated",
    "transfer_international": "International wire transfer initiated",
    "transfer_failed": "Transfer attempt failed",
    "transfer_cancelled": "Transfer was cancelled",

    # Payment actions
    "bill_payment": "Bill payment initiated",
    "scheduled_payment": "Scheduled payment created",
    "recurring_payment": "Recurring payment setup",

    # Account management
    "add_beneficiary": "New beneficiary added",
    "remove_beneficiary": "Beneficiary removed",
    "update_contact": "Contact information updated",
    "change_password": "Password changed",
    "change_limits": "Transaction limits modified",

    # Session actions
    "session_timeout": "Session timed out due to inactivity",
    "session_expired": "Session expired",
    "concurrent_login": "Concurrent login detected",
}

# Suspicious action patterns
SUSPICIOUS_PATTERNS = [
    # Rapid account reconnaissance
    ["login", "balance_check", "view_transactions", "logout"],
    # Quick beneficiary addition followed by transfer
    ["login", "add_beneficiary", "transfer_external", "logout"],
    # Multiple failed attempts
    ["login_failed", "login_failed", "login"],
    # Password change followed by immediate transfer
    ["login", "change_password", "transfer_external"],
    # International transfer pattern
    ["login", "balance_check", "transfer_international", "logout"],
    # Limit changes followed by large transfer
    ["login", "change_limits", "transfer_external"],
]


def action_sequence_to_text(
    actions: Sequence[str],
    include_descriptions: bool = True,
    separator: str = " → ",
) -> str:
    """Convert action sequence to natural language text.

    Args:
        actions: Sequence of action tokens.
        include_descriptions: Add action descriptions.
        separator: Separator between actions.

    Returns:
        Text representation of the action sequence.
    """
    if include_descriptions:
        parts = []
        for action in actions:
            desc = ACTION_VOCABULARY.get(action, action)
            parts.append(f"{action} ({desc})")
        return separator.join(parts)
    else:
        return separator.join(actions)


def detect_suspicious_patterns(actions: Sequence[str]) -> List[str]:
    """Detect known suspicious patterns in action sequence.

    Args:
        actions: Sequence of actions.

    Returns:
        List of detected pattern names.
    """
    detected = []
    action_str = " ".join(actions)

    for pattern in SUSPICIOUS_PATTERNS:
        pattern_str = " ".join(pattern)
        if pattern_str in action_str:
            detected.append("_".join(pattern[:2]))  # Name by first two actions

    # Additional heuristic patterns
    if actions.count("login_failed") >= 2:
        detected.append("multiple_failed_logins")

    if "transfer_external" in actions or "transfer_international" in actions:
        # Quick transfer after login
        if len(actions) <= 4:
            detected.append("quick_transfer")

    if "add_beneficiary" in actions and any("transfer" in a for a in actions):
        detected.append("new_beneficiary_transfer")

    if "change_limits" in actions:
        detected.append("limit_modification")

    return detected


@DatasetRegistry.register(DatasetSource.FRAUDNLP)
class FraudNLPLoader(BaseTextDataset):
    """Loader for FraudNLP action sequence dataset.

    The FraudNLP dataset treats online banking action sequences as
    "language" - sequences of discrete actions that can be processed
    using NLP techniques like transformers.

    Key insight: Actions form a "grammar" similar to natural language,
    where certain sequences are suspicious (like grammatically incorrect
    sentences indicate problems).

    Dataset statistics:
    - ~105K transactions
    - ~2K users
    - Binary fraud labels
    - Action sequences as input features
    """

    source = DatasetSource.FRAUDNLP

    def __init__(
        self,
        config: DatasetConfig,
        include_descriptions: bool = True,
        max_sequence_length: int = 50,
    ) -> None:
        """Initialize FraudNLP loader.

        Args:
            config: Dataset configuration.
            include_descriptions: Include action descriptions in text.
            max_sequence_length: Maximum actions per sequence.
        """
        super().__init__(config)
        self.include_descriptions = include_descriptions
        self.max_sequence_length = max_sequence_length

    def download(self) -> bool:
        """Download FraudNLP data from GitHub.

        Returns:
            True if data is available.
        """
        # Check for local files
        data_files = list(self.config.data_dir.glob("*.csv")) + \
                    list(self.config.data_dir.glob("*.json")) + \
                    list(self.config.data_dir.glob("*.pkl"))

        if data_files:
            logger.info(f"Found local FraudNLP data: {[f.name for f in data_files]}")
            return True

        # Try to download from GitHub
        if self.config.auto_download:
            return self._download_from_github()

        logger.warning(
            f"FraudNLP data not found at {self.config.data_dir}\n"
            f"Clone the repository:\n"
            f"  git clone {FRAUDNLP_REPO}\n"
            f"Or enable auto_download in config."
        )
        return False

    def _download_from_github(self) -> bool:
        """Download dataset files from GitHub."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Try common file names from the repo
        possible_files = [
            "data/transactions.csv",
            "data/fraud_data.csv",
            "data/sequences.json",
            "dataset/transactions.csv",
        ]

        for file_path in possible_files:
            url = f"{FRAUDNLP_RAW_URL}/{file_path}"
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    local_path = self.config.data_dir / Path(file_path).name
                    local_path.write_bytes(response.content)
                    logger.info(f"Downloaded {file_path} to {local_path}")
                    return True
            except Exception as e:
                logger.debug(f"Failed to download {url}: {e}")

        logger.warning("Could not auto-download FraudNLP data")
        return False

    def load(self) -> List[TextSample]:
        """Load FraudNLP dataset into TextSample objects.

        Returns:
            List of TextSample objects with action sequences as text.
        """
        samples = []

        # Try different file formats
        for csv_file in self.config.data_dir.glob("*.csv"):
            samples.extend(self._load_from_csv(csv_file))

        for json_file in self.config.data_dir.glob("*.json"):
            samples.extend(self._load_from_json(json_file))

        # If no files found, generate synthetic examples for testing
        if not samples and self.config.auto_download:
            logger.warning("No FraudNLP data found, generating synthetic examples")
            samples = self._generate_synthetic_examples()

        logger.info(f"Loaded {len(samples)} FraudNLP samples")
        return samples

    def _load_from_csv(self, csv_path: Path) -> List[TextSample]:
        """Load from CSV file."""
        samples = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    sample = self._parse_row(row, idx)
                    if sample:
                        samples.append(sample)

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")

        return samples

    def _load_from_json(self, json_path: Path) -> List[TextSample]:
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
                sample = self._parse_row(record, idx)
                if sample:
                    samples.append(sample)

                if self.config.max_samples and len(samples) >= self.config.max_samples:
                    break

        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")

        return samples

    def _parse_row(self, row: Dict[str, Any], idx: int) -> Optional[TextSample]:
        """Parse a single row into a TextSample."""
        # Extract action sequence
        actions = None

        # Try different column names
        for field in ["actions", "sequence", "action_sequence", "events", "steps"]:
            if field in row:
                value = row[field]
                if isinstance(value, list):
                    actions = value
                elif isinstance(value, str):
                    # Parse string representation
                    if value.startswith("["):
                        try:
                            actions = json.loads(value)
                        except json.JSONDecodeError:
                            actions = value.strip("[]").split(",")
                    else:
                        # Assume space or comma separated
                        actions = re.split(r"[,\s→]+", value)
                break

        if not actions or len(actions) < 2:
            return None

        # Clean and limit actions
        actions = [a.strip().lower() for a in actions if a.strip()]
        actions = actions[:self.max_sequence_length]

        # Convert to text
        text = action_sequence_to_text(
            actions,
            include_descriptions=self.include_descriptions,
        )

        if len(text) < self.config.min_text_length:
            return None

        # Extract label
        label = FraudLabel.UNKNOWN
        for field in ["fraud", "is_fraud", "label", "fraudulent", "suspicious"]:
            if field in row:
                val = row[field]
                if isinstance(val, bool):
                    label = FraudLabel.FRAUD if val else FraudLabel.NON_FRAUD
                elif isinstance(val, (int, float)):
                    label = FraudLabel.FRAUD if val == 1 else FraudLabel.NON_FRAUD
                elif isinstance(val, str):
                    if val.lower() in ["1", "true", "fraud", "yes"]:
                        label = FraudLabel.FRAUD
                    elif val.lower() in ["0", "false", "legitimate", "no"]:
                        label = FraudLabel.NON_FRAUD
                break

        # Detect suspicious patterns
        suspicious_patterns = detect_suspicious_patterns(actions)

        # Extract user ID if present
        user_id = row.get("user_id") or row.get("user") or row.get("customer_id")

        sample = TextSample(
            text=text,
            source=DatasetSource.FRAUDNLP,
            sample_id=f"fraudnlp_{idx}",
            category=TextCategory.ACTION_SEQUENCE,
            label=label,
            typologies=suspicious_patterns,
            indicators=suspicious_patterns,
            metadata={
                "actions": actions,
                "action_count": len(actions),
                "user_id": user_id,
                "suspicious_patterns": suspicious_patterns,
            },
        )

        return sample

    def _generate_synthetic_examples(self, n_samples: int = 1000) -> List[TextSample]:
        """Generate synthetic action sequences for testing.

        Args:
            n_samples: Number of synthetic samples.

        Returns:
            List of synthetic TextSample objects.
        """
        import random

        random.seed(self.config.random_seed)
        samples = []

        # Normal action templates
        normal_templates = [
            ["login", "balance_check", "logout"],
            ["login", "view_transactions", "view_statement", "logout"],
            ["login", "balance_check", "bill_payment", "logout"],
            ["login", "view_profile", "update_contact", "logout"],
            ["login", "balance_check", "transfer_internal", "logout"],
            ["login", "view_transactions", "balance_check", "logout"],
        ]

        # Suspicious action templates
        suspicious_templates = [
            ["login_failed", "login_failed", "login", "add_beneficiary", "transfer_external", "logout"],
            ["login", "change_limits", "transfer_international", "logout"],
            ["login", "add_beneficiary", "balance_check", "transfer_external", "logout"],
            ["login", "change_password", "add_beneficiary", "transfer_external", "logout"],
            ["login", "balance_check", "transfer_external", "transfer_external", "logout"],
        ]

        n_suspicious = n_samples // 10  # 10% fraud rate
        n_normal = n_samples - n_suspicious

        # Generate normal samples
        for i in range(n_normal):
            template = random.choice(normal_templates)
            # Add some variation
            actions = template.copy()
            if random.random() > 0.7:
                actions.insert(random.randint(1, len(actions) - 1), "balance_check")

            text = action_sequence_to_text(actions, self.include_descriptions)

            sample = TextSample(
                text=text,
                source=DatasetSource.FRAUDNLP,
                sample_id=f"fraudnlp_synth_normal_{i}",
                category=TextCategory.ACTION_SEQUENCE,
                label=FraudLabel.NON_FRAUD,
                typologies=[],
                metadata={
                    "actions": actions,
                    "synthetic": True,
                },
            )
            samples.append(sample)

        # Generate suspicious samples
        for i in range(n_suspicious):
            template = random.choice(suspicious_templates)
            actions = template.copy()

            text = action_sequence_to_text(actions, self.include_descriptions)
            patterns = detect_suspicious_patterns(actions)

            sample = TextSample(
                text=text,
                source=DatasetSource.FRAUDNLP,
                sample_id=f"fraudnlp_synth_fraud_{i}",
                category=TextCategory.ACTION_SEQUENCE,
                label=FraudLabel.FRAUD,
                typologies=patterns,
                indicators=patterns,
                metadata={
                    "actions": actions,
                    "synthetic": True,
                    "suspicious_patterns": patterns,
                },
            )
            samples.append(sample)

        random.shuffle(samples)
        return samples

    def get_by_pattern(self, pattern_name: str) -> List[TextSample]:
        """Get samples containing a specific suspicious pattern.

        Args:
            pattern_name: Name of the pattern.

        Returns:
            Filtered list of samples.
        """
        return [
            s for s in self.get_samples()
            if pattern_name in s.metadata.get("suspicious_patterns", [])
        ]

    def get_by_action_count(
        self,
        min_actions: int = 0,
        max_actions: int = 100,
    ) -> List[TextSample]:
        """Get samples within an action count range.

        Args:
            min_actions: Minimum number of actions.
            max_actions: Maximum number of actions.

        Returns:
            Filtered list of samples.
        """
        return [
            s for s in self.get_samples()
            if min_actions <= s.metadata.get("action_count", 0) <= max_actions
        ]

    def get_action_vocabulary(self) -> Dict[str, int]:
        """Get frequency counts for all actions in the dataset.

        Returns:
            Dictionary mapping actions to their frequencies.
        """
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            for action in sample.metadata.get("actions", []):
                counts[action] = counts.get(action, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
