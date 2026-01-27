"""
OpenSanctions dataset loader.

Loads entity data from OpenSanctions - an aggregated database of sanctions,
PEPs (Politically Exposed Persons), and persons of interest.

Data source: https://www.opensanctions.org/datasets/
API docs: https://www.opensanctions.org/docs/api/

Key datasets:
- default: Combined sanctions and PEP data
- sanctions: Pure sanctions lists
- peps: Politically exposed persons
- crime: Criminal entities
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

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


# OpenSanctions data URLs
OPENSANCTIONS_BASE_URL = "https://data.opensanctions.org/datasets/latest"
OPENSANCTIONS_DATASETS = {
    "default": f"{OPENSANCTIONS_BASE_URL}/default/entities.ftm.json",
    "sanctions": f"{OPENSANCTIONS_BASE_URL}/sanctions/entities.ftm.json",
    "peps": f"{OPENSANCTIONS_BASE_URL}/peps/entities.ftm.json",
    "crime": f"{OPENSANCTIONS_BASE_URL}/crime/entities.ftm.json",
}

# Alternative CSV format for simpler processing
OPENSANCTIONS_CSV = {
    "default": f"{OPENSANCTIONS_BASE_URL}/default/targets.simple.csv",
    "sanctions": f"{OPENSANCTIONS_BASE_URL}/sanctions/targets.simple.csv",
}

# Entity schema types of interest
RELEVANT_SCHEMAS = {
    "Person",
    "Organization",
    "Company",
    "LegalEntity",
    "PublicBody",
}

# Risk categories
RISK_CATEGORIES = {
    "sanctions": ["OFAC", "EU", "UN", "UK sanctions"],
    "pep": ["PEP", "Politically Exposed Person", "Government Official"],
    "crime": ["Criminal", "Organized Crime", "Fraud", "Money Laundering"],
    "terrorism": ["Terrorist", "Terrorism Financing", "Terror"],
}


class TemplateMode:
    """Template modes for entity text generation."""

    SIMPLE = "simple"      # Basic pipe-delimited fields
    ENRICHED = "enriched"  # Natural language sentences
    NARRATIVE = "narrative"  # Full narrative paragraphs


def entity_to_text(entity: Dict[str, Any], template_mode: str = TemplateMode.SIMPLE) -> str:
    """Convert entity record to natural language text.

    Args:
        entity: OpenSanctions entity dictionary.
        template_mode: How to format the output (simple, enriched, narrative).

    Returns:
        Text representation of the entity.
    """
    if template_mode == TemplateMode.ENRICHED:
        return _entity_to_enriched_text(entity)
    elif template_mode == TemplateMode.NARRATIVE:
        return _entity_to_narrative_text(entity)
    else:
        return _entity_to_simple_text(entity)


def _entity_to_simple_text(entity: Dict[str, Any]) -> str:
    """Simple pipe-delimited format (original behavior)."""
    parts = []

    # Name
    name = entity.get("caption") or entity.get("name")
    if name:
        parts.append(f"Entity: {name}")

    # Schema type
    schema = entity.get("schema")
    if schema:
        parts.append(f"Type: {schema}")

    # Properties
    props = entity.get("properties", {})

    # Aliases
    aliases = props.get("alias", []) or props.get("weakAlias", [])
    if aliases:
        parts.append(f"Also known as: {', '.join(aliases[:5])}")

    # Country
    countries = props.get("country", [])
    if countries:
        parts.append(f"Country: {', '.join(countries)}")

    # Birth date (for persons)
    birth_dates = props.get("birthDate", [])
    if birth_dates:
        parts.append(f"Birth date: {birth_dates[0]}")

    # Position (for PEPs)
    positions = props.get("position", [])
    if positions:
        parts.append(f"Position: {', '.join(positions[:3])}")

    # Sanctions program
    programs = props.get("program", []) or props.get("topics", [])
    if programs:
        parts.append(f"Program/Topic: {', '.join(programs)}")

    # Reason/Notes
    notes = props.get("notes", []) or props.get("summary", [])
    if notes:
        # Take first note, limit length
        note_text = notes[0][:500] if len(notes[0]) > 500 else notes[0]
        parts.append(f"Notes: {note_text}")

    # Description
    descriptions = props.get("description", [])
    if descriptions:
        desc_text = descriptions[0][:500] if len(descriptions[0]) > 500 else descriptions[0]
        parts.append(f"Description: {desc_text}")

    return " | ".join(parts)


def _entity_to_enriched_text(entity: Dict[str, Any]) -> str:
    """Enriched natural language sentence format."""
    sentences = []

    name = entity.get("caption") or entity.get("name") or "Unknown entity"
    schema = entity.get("schema", "Entity")
    props = entity.get("properties", {})
    datasets = entity.get("datasets", [])

    # Determine entity type description
    type_desc = {
        "Person": "individual",
        "Organization": "organization",
        "Company": "company",
        "LegalEntity": "legal entity",
        "PublicBody": "public body",
    }.get(schema, "entity")

    # Build main sentence
    countries = props.get("country", [])
    country_str = f"from {', '.join(countries)}" if countries else ""

    # Check for sanctions/PEP status
    topics = props.get("topics", []) + props.get("program", [])
    topics_lower = " ".join(topics).lower()
    datasets_lower = " ".join(datasets).lower()
    all_text = topics_lower + " " + datasets_lower

    status_parts = []
    if any(x in all_text for x in ["sanction", "ofac", "sdn"]):
        status_parts.append("sanctioned")
    if any(x in all_text for x in ["pep", "politically exposed"]):
        status_parts.append("politically exposed")
    if any(x in all_text for x in ["criminal", "crime", "fraud"]):
        status_parts.append("associated with criminal activity")

    status_str = ", ".join(status_parts) if status_parts else "of interest"

    # Main sentence
    main_sentence = f"{name} is a {status_str} {type_desc} {country_str}".strip()
    main_sentence = main_sentence.rstrip() + "."
    sentences.append(main_sentence)

    # Add aliases
    aliases = props.get("alias", []) or props.get("weakAlias", [])
    if aliases:
        alias_str = ", ".join(aliases[:5])
        sentences.append(f"Also known as: {alias_str}.")

    # Add position for PEPs
    positions = props.get("position", [])
    if positions:
        sentences.append(f"Held position(s): {', '.join(positions[:3])}.")

    # Add listing information
    if datasets:
        sentences.append(f"Listed under: {', '.join(datasets[:3])}.")

    # Add notes/reasons
    notes = props.get("notes", []) or props.get("summary", [])
    if notes:
        note_text = notes[0][:300] if len(notes[0]) > 300 else notes[0]
        sentences.append(note_text)

    return " ".join(sentences)


def _entity_to_narrative_text(entity: Dict[str, Any]) -> str:
    """Full narrative paragraph format for high-quality training data."""
    name = entity.get("caption") or entity.get("name") or "Unknown entity"
    schema = entity.get("schema", "Entity")
    props = entity.get("properties", {})
    datasets = entity.get("datasets", [])

    # Determine entity type
    type_desc = {
        "Person": "individual",
        "Organization": "organization",
        "Company": "company",
        "LegalEntity": "legal entity",
        "PublicBody": "government body",
    }.get(schema, "entity")

    paragraphs = []

    # First paragraph: identification
    countries = props.get("country", [])
    birth_dates = props.get("birthDate", [])
    nationalities = props.get("nationality", [])

    id_parts = [f"{name} is a {type_desc}"]
    if countries:
        id_parts.append(f"associated with {', '.join(countries)}")
    if birth_dates:
        id_parts.append(f"born {birth_dates[0]}")
    if nationalities:
        id_parts.append(f"with {', '.join(nationalities)} nationality")

    paragraphs.append(". ".join(id_parts) + ".")

    # Second paragraph: why they are listed
    topics = props.get("topics", []) + props.get("program", [])
    if datasets or topics:
        listing_parts = []
        if datasets:
            listing_parts.append(f"This {type_desc} appears in the following databases: {', '.join(datasets[:5])}")
        if topics:
            listing_parts.append(f"Relevant topics include: {', '.join(topics[:5])}")
        paragraphs.append(". ".join(listing_parts) + ".")

    # Third paragraph: additional details
    notes = props.get("notes", []) or props.get("summary", []) or props.get("description", [])
    if notes:
        paragraphs.append(notes[0][:500])

    # Fourth paragraph: aliases for entity matching
    aliases = props.get("alias", []) or props.get("weakAlias", [])
    if aliases:
        paragraphs.append(f"This {type_desc} may also be known by the following names: {', '.join(aliases[:10])}.")

    return "\n\n".join(paragraphs)


def classify_entity_risk(entity: Dict[str, Any]) -> List[str]:
    """Classify entity into risk categories.

    Args:
        entity: OpenSanctions entity.

    Returns:
        List of risk categories.
    """
    risks = []

    # Check topics/programs
    props = entity.get("properties", {})
    topics = props.get("topics", []) + props.get("program", [])
    topics_lower = [t.lower() for t in topics]

    # Check datasets
    datasets = entity.get("datasets", [])
    datasets_lower = [d.lower() for d in datasets]

    all_text = " ".join(topics_lower + datasets_lower)

    for category, keywords in RISK_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in all_text:
                risks.append(category)
                break

    return list(set(risks))


@DatasetRegistry.register(DatasetSource.OPENSANCTIONS)
class OpenSanctionsLoader(BaseTextDataset):
    """Loader for OpenSanctions entity data.

    OpenSanctions aggregates data from 321+ global sources including:
    - Official sanctions lists (OFAC, EU, UN, UK)
    - PEP databases
    - Criminal/crime databases
    - Corporate registries

    The text representations are useful for:
    - Entity matching training
    - Risk classification
    - Name normalization
    - Sanctions reasoning

    Token capping: OpenSanctions has ~500K short entity descriptions which can
    dominate the training token distribution. Use `max_tokens` to cap the
    contribution to a reasonable fraction of the total corpus.
    """

    source = DatasetSource.OPENSANCTIONS

    def __init__(
        self,
        config: DatasetConfig,
        dataset_name: str = "default",
        include_schemas: Optional[Set[str]] = None,
        max_aliases: int = 5,
        template_mode: str = TemplateMode.SIMPLE,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize OpenSanctions loader.

        Args:
            config: Dataset configuration.
            dataset_name: Which dataset to load (default, sanctions, peps, crime).
            include_schemas: Entity schemas to include (None for all relevant).
            max_aliases: Maximum aliases to include in text.
            template_mode: Text generation mode (simple, enriched, narrative).
                - simple: Pipe-delimited fields (short, ~50 tokens)
                - enriched: Natural language sentences (~100-150 tokens)
                - narrative: Full paragraphs (~200-400 tokens)
            max_tokens: Maximum total tokens to include from this source.
                Useful for capping OpenSanctions contribution to 5-10% of corpus.
                None means no limit.
        """
        super().__init__(config)
        self.dataset_name = dataset_name
        self.include_schemas = include_schemas or RELEVANT_SCHEMAS
        self.max_aliases = max_aliases
        self.template_mode = template_mode
        self.max_tokens = max_tokens
        self._total_tokens = 0

        if dataset_name not in OPENSANCTIONS_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self._data_url = OPENSANCTIONS_DATASETS[dataset_name]

    def download(self) -> bool:
        """Download OpenSanctions data.

        Returns:
            True if data is available.
        """
        # Check for local files
        json_files = list(self.config.data_dir.glob("*.json"))
        jsonl_files = list(self.config.data_dir.glob("*.jsonl"))
        csv_files = list(self.config.data_dir.glob("*.csv"))

        if json_files or jsonl_files or csv_files:
            logger.info(f"Found local OpenSanctions data")
            return True

        # Download from OpenSanctions
        if self.config.auto_download:
            return self._download_data()

        logger.warning(
            f"OpenSanctions data not found at {self.config.data_dir}\n"
            f"Download from: {self._data_url}"
        )
        return False

    def _download_data(self) -> bool:
        """Download data from OpenSanctions."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading OpenSanctions {self.dataset_name}...")
            response = requests.get(self._data_url, timeout=120, stream=True)
            response.raise_for_status()

            # Determine output path
            local_path = self.config.data_dir / f"opensanctions_{self.dataset_name}.json"

            # Write content
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download: {e}")
            return False

    def load(self) -> List[TextSample]:
        """Load OpenSanctions data.

        Returns:
            List of TextSample objects for entities.
        """
        if not self.download():
            return []

        samples = []

        # Try different file formats
        for json_file in self.config.data_dir.glob("*.json"):
            samples.extend(self._load_ftm_json(json_file))

        for jsonl_file in self.config.data_dir.glob("*.jsonl"):
            samples.extend(self._load_jsonl(jsonl_file))

        for csv_file in self.config.data_dir.glob("*.csv"):
            samples.extend(self._load_csv(csv_file))

        logger.info(f"Loaded {len(samples)} OpenSanctions samples")
        return samples

    def _load_ftm_json(self, json_path: Path) -> List[TextSample]:
        """Load from Follow The Money JSON format."""
        samples = []

        try:
            # The file might be a JSON array or newline-delimited JSON
            content = json_path.read_text(encoding="utf-8")

            # Try as JSON array first
            try:
                entities = json.loads(content)
                if isinstance(entities, dict):
                    entities = [entities]
            except json.JSONDecodeError:
                # Try as newline-delimited
                entities = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            entities.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            for idx, entity in enumerate(entities):
                sample = self._parse_entity(entity, idx)
                if sample:
                    samples.append(sample)

                if self.config.max_samples and len(samples) >= self.config.max_samples:
                    break

        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")

        return samples

    def _load_jsonl(self, jsonl_path: Path) -> List[TextSample]:
        """Load from JSONL format."""
        samples = []

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entity = json.loads(line)
                        sample = self._parse_entity(entity, idx)
                        if sample:
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except Exception as e:
            logger.error(f"Error loading {jsonl_path}: {e}")

        return samples

    def _load_csv(self, csv_path: Path) -> List[TextSample]:
        """Load from CSV format."""
        samples = []

        try:
            import csv as csv_module

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)

                for idx, row in enumerate(reader):
                    # Convert CSV row to entity-like structure
                    entity = {
                        "id": row.get("id"),
                        "caption": row.get("caption") or row.get("name"),
                        "schema": row.get("schema") or row.get("type"),
                        "properties": {
                            "country": [row.get("countries")] if row.get("countries") else [],
                            "program": [row.get("datasets")] if row.get("datasets") else [],
                        },
                        "datasets": row.get("datasets", "").split(";") if row.get("datasets") else [],
                    }

                    sample = self._parse_entity(entity, idx)
                    if sample:
                        samples.append(sample)

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")

        return samples

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4

    def _parse_entity(self, entity: Dict[str, Any], idx: int) -> Optional[TextSample]:
        """Parse entity into TextSample."""
        # Filter by schema
        schema = entity.get("schema", "")
        if self.include_schemas and schema not in self.include_schemas:
            return None

        # Check token cap before processing
        if self.max_tokens is not None and self._total_tokens >= self.max_tokens:
            return None

        # Convert to text using selected template mode
        text = entity_to_text(entity, template_mode=self.template_mode)
        if len(text) < self.config.min_text_length:
            return None

        # Check if adding this would exceed token cap
        estimated_tokens = self._estimate_tokens(text)
        if self.max_tokens is not None:
            if self._total_tokens + estimated_tokens > self.max_tokens:
                return None
            self._total_tokens += estimated_tokens

        # Classify risk
        risk_categories = classify_entity_risk(entity)

        # Determine category
        if "pep" in risk_categories:
            category = TextCategory.PEP_PROFILE
        elif "sanctions" in risk_categories:
            category = TextCategory.SANCTIONS_REASON
        else:
            category = TextCategory.ENTITY_DESCRIPTION

        # Determine label
        if risk_categories:
            label = FraudLabel.SUSPICIOUS
        else:
            label = FraudLabel.UNKNOWN

        # Extract entity names for the entities field
        props = entity.get("properties", {})
        aliases = props.get("alias", [])[:self.max_aliases]
        entity_names = [entity.get("caption", "")] + aliases
        entity_names = [n for n in entity_names if n]

        sample = TextSample(
            text=text,
            source=DatasetSource.OPENSANCTIONS,
            sample_id=f"opensanctions_{self.dataset_name}_{idx}",
            category=category,
            label=label,
            entities=entity_names,
            typologies=risk_categories,
            metadata={
                "entity_id": entity.get("id"),
                "schema": schema,
                "datasets": entity.get("datasets", []),
                "risk_categories": risk_categories,
                "countries": props.get("country", []),
            },
        )

        return sample

    def iter_entities(self) -> Generator[Dict[str, Any], None, None]:
        """Iterate through raw entity records.

        Yields:
            Raw entity dictionaries.
        """
        for json_file in self.config.data_dir.glob("*.json"):
            try:
                content = json_file.read_text(encoding="utf-8")
                try:
                    entities = json.loads(content)
                    if isinstance(entities, dict):
                        entities = [entities]
                    for entity in entities:
                        yield entity
                except json.JSONDecodeError:
                    for line in content.split("\n"):
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")

    def get_by_risk_category(self, category: str) -> List[TextSample]:
        """Get entities in a specific risk category.

        Args:
            category: Risk category (sanctions, pep, crime, terrorism).

        Returns:
            Filtered samples.
        """
        return [
            s for s in self.get_samples()
            if category in s.metadata.get("risk_categories", [])
        ]

    def get_by_country(self, country: str) -> List[TextSample]:
        """Get entities from a specific country.

        Args:
            country: Country code or name.

        Returns:
            Filtered samples.
        """
        country_lower = country.lower()
        return [
            s for s in self.get_samples()
            if any(country_lower in c.lower() for c in s.metadata.get("countries", []))
        ]

    def get_peps(self) -> List[TextSample]:
        """Get PEP (Politically Exposed Persons) entities."""
        return [s for s in self.get_samples() if s.category == TextCategory.PEP_PROFILE]

    def get_sanctioned_entities(self) -> List[TextSample]:
        """Get sanctioned entities."""
        return [s for s in self.get_samples() if s.category == TextCategory.SANCTIONS_REASON]

    def get_country_distribution(self) -> Dict[str, int]:
        """Get distribution of entities by country."""
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            for country in sample.metadata.get("countries", []):
                counts[country] = counts.get(country, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_dataset_distribution(self) -> Dict[str, int]:
        """Get distribution by source dataset."""
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            for dataset in sample.metadata.get("datasets", []):
                counts[dataset] = counts.get(dataset, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
