"""
GDELT news data loader for adverse media.

Queries the GDELT (Global Database of Events, Language, and Tone) for
news articles related to financial crime, money laundering, sanctions, etc.

Data source: https://www.gdeltproject.org/
BigQuery: https://console.cloud.google.com/bigquery?p=gdelt-bq

GDELT provides:
- Global news coverage in 65+ languages
- Event extraction and classification
- Sentiment and tone analysis
- Entity mentions
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set
from urllib.parse import quote, urlencode

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


# GDELT API endpoints
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TV_API = "https://api.gdeltproject.org/api/v2/tv/tv"
GDELT_GEO_API = "https://api.gdeltproject.org/api/v2/geo/geo"

# BigQuery dataset (requires GCP credentials)
GDELT_BQ_EVENTS = "gdelt-bq.gdeltv2.events"
GDELT_BQ_GKG = "gdelt-bq.gdeltv2.gkg"

# Financial crime keywords for querying
FINANCIAL_CRIME_KEYWORDS = [
    "money laundering",
    "financial fraud",
    "sanctions violation",
    "sanctions evasion",
    "shell company",
    "offshore account",
    "tax evasion",
    "bribery corruption",
    "embezzlement",
    "wire fraud",
    "bank fraud",
    "terrorist financing",
    "suspicious transaction",
    "ponzi scheme",
    "securities fraud",
    "insider trading",
    "market manipulation",
]

# GDELT themes related to financial crime
GDELT_THEMES = {
    "TAX_EVASION": "Tax evasion",
    "CORRUPTION": "Corruption",
    "ORGANIZED_CRIME": "Organized crime",
    "MONEY_LAUNDERING": "Money laundering",
    "FRAUD": "Fraud",
    "SANCTIONS": "Sanctions",
    "TERROR_FINANCING": "Terror financing",
    "FINANCIAL_CRIME": "Financial crime",
}


def build_gdelt_query(
    keywords: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    source_countries: Optional[List[str]] = None,
    max_records: int = 250,
    mode: str = "artlist",
    format: str = "json",
) -> str:
    """Build GDELT API query URL.

    Args:
        keywords: Search keywords.
        start_date: Start of date range.
        end_date: End of date range.
        source_countries: Filter by source country.
        max_records: Maximum records to return.
        mode: Query mode (artlist, timeline, etc.).
        format: Output format (json, csv).

    Returns:
        Query URL string.
    """
    params = {
        "query": " OR ".join(f'"{k}"' for k in keywords),
        "mode": mode,
        "maxrecords": max_records,
        "format": format,
    }

    if start_date:
        params["startdatetime"] = start_date.strftime("%Y%m%d%H%M%S")
    if end_date:
        params["enddatetime"] = end_date.strftime("%Y%m%d%H%M%S")
    if source_countries:
        params["sourcecountry"] = ",".join(source_countries)

    return f"{GDELT_DOC_API}?{urlencode(params)}"


def parse_gdelt_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a GDELT article record.

    Args:
        article: Raw GDELT article data.

    Returns:
        Normalized article dictionary.
    """
    return {
        "url": article.get("url"),
        "title": article.get("title"),
        "source": article.get("domain") or article.get("source"),
        "language": article.get("language"),
        "country": article.get("sourcecountry"),
        "date": article.get("seendate"),
        "tone": article.get("tone"),
        "themes": article.get("themes", "").split(";") if article.get("themes") else [],
        "organizations": article.get("organizations", "").split(";") if article.get("organizations") else [],
        "persons": article.get("persons", "").split(";") if article.get("persons") else [],
        "locations": article.get("locations", "").split(";") if article.get("locations") else [],
    }


@DatasetRegistry.register(DatasetSource.GDELT)
class GDELTLoader(BaseTextDataset):
    """Loader for GDELT adverse media data.

    GDELT monitors news worldwide and provides structured metadata
    about articles mentioning specific themes, entities, and events.

    Use cases for AML:
    - Adverse media screening
    - Entity risk monitoring
    - News-based supervision signals
    - Weak labels for pretraining

    Note: GDELT API has rate limits. For large-scale access,
    use BigQuery with GCP credentials.
    """

    source = DatasetSource.GDELT

    def __init__(
        self,
        config: DatasetConfig,
        keywords: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        days_back: int = 30,
        max_articles: int = 1000,
        fetch_content: bool = False,
    ) -> None:
        """Initialize GDELT loader.

        Args:
            config: Dataset configuration.
            keywords: Search keywords (default: financial crime terms).
            themes: GDELT themes to filter.
            days_back: Number of days to look back.
            max_articles: Maximum articles to fetch.
            fetch_content: Whether to fetch full article content.
        """
        super().__init__(config)
        self.keywords = keywords or FINANCIAL_CRIME_KEYWORDS[:5]  # Start with subset
        self.themes = themes
        self.days_back = days_back
        self.max_articles = max_articles
        self.fetch_content = fetch_content

    def download(self) -> bool:
        """Check for cached data or prepare for API queries.

        Returns:
            True if data is available or can be fetched.
        """
        # Check for cached files
        cache_files = (
            list(self.config.data_dir.glob("gdelt_*.json")) +
            list(self.config.data_dir.glob("gdelt_*.csv"))
        )

        if cache_files:
            logger.info(f"Found cached GDELT data: {[f.name for f in cache_files]}")
            return True

        # We can always try the API
        if self.config.auto_download:
            logger.info("GDELT data will be fetched via API")
            return True

        logger.warning(
            f"No cached GDELT data at {self.config.data_dir}\n"
            f"Enable auto_download to fetch from GDELT API"
        )
        return False

    def load(self) -> List[TextSample]:
        """Load GDELT data.

        Returns:
            List of TextSample objects from news articles.
        """
        samples = []

        # Load from cache first
        for json_file in self.config.data_dir.glob("gdelt_*.json"):
            samples.extend(self._load_cached_json(json_file))

        for csv_file in self.config.data_dir.glob("gdelt_*.csv"):
            samples.extend(self._load_cached_csv(csv_file))

        # Fetch from API if needed
        if not samples and self.config.auto_download:
            samples = self._fetch_from_api()

        logger.info(f"Loaded {len(samples)} GDELT samples")
        return samples

    def _fetch_from_api(self) -> List[TextSample]:
        """Fetch articles from GDELT API."""
        samples = []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)

        # Fetch in batches by keyword to avoid hitting limits
        articles_fetched = 0
        for keyword in self.keywords:
            if articles_fetched >= self.max_articles:
                break

            try:
                url = build_gdelt_query(
                    keywords=[keyword],
                    start_date=start_date,
                    end_date=end_date,
                    max_records=min(250, self.max_articles - articles_fetched),
                )

                logger.info(f"Querying GDELT for: {keyword}")
                response = requests.get(url, timeout=60)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        articles = data.get("articles", [])

                        for article in articles:
                            parsed = parse_gdelt_article(article)
                            sample = self._article_to_sample(parsed, articles_fetched)
                            if sample:
                                samples.append(sample)
                                articles_fetched += 1

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse GDELT response for {keyword}")

                else:
                    logger.warning(f"GDELT API returned {response.status_code}")

            except Exception as e:
                logger.error(f"Error fetching GDELT data: {e}")

        # Cache the results
        if samples:
            self._cache_results(samples)

        return samples

    def _article_to_sample(
        self,
        article: Dict[str, Any],
        idx: int,
    ) -> Optional[TextSample]:
        """Convert article to TextSample."""
        # Build text from available fields
        parts = []

        title = article.get("title")
        if title:
            parts.append(f"Title: {title}")

        source = article.get("source")
        if source:
            parts.append(f"Source: {source}")

        country = article.get("country")
        if country:
            parts.append(f"Country: {country}")

        # Include entities
        organizations = article.get("organizations", [])
        if organizations:
            parts.append(f"Organizations: {', '.join(organizations[:5])}")

        persons = article.get("persons", [])
        if persons:
            parts.append(f"Persons: {', '.join(persons[:5])}")

        themes = article.get("themes", [])
        if themes:
            parts.append(f"Themes: {', '.join(themes[:5])}")

        text = " | ".join(parts)

        if len(text) < self.config.min_text_length:
            return None

        # Parse date
        date = None
        date_str = article.get("date")
        if date_str:
            try:
                date = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                pass

        # Determine risk relevance
        indicators = []
        for theme in themes:
            theme_lower = theme.lower()
            for key, desc in GDELT_THEMES.items():
                if key.lower() in theme_lower:
                    indicators.append(desc)

        # Extract entities
        entities = organizations + persons
        entities = list(set(entities))[:10]

        sample = TextSample(
            text=text,
            source=DatasetSource.GDELT,
            sample_id=f"gdelt_{idx}",
            category=TextCategory.ADVERSE_MEDIA if indicators else TextCategory.NEWS_ARTICLE,
            label=FraudLabel.UNKNOWN,  # Weak supervision
            entities=entities,
            indicators=indicators,
            timestamp=date,
            metadata={
                "url": article.get("url"),
                "source_domain": source,
                "country": country,
                "tone": article.get("tone"),
                "themes": themes,
                "organizations": organizations,
                "persons": persons,
            },
        )

        return sample

    def _load_cached_json(self, json_path: Path) -> List[TextSample]:
        """Load from cached JSON file."""
        samples = []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if "text" in item:
                        # Already a TextSample format
                        try:
                            sample = TextSample.from_dict(item)
                            samples.append(sample)
                        except Exception:
                            pass
                    else:
                        # Raw article format
                        sample = self._article_to_sample(item, idx)
                        if sample:
                            samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading cached JSON {json_path}: {e}")

        return samples

    def _load_cached_csv(self, csv_path: Path) -> List[TextSample]:
        """Load from cached CSV file."""
        samples = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    article = {
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "source": row.get("domain") or row.get("source"),
                        "country": row.get("sourcecountry"),
                        "date": row.get("seendate"),
                        "themes": row.get("themes", "").split(";"),
                        "organizations": row.get("organizations", "").split(";"),
                        "persons": row.get("persons", "").split(";"),
                    }

                    sample = self._article_to_sample(article, idx)
                    if sample:
                        samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading cached CSV {csv_path}: {e}")

        return samples

    def _cache_results(self, samples: List[TextSample]) -> None:
        """Cache fetched results to disk."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = self.config.data_dir / f"gdelt_cache_{timestamp}.json"

        try:
            data = [s.to_dict() for s in samples]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(samples)} samples to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    def fetch_by_entity(
        self,
        entity_name: str,
        days_back: int = 30,
        max_articles: int = 100,
    ) -> List[TextSample]:
        """Fetch news articles mentioning a specific entity.

        Args:
            entity_name: Entity name to search.
            days_back: Days to look back.
            max_articles: Maximum articles.

        Returns:
            List of samples mentioning the entity.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        try:
            url = build_gdelt_query(
                keywords=[entity_name],
                start_date=start_date,
                end_date=end_date,
                max_records=max_articles,
            )

            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                return []

            data = response.json()
            samples = []

            for idx, article in enumerate(data.get("articles", [])):
                parsed = parse_gdelt_article(article)
                sample = self._article_to_sample(parsed, idx)
                if sample:
                    samples.append(sample)

            return samples

        except Exception as e:
            logger.error(f"Error fetching for entity {entity_name}: {e}")
            return []

    def get_adverse_media(self) -> List[TextSample]:
        """Get samples classified as adverse media."""
        return [
            s for s in self.get_samples()
            if s.category == TextCategory.ADVERSE_MEDIA
        ]

    def get_by_theme(self, theme: str) -> List[TextSample]:
        """Get samples containing a specific theme.

        Args:
            theme: GDELT theme string.

        Returns:
            Filtered samples.
        """
        theme_lower = theme.lower()
        return [
            s for s in self.get_samples()
            if any(theme_lower in t.lower() for t in s.metadata.get("themes", []))
        ]

    def get_by_organization(self, org: str) -> List[TextSample]:
        """Get samples mentioning an organization.

        Args:
            org: Organization name.

        Returns:
            Filtered samples.
        """
        org_lower = org.lower()
        return [
            s for s in self.get_samples()
            if any(org_lower in o.lower() for o in s.metadata.get("organizations", []))
        ]

    def get_theme_distribution(self) -> Dict[str, int]:
        """Get distribution of themes."""
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            for theme in sample.metadata.get("themes", []):
                counts[theme] = counts.get(theme, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:50])  # Top 50

    def get_country_distribution(self) -> Dict[str, int]:
        """Get distribution by source country."""
        counts: Dict[str, int] = {}
        for sample in self.get_samples():
            country = sample.metadata.get("country")
            if country:
                counts[country] = counts.get(country, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
