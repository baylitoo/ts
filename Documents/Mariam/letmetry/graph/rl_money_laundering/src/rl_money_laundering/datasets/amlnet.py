"""AMLNet dataset loader and feature engineering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import BaseGraphDataset, DatasetSplits
from .features import add_cyclical_time_features, encode_categoricals, log_safe, scale_numeric

AML_CATEGORICAL_COLUMNS: Tuple[str, ...] = ("type", "category", "laundering_typology")
TEMPORAL_PERIODS: Dict[str, int] = {
    "hour": 24,
    "day_of_week": 7,
    "day_of_month": 31,
    "month": 12,
}


@dataclass(slots=True)
class AMLNetConfig:
    """Configuration options for :class:`AMLNetLoader`."""

    csv_path: Path | str
    target_column: str = "isMoneyLaundering"
    parse_metadata: bool = False
    metadata_prefixes: Tuple[str, ...] = ("risk_indicators",)
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 7
    max_edges: Optional[int] = None

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)


class AMLNetLoader(BaseGraphDataset):
    """Loader for the AMLNet synthetic transaction dataset."""

    def __init__(self, config: AMLNetConfig):
        super().__init__(config.csv_path.parent)
        self.config = config
        self.ensure_files_exist([config.csv_path])

        self.transactions: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.targets: pd.Series | None = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # Loading and metadata handling
    # ------------------------------------------------------------------
    def load_raw(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load the raw CSV file into memory."""
        self.logger.info("Loading AMLNet csv from %s", self.config.csv_path)
        df = pd.read_csv(self.config.csv_path, nrows=nrows, low_memory=False)

        if self.config.parse_metadata and "metadata" in df.columns:
            meta_df = self._expand_metadata(df["metadata"])
            df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)

        self.transactions = df
        self.logger.info("Loaded %s rows", len(df))
        return df

    def _expand_metadata(self, series: pd.Series) -> pd.DataFrame:
        """Expand the JSON metadata column into a flat dataframe."""
        parsed = series.fillna("{}").apply(
            lambda val: json.loads(val) if isinstance(val, str) and val.strip() else {}
        )
        normalized = pd.json_normalize(parsed)

        if self.config.metadata_prefixes:
            keep_columns: List[str] = []
            for prefix in self.config.metadata_prefixes:
                keep_columns.extend(
                    col for col in normalized.columns if col.startswith(prefix)
                )
            normalized = normalized[keep_columns]

        normalized.columns = [f"meta_{col.replace('.', '_')}" for col in normalized.columns]
        return normalized

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def feature_pipeline(
        self,
        df: Optional[pd.DataFrame] = None,
        scale_numeric_features: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate model-ready features and targets."""
        if df is None:
            if self.transactions is None:
                raise RuntimeError("Call load_raw() before feature_pipeline().")
            df = self.transactions

        working = df.copy()

        # Ensure expected columns exist
        required_columns = {
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "step",
            self.config.target_column,
            "nameOrig",
            "nameDest",
        }
        missing = required_columns - set(working.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Amount-based features
        working["log_amount"] = log_safe(working["amount"])
        working["balance_delta"] = working["newbalanceOrig"] - working["oldbalanceOrg"]
        working["abs_balance_delta"] = working["balance_delta"].abs()
        working["amount_ratio_balance"] = working["amount"] / (working["oldbalanceOrg"].abs() + 1.0)

        # Customer + merchant aggregation features
        working["origin_tx_count"] = working.groupby("nameOrig")["step"].transform("count")
        working["origin_amount_mean"] = working.groupby("nameOrig")["amount"].transform("mean")
        working["origin_amount_ratio"] = working["amount"] / (working["origin_amount_mean"] + 1.0)

        working["dest_tx_count"] = working.groupby("nameDest")["step"].transform("count")
        working["dest_amount_mean"] = working.groupby("nameDest")["amount"].transform("mean")
        working["dest_amount_ratio"] = working["amount"] / (working["dest_amount_mean"] + 1.0)

        # Temporal features (cyclical encoding)
        for column, period in TEMPORAL_PERIODS.items():
            if column in working.columns:
                add_cyclical_time_features(working, column, period)

        # Encode categorical attributes
        categorical_cols = [col for col in AML_CATEGORICAL_COLUMNS if col in working.columns]
        if categorical_cols:
            working, self.encoders = encode_categoricals(working, categorical_cols, self.encoders)

        # Select numeric columns for scaling
        numeric_candidates = working.select_dtypes(include=["number"]).columns.tolist()
        protected_cols = {self.config.target_column, "step"}
        numeric_features = [col for col in numeric_candidates if col not in protected_cols]

        if scale_numeric_features and numeric_features:
            working, self.scaler = scale_numeric(working, numeric_features, self.scaler)

        # Targets and feature frame
        targets = working[self.config.target_column].astype(int)
        identifiers = {"nameOrig", "nameDest", self.config.target_column}
        features = working.drop(columns=[col for col in working.columns if col in identifiers])

        self.features = features
        self.targets = targets
        return features, targets

    # ------------------------------------------------------------------
    # Splits and statistics
    # ------------------------------------------------------------------
    def create_splits(
        self,
        df: Optional[pd.DataFrame] = None,
        stratify: bool = True,
    ) -> DatasetSplits:
        """Create train/validation/test splits."""
        if df is None:
            if self.transactions is None:
                raise RuntimeError("Call load_raw() before create_splits().")
            df = self.transactions

        target = df[self.config.target_column]
        stratify_target = target if stratify and target.nunique() > 1 else None

        train_val, test = train_test_split(
            df,
            test_size=self.config.test_size,
            stratify=stratify_target,
            random_state=self.config.random_state,
        )

        val_size = self.config.validation_size / (1.0 - self.config.test_size)
        stratify_train = (
            train_val[self.config.target_column] if stratify_target is not None else None
        )
        train, validation = train_test_split(
            train_val,
            test_size=val_size,
            stratify=stratify_train,
            random_state=self.config.random_state,
        )

        return DatasetSplits(train=train, validation=validation, test=test)

    def describe(self) -> Dict[str, float | int | Dict[str, int]]:
        """Return key dataset statistics."""
        if self.transactions is None:
            raise RuntimeError("Call load_raw() before describe().")
        df = self.transactions
        stats: Dict[str, float | int | Dict[str, int]] = {
            "rows": int(len(df)),
            "fraudulent": int(df[self.config.target_column].sum()),
            "fraud_rate": float(df[self.config.target_column].mean()),
            "customers": int(df["nameOrig"].nunique()),
            "merchants": int(df["nameDest"].nunique()),
        }
        if "type" in df.columns:
            stats["payment_types"] = df["type"].value_counts().to_dict()
        if "laundering_typology" in df.columns:
            stats["typologies"] = df["laundering_typology"].value_counts().to_dict()
        return stats

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def build_graph(
        self,
        df: Optional[pd.DataFrame] = None,
        weight_column: str = "amount",
        max_edges: Optional[int] = None,
    ) -> nx.DiGraph:
        """Construct a directed transaction graph."""
        if df is None:
            if self.transactions is None:
                raise RuntimeError("Call load_raw() before build_graph().")
            df = self.transactions

        if weight_column not in df.columns:
            raise KeyError(f"Column {weight_column} not present in dataframe")

        subset = df if max_edges is None else df.head(max_edges)
        graph = nx.DiGraph()

        nodes = set(subset["nameOrig"]).union(set(subset["nameDest"]))
        graph.add_nodes_from(nodes)

        # Include fraud_probability if available
        edge_columns = ["nameOrig", "nameDest", "amount", "step", weight_column, self.config.target_column]
        if "fraud_probability" in subset.columns:
            edge_columns.append("fraud_probability")

        edge_iter = (
            (
                row.nameOrig,
                row.nameDest,
                {
                    "amount": float(row.amount),
                    "step": int(row.step),
                    "timestamp": float(row.step),
                    "time": float(row.step),
                    "weight": float(getattr(row, weight_column)),
                    "is_money_laundering": int(getattr(row, self.config.target_column)),
                    "isFraud": int(getattr(row, self.config.target_column)),  # Add isFraud alias
                    "fraud_probability": float(getattr(row, "fraud_probability", 0.5)),  # Include fraud_probability
                },
            )
            for row in subset[edge_columns].itertuples(index=False)
        )
        graph.add_edges_from(edge_iter)

        self.logger.info(
            "Graph built with %s nodes, %s edges", graph.number_of_nodes(), graph.number_of_edges()
        )
        return graph

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def class_weights(self) -> Dict[int, float]:
        """Compute inverse frequency class weights."""
        if self.targets is None:
            raise RuntimeError("Run feature_pipeline() before class_weights().")
        counts = self.targets.value_counts()
        total = counts.sum()
        return {int(label): total / (len(counts) * count) for label, count in counts.items() if count > 0}


__all__ = ["AMLNetConfig", "AMLNetLoader"]
