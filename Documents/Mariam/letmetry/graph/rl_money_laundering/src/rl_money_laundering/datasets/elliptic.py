"""Elliptic Bitcoin transaction dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseGraphDataset, DatasetSplits


@dataclass(slots=True)
class EllipticConfig:
    """Configuration for :class:`EllipticLoader`."""

    data_dir: Path | str
    target_column: str = "label_illicit"
    time_column: str = "time_step"
    validation_time_start: int = 35
    test_time_start: int = 42
    include_unlabeled: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)


class EllipticLoader(BaseGraphDataset):
    """Loader and preprocessor for Elliptic transaction graphs."""

    FEATURES_FILE = "elliptic_txs_features.csv"
    CLASSES_FILE = "elliptic_txs_classes.csv"
    EDGES_FILE = "elliptic_txs_edgelist.csv"

    def __init__(self, config: EllipticConfig):
        super().__init__(config.data_dir)
        self.config = config

        self.features_df: pd.DataFrame | None = None
        self.classes_df: pd.DataFrame | None = None
        self.edges_df: pd.DataFrame | None = None
        self.transactions: pd.DataFrame | None = None
        self.scaler: StandardScaler | None = None

        expected = [
            self.config.data_dir / self.FEATURES_FILE,
            self.config.data_dir / self.CLASSES_FILE,
            self.config.data_dir / self.EDGES_FILE,
        ]
        self.ensure_files_exist(expected)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load features, classes, and edges CSVs."""
        self.logger.info("Loading Elliptic dataset from %s", self.config.data_dir)

        features = pd.read_csv(self.config.data_dir / self.FEATURES_FILE, header=None)
        feature_total = features.shape[1] - 2
        feature_columns = [f"feature_{i}" for i in range(1, feature_total + 1)]
        features.columns = ["txId", self.config.time_column] + feature_columns

        classes = pd.read_csv(self.config.data_dir / self.CLASSES_FILE)
        classes.columns = ["txId", "class"]
        classes["label_illicit"] = (classes["class"] == "1").astype(int)
        classes["label_licit"] = (classes["class"] == "2").astype(int)
        classes["is_labeled"] = classes["class"].isin(["1", "2"]).astype(int)

        edges = pd.read_csv(self.config.data_dir / self.EDGES_FILE, names=["src", "dst"], header=None)

        merged = features.merge(classes, on="txId", how="left")
        merged["class"].fillna("unknown", inplace=True)
        merged["label_illicit"].fillna(0, inplace=True)
        merged["label_licit"].fillna(0, inplace=True)
        merged["is_labeled"].fillna(0, inplace=True)

        self.features_df = features
        self.classes_df = classes
        self.edges_df = edges
        self.transactions = merged

        self.logger.info("Loaded %s transactions, %s edges", len(merged), len(edges))
        return features, classes, edges

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def feature_pipeline(
        self,
        df: Optional[pd.DataFrame] = None,
        scale_features: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if df is None:
            if self.transactions is None:
                raise RuntimeError("Call load_raw() before feature_pipeline().")
            df = self.transactions

        working = df.copy()
        if not self.config.include_unlabeled:
            working = working[working["is_labeled"] == 1].copy()

        feature_cols = [col for col in working.columns if col.startswith("feature_")]
        if not feature_cols:
            raise KeyError("Feature columns missing; ensure load_raw() executed successfully.")

        # Temporal context features
        time_counts = working.groupby(self.config.time_column)["txId"].transform("count")
        working["time_tx_density"] = np.log1p(time_counts)

        illicit_ratio = working.groupby(self.config.time_column)["label_illicit"].transform(
            lambda s: (s.sum() + 1) / (len(s) + 2)
        )
        working["time_illicit_ratio"] = illicit_ratio

        local_cols = [col for col in feature_cols if int(col.split("_")[1]) <= 94]
        if local_cols:
            working["local_mean"] = working[local_cols].mean(axis=1)
            working["local_std"] = working[local_cols].std(axis=1).fillna(0.0)

        agg_cols = [col for col in feature_cols if col not in local_cols]
        if agg_cols:
            working["agg_mean"] = working[agg_cols].mean(axis=1)
            working["agg_std"] = working[agg_cols].std(axis=1).fillna(0.0)

        numeric_cols = feature_cols + [
            "time_tx_density",
            "time_illicit_ratio",
            "local_mean",
            "local_std",
            "agg_mean",
            "agg_std",
        ]
        numeric_cols = [col for col in numeric_cols if col in working.columns]

        if scale_features:
            self.scaler = self.scaler or StandardScaler()
            working[numeric_cols] = self.scaler.fit_transform(working[numeric_cols])

        target = working[self.config.target_column].astype(int)
        keep_columns = numeric_cols + [self.config.time_column, self.config.target_column, "txId"]
        if "is_labeled" in working.columns:
            keep_columns.append("is_labeled")
        working = working[keep_columns]

        features = working.drop(columns=[self.config.target_column])
        return features, target

    # ------------------------------------------------------------------
    # Splits and statistics
    # ------------------------------------------------------------------
    def create_temporal_splits(self) -> DatasetSplits:
        if self.transactions is None:
            raise RuntimeError("Call load_raw() before create_temporal_splits().")

        df = self.transactions.copy()
        if not self.config.include_unlabeled:
            df = df[df["is_labeled"] == 1].copy()

        train_mask = df[self.config.time_column] < self.config.validation_time_start
        val_mask = (
            (df[self.config.time_column] >= self.config.validation_time_start)
            & (df[self.config.time_column] < self.config.test_time_start)
        )
        test_mask = df[self.config.time_column] >= self.config.test_time_start

        train = df.loc[train_mask].reset_index(drop=True)
        val = df.loc[val_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)

        return DatasetSplits(train=train, validation=val, test=test)

    def describe(self) -> Dict[str, int | float]:
        if self.transactions is None:
            raise RuntimeError("Call load_raw() before describe().")
        df = self.transactions
        summary: Dict[str, int | float] = {
            "rows": int(len(df)),
            "edges": int(len(self.edges_df)) if self.edges_df is not None else 0,
            "time_steps": int(df[self.config.time_column].nunique()),
            "illicit": int(df["label_illicit"].sum()),
            "licit": int(df["label_licit"].sum()),
            "unknown": int((df["label_illicit"] + df["label_licit"] == 0).sum()),
        }
        return summary

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(
        self,
        include_unknown: Optional[bool] = None,
        processed_features: Optional[pd.DataFrame] = None,
    ) -> nx.DiGraph:
        if self.transactions is None or self.edges_df is None:
            raise RuntimeError("Call load_raw() before build_graph().")

        include_unknown = (
            self.config.include_unlabeled if include_unknown is None else include_unknown
        )

        df = self.transactions
        node_mask = df["is_labeled"].astype(bool) | include_unknown
        attributes = df.loc[node_mask, ["txId", self.config.time_column, "label_illicit"]]

        graph = nx.DiGraph()
        allowed_nodes: set[str] | None = None
        processed: Optional[pd.DataFrame] = None
        feature_cols: list[str] = []

        if processed_features is not None:
            processed = processed_features.copy()

            def _normalize_txid(value: Any) -> str:
                if pd.isna(value):
                    return ""
                if isinstance(value, (int, np.integer)):
                    return str(int(value))
                if isinstance(value, (float, np.floating)):
                    return str(int(value))
                text = str(value).strip()
                if text.endswith(".0"):
                    text = text[:-2]
                return text

            processed["txId"] = processed["txId"].apply(_normalize_txid)
            allowed_nodes = set(processed["txId"].tolist())

            feature_cols = [
                col
                for col in processed.columns
                if col.startswith("feature_")
                or col
                in [
                    "time_tx_density",
                    "time_illicit_ratio",
                    "local_mean",
                    "local_std",
                    "agg_mean",
                    "agg_std",
                ]
            ]

        for tx_id, time_step, label in attributes.itertuples(index=False):
            node_id = str(tx_id)
            if allowed_nodes is not None and node_id not in allowed_nodes:
                continue
            graph.add_node(node_id, time=int(time_step), label=int(label))

        if processed is not None:
            sample_graph_nodes = list(graph.nodes())[:3]
            sample_df_ids = processed["txId"].iloc[:3].tolist()
            self.logger.info("DEBUG: Sample graph nodes: %s", sample_graph_nodes)
            self.logger.info("DEBUG: Sample df txIds: %s", sample_df_ids)
            self.logger.info(
                "DEBUG: Total nodes in graph: %s, rows in df: %s",
                len(graph.nodes()),
                len(processed),
            )

            nodes_updated = 0
            nodes_checked = 0
            for _, row in processed.iterrows():
                node_id = row["txId"]
                nodes_checked += 1

                if nodes_checked <= 3:
                    self.logger.info(
                        "DEBUG: Checking node %s, exists: %s",
                        node_id,
                        node_id in graph,
                    )

                if node_id in graph:
                    for col in feature_cols:
                        graph.nodes[node_id][col] = float(row[col])
                    nodes_updated += 1

                    if nodes_updated == 1:
                        num_attrs = len(graph.nodes[node_id])
                        self.logger.info("DEBUG: First node %s now has %s attributes", node_id, num_attrs)

            self.logger.info("Attached %s features to %s nodes", len(feature_cols), nodes_updated)

            if nodes_updated > 0:
                test_node = list(graph.nodes())[0]
                test_attrs = len(graph.nodes[test_node])
                if test_attrs > 2:
                    self.logger.info("✓ Verification: Node %s has %s attributes", test_node, test_attrs)
                else:
                    self.logger.warning("✗ Verification FAILED: Node %s only has %s attributes!", test_node, test_attrs)

        for src, dst in self.edges_df.itertuples(index=False):
            src_id = str(src)
            dst_id = str(dst)
            if allowed_nodes is not None and (src_id not in allowed_nodes or dst_id not in allowed_nodes):
                continue
            if src_id in graph and dst_id in graph:
                fraud_flag = int(
                    graph.nodes[src_id].get("label", 0)
                    or graph.nodes[dst_id].get("label", 0)
                )
                src_time = graph.nodes[src_id].get("time", 0)
                dst_time = graph.nodes[dst_id].get("time", 0)
                edge_time = float(max(src_time, dst_time))
                graph.add_edge(
                    src_id,
                    dst_id,
                    is_fraud=fraud_flag,
                    is_money_laundering=fraud_flag,
                    isFraud=fraud_flag,
                    time=edge_time,
                    timestamp=edge_time,
                    time_step=int(edge_time),
                )

        self.logger.info(
            "Graph built with %s nodes, %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph


__all__ = ["EllipticConfig", "EllipticLoader"]
