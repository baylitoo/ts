"""Common dataset loading utilities for transaction graph datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import logging
import networkx as nx  # type: ignore[import-untyped]
import pandas as pd


@dataclass(slots=True)
class DatasetSplits:
    """Container for train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the repository."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class BaseGraphDataset:
    """Abstract base class for transaction graph datasets."""

    def __init__(self, root: Path | str, logger: Optional[logging.Logger] = None) -> None:
        self.root = Path(root)
        self.logger = logger or get_logger(self.__class__.__name__)

    def load_raw(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # pragma: no cover - abstract style
        raise NotImplementedError

    def build_graph(self, *args: Any, **kwargs: Any) -> nx.DiGraph:  # pragma: no cover - abstract style
        raise NotImplementedError

    def feature_pipeline(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # pragma: no cover - abstract style
        raise NotImplementedError

    def ensure_files_exist(self, expected_files: Iterable[Path]) -> None:
        missing: Dict[str, Path] = {}
        for path in expected_files:
            if not path.exists():
                missing[path.name] = path
        if missing:
            joined = ", ".join(f"{name} -> {path}" for name, path in missing.items())
            raise FileNotFoundError(f"Dataset assets missing: {joined}")

    def describe_target(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        counts = df[target_column].value_counts(dropna=False)
        ratios = (counts / len(df)).round(4)
        return {
            "counts": counts.to_dict(),
            "ratios": ratios.to_dict(),
            "total": len(df),
        }
