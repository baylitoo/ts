"""Feature engineering helpers shared across datasets."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categoricals(
    df: pd.DataFrame,
    columns: Iterable[str],
    encoders: Dict[str, LabelEncoder] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns and return fitted encoders."""
    encoders = {} if encoders is None else encoders
    result = df.copy()
    for col in columns:
        encoder = encoders.get(col, LabelEncoder())
        result[col] = encoder.fit_transform(result[col].astype(str))
        encoders[col] = encoder
    return result, encoders


def scale_numeric(
    df: pd.DataFrame,
    columns: Iterable[str],
    scaler: StandardScaler | None = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric columns using StandardScaler."""
    scaler = scaler or StandardScaler()
    result = df.copy()
    result[list(columns)] = scaler.fit_transform(result[list(columns)])
    return result, scaler


def add_cyclical_time_features(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Append sine/cosine representations for periodic temporal features."""
    values = df[column].fillna(0).astype(float)
    radians = 2 * np.pi * values / period
    df[f"{column}_sin"] = np.sin(radians)
    df[f"{column}_cos"] = np.cos(radians)
    return df


def log_safe(series: pd.Series, offset: float = 1.0) -> pd.Series:
    """Compute log1p on absolute values with configurable offset."""
    return np.log(series.abs() + offset)
