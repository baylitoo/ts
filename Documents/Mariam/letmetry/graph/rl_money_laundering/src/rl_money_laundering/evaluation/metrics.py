"""
Core Evaluation Metrics for AML Detection

Comprehensive metrics suite for classification, ranking, and detection performance:

1. Classification Metrics:
   - Typology: Macro-F1, Micro-F1, per-class precision/recall
   - Suspiciousness: AUC-ROC, AUC-PR, F1 at various thresholds
   - Multi-label: Hamming loss, subset accuracy

2. Ranking Metrics:
   - NDCG@k (Normalized Discounted Cumulative Gain)
   - MRR (Mean Reciprocal Rank)
   - Precision@k, Recall@k

3. Detection Metrics:
   - Alert-to-SAR conversion rate
   - Investigation efficiency
   - False positive rate at fixed recall

References:
- Manning et al. (2008): Information Retrieval - ranking metrics
- Saito & Rehmsmeier (2015): Precision-Recall curves for imbalanced data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        classification_report,
        hamming_loss,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available - some metrics will not work")


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None

    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
        }
        if self.auc_roc is not None:
            result["auc_roc"] = self.auc_roc
        if self.auc_pr is not None:
            result["auc_pr"] = self.auc_pr
        if self.per_class_precision:
            result["per_class_precision"] = self.per_class_precision
            result["per_class_recall"] = self.per_class_recall
            result["per_class_f1"] = self.per_class_f1
        return result


@dataclass
class RankingMetrics:
    """Container for ranking metrics."""
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0  # Mean Average Precision

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"mrr": self.mrr, "map": self.map_score}
        for k, v in self.ndcg_at_k.items():
            result[f"ndcg@{k}"] = v
        for k, v in self.precision_at_k.items():
            result[f"precision@{k}"] = v
        for k, v in self.recall_at_k.items():
            result[f"recall@{k}"] = v
        return result


class MetricsComputer:
    """Compute comprehensive evaluation metrics.

    This class provides methods for computing:
    - Classification metrics (binary and multi-class)
    - Ranking metrics (for retrieval-style evaluation)
    - Threshold optimization

    Example:
        ```python
        from rl_money_laundering.evaluation import MetricsComputer

        computer = MetricsComputer()

        # Binary classification
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        y_scores = [0.1, 0.9, 0.4, 0.2, 0.8]

        metrics = computer.compute_classification_metrics(
            y_true, y_pred, y_scores
        )
        print(f"F1: {metrics.f1:.3f}, AUC-ROC: {metrics.auc_roc:.3f}")

        # Ranking evaluation
        ranking_metrics = computer.compute_ranking_metrics(
            y_true, y_scores, k_values=[5, 10, 20]
        )
        print(f"NDCG@10: {ranking_metrics.ndcg_at_k[10]:.3f}")
        ```
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize metrics computer.

        Args:
            class_names: Optional list of class names for per-class metrics
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn is required for MetricsComputer")

        self.class_names = class_names or ["normal", "fraud"]

    def compute_classification_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_scores: Optional[Union[np.ndarray, List]] = None,
        average: str = "binary",
    ) -> ClassificationMetrics:
        """Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities (for AUC)
            average: Averaging method ('binary', 'macro', 'micro', 'weighted')

        Returns:
            ClassificationMetrics with computed values
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic metrics
        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average=average, zero_division=0),
            recall=recall_score(y_true, y_pred, average=average, zero_division=0),
            f1=f1_score(y_true, y_pred, average=average, zero_division=0),
        )

        # Multi-class averages
        if len(np.unique(y_true)) > 2:
            metrics.macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics.micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            metrics.weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            metrics.macro_f1 = metrics.f1
            metrics.micro_f1 = metrics.f1
            metrics.weighted_f1 = metrics.f1

        # Confusion matrix
        metrics.confusion_matrix = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        unique_classes = np.unique(y_true)
        for i, cls in enumerate(unique_classes):
            cls_name = self.class_names[i] if i < len(self.class_names) else str(cls)
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)

            metrics.per_class_precision[cls_name] = precision_score(
                binary_true, binary_pred, zero_division=0
            )
            metrics.per_class_recall[cls_name] = recall_score(
                binary_true, binary_pred, zero_division=0
            )
            metrics.per_class_f1[cls_name] = f1_score(
                binary_true, binary_pred, zero_division=0
            )

        # AUC metrics (if scores provided)
        if y_scores is not None:
            y_scores = np.asarray(y_scores)
            try:
                # For binary classification
                if len(np.unique(y_true)) == 2:
                    metrics.auc_roc = roc_auc_score(y_true, y_scores)
                    metrics.auc_pr = average_precision_score(y_true, y_scores)
                else:
                    # Multi-class: use one-vs-rest
                    metrics.auc_roc = roc_auc_score(
                        y_true, y_scores, multi_class='ovr', average='macro'
                    )
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")

        return metrics

    def compute_ranking_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        k_values: List[int] = [5, 10, 20, 50, 100],
    ) -> RankingMetrics:
        """Compute ranking metrics.

        Args:
            y_true: True binary labels (1 = relevant/fraud)
            y_scores: Prediction scores (higher = more suspicious)
            k_values: K values for @k metrics

        Returns:
            RankingMetrics with NDCG, MRR, Precision@k, etc.
        """
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        metrics = RankingMetrics()

        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]

        # Compute metrics at each k
        for k in k_values:
            if k > len(y_true):
                k = len(y_true)

            # Precision@k
            metrics.precision_at_k[k] = self._precision_at_k(sorted_labels, k)

            # Recall@k
            metrics.recall_at_k[k] = self._recall_at_k(sorted_labels, k, y_true.sum())

            # NDCG@k
            metrics.ndcg_at_k[k] = self._ndcg_at_k(sorted_labels, k)

        # MRR (Mean Reciprocal Rank)
        metrics.mrr = self._compute_mrr(sorted_labels)

        # MAP (Mean Average Precision) - same as AUC-PR for binary
        metrics.map_score = average_precision_score(y_true, y_scores)

        return metrics

    def _precision_at_k(self, sorted_labels: np.ndarray, k: int) -> float:
        """Compute Precision@k."""
        return float(sorted_labels[:k].sum() / k)

    def _recall_at_k(self, sorted_labels: np.ndarray, k: int, total_relevant: int) -> float:
        """Compute Recall@k."""
        if total_relevant == 0:
            return 0.0
        return float(sorted_labels[:k].sum() / total_relevant)

    def _ndcg_at_k(self, sorted_labels: np.ndarray, k: int) -> float:
        """Compute NDCG@k (Normalized Discounted Cumulative Gain).

        NDCG measures ranking quality by giving higher weight to
        relevant items at top positions.
        """
        # DCG@k
        dcg = 0.0
        for i in range(min(k, len(sorted_labels))):
            rel = sorted_labels[i]
            # log2(i+2) because positions are 1-indexed
            dcg += (2**rel - 1) / np.log2(i + 2)

        # Ideal DCG (perfect ranking)
        ideal_sorted = np.sort(sorted_labels)[::-1]
        idcg = 0.0
        for i in range(min(k, len(ideal_sorted))):
            rel = ideal_sorted[i]
            idcg += (2**rel - 1) / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _compute_mrr(self, sorted_labels: np.ndarray) -> float:
        """Compute Mean Reciprocal Rank.

        MRR is 1/rank of the first relevant item.
        """
        for i, label in enumerate(sorted_labels):
            if label == 1:
                return 1.0 / (i + 1)
        return 0.0

    def find_optimal_threshold(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        metric: str = "f1",
        min_recall: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Find optimal classification threshold.

        Args:
            y_true: True labels
            y_scores: Prediction scores
            metric: Metric to optimize ('f1', 'precision', 'recall')
            min_recall: If set, find threshold with min recall constraint

        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # Get thresholds from precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Compute F1 at each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        if min_recall is not None:
            # Find thresholds that satisfy minimum recall
            valid_indices = np.where(recall >= min_recall)[0]
            if len(valid_indices) == 0:
                logger.warning(f"No threshold achieves recall >= {min_recall}")
                valid_indices = np.arange(len(thresholds))
        else:
            valid_indices = np.arange(len(thresholds))

        if metric == "f1":
            scores = f1_scores[valid_indices]
        elif metric == "precision":
            scores = precision[valid_indices]
        elif metric == "recall":
            scores = recall[valid_indices]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        best_idx = valid_indices[np.argmax(scores)]

        # Handle edge case where best_idx equals len(thresholds)
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1

        return float(thresholds[best_idx]), float(scores.max())

    def compute_at_threshold(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        threshold: float,
    ) -> ClassificationMetrics:
        """Compute metrics at a specific threshold.

        Args:
            y_true: True labels
            y_scores: Prediction scores
            threshold: Classification threshold

        Returns:
            ClassificationMetrics at the given threshold
        """
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        y_pred = (y_scores >= threshold).astype(int)

        return self.compute_classification_metrics(y_true, y_pred, y_scores)

    def compute_threshold_curve(
        self,
        y_true: Union[np.ndarray, List],
        y_scores: Union[np.ndarray, List],
        num_thresholds: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Compute metrics across multiple thresholds.

        Args:
            y_true: True labels
            y_scores: Prediction scores
            num_thresholds: Number of threshold points

        Returns:
            Dictionary with arrays of metrics at each threshold
        """
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        thresholds = np.linspace(0, 1, num_thresholds)

        results = {
            "thresholds": thresholds,
            "precision": np.zeros(num_thresholds),
            "recall": np.zeros(num_thresholds),
            "f1": np.zeros(num_thresholds),
            "accuracy": np.zeros(num_thresholds),
        }

        for i, thresh in enumerate(thresholds):
            y_pred = (y_scores >= thresh).astype(int)
            results["precision"][i] = precision_score(y_true, y_pred, zero_division=0)
            results["recall"][i] = recall_score(y_true, y_pred, zero_division=0)
            results["f1"][i] = f1_score(y_true, y_pred, zero_division=0)
            results["accuracy"][i] = accuracy_score(y_true, y_pred)

        return results


class MultiLabelMetrics:
    """Metrics for multi-label classification (multiple fraud types).

    In AML, a transaction might exhibit multiple fraud patterns
    (e.g., both structuring AND layering). These metrics handle
    the multi-label case.
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute multi-label metrics.

        Args:
            y_true: True labels (n_samples, n_labels)
            y_pred: Predicted labels (n_samples, n_labels)
            label_names: Names of labels

        Returns:
            Dictionary with multi-label metrics
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for multi-label metrics")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        n_labels = y_true.shape[1] if y_true.ndim > 1 else 1
        label_names = label_names or [f"label_{i}" for i in range(n_labels)]

        metrics = {
            # Hamming Loss: fraction of labels incorrectly predicted
            "hamming_loss": hamming_loss(y_true, y_pred),

            # Subset Accuracy: fraction of samples with all labels correct
            "subset_accuracy": accuracy_score(y_true, y_pred),

            # Macro F1: average F1 across labels
            "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),

            # Micro F1: global F1 treating all predictions as one
            "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),

            # Per-label metrics
            "per_label": {},
        }

        # Per-label metrics
        for i, name in enumerate(label_names):
            if y_true.ndim > 1:
                label_true = y_true[:, i]
                label_pred = y_pred[:, i]
            else:
                label_true = y_true
                label_pred = y_pred

            metrics["per_label"][name] = {
                "precision": precision_score(label_true, label_pred, zero_division=0),
                "recall": recall_score(label_true, label_pred, zero_division=0),
                "f1": f1_score(label_true, label_pred, zero_division=0),
                "support": int(label_true.sum()),
            }

        return metrics


class TypologyMetrics:
    """Metrics specific to AML fraud typology classification.

    Fraud typologies include:
    - Structuring (smurfing)
    - Layering
    - Cycling
    - Integration
    - Trade-based ML
    - Shell company usage
    """

    TYPOLOGIES = [
        "structuring",
        "layering",
        "cycling",
        "integration",
        "trade_based",
        "shell_company",
        "other",
    ]

    def __init__(self, typologies: Optional[List[str]] = None):
        """Initialize with typology list.

        Args:
            typologies: List of typology names (uses default if None)
        """
        self.typologies = typologies or self.TYPOLOGIES
        self.computer = MetricsComputer(class_names=self.typologies)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute typology classification metrics.

        Args:
            y_true: True typology labels (class indices)
            y_pred: Predicted typology labels
            y_scores: Optional prediction scores (n_samples, n_classes)

        Returns:
            Dictionary with typology metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Overall metrics
        metrics = self.computer.compute_classification_metrics(
            y_true, y_pred, average='macro'
        )

        result = {
            "macro_f1": metrics.macro_f1,
            "micro_f1": metrics.micro_f1,
            "weighted_f1": metrics.weighted_f1,
            "accuracy": metrics.accuracy,
            "per_typology": {},
        }

        # Per-typology metrics
        for i, typology in enumerate(self.typologies):
            if i >= len(np.unique(y_true)):
                continue

            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)

            result["per_typology"][typology] = {
                "precision": precision_score(binary_true, binary_pred, zero_division=0),
                "recall": recall_score(binary_true, binary_pred, zero_division=0),
                "f1": f1_score(binary_true, binary_pred, zero_division=0),
                "support": int(binary_true.sum()),
            }

        # Confusion matrix
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        return result


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    k_values: List[int] = [10, 20, 50, 100],
) -> Dict[str, Any]:
    """Convenience function to compute all metrics at once.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (optional)
        k_values: K values for ranking metrics

    Returns:
        Dictionary with all computed metrics
    """
    computer = MetricsComputer()

    # Classification metrics
    clf_metrics = computer.compute_classification_metrics(y_true, y_pred, y_scores)

    result = {
        "classification": clf_metrics.to_dict(),
    }

    # Ranking metrics (if scores provided)
    if y_scores is not None:
        ranking_metrics = computer.compute_ranking_metrics(y_true, y_scores, k_values)
        result["ranking"] = ranking_metrics.to_dict()

        # Optimal threshold
        opt_threshold, opt_f1 = computer.find_optimal_threshold(y_true, y_scores)
        result["optimal_threshold"] = {
            "threshold": opt_threshold,
            "f1_at_threshold": opt_f1,
        }

    return result


# Export
__all__ = [
    "ClassificationMetrics",
    "RankingMetrics",
    "MetricsComputer",
    "MultiLabelMetrics",
    "TypologyMetrics",
    "compute_all_metrics",
]
