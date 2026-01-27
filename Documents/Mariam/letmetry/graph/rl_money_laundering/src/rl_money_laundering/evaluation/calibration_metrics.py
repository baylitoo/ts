"""
Calibration Metrics for AML Detection Models

Implements probability calibration assessment:
1. Expected Calibration Error (ECE)
2. Maximum Calibration Error (MCE)
3. Brier Score
4. Reliability Diagrams
5. Calibration Curves

Calibration is critical for AML where predicted probabilities
inform investigation priorities and resource allocation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict


@dataclass
class CalibrationMetrics:
    """Container for all calibration metrics."""

    # Expected Calibration Error (lower is better)
    ece: float = 0.0

    # Maximum Calibration Error (lower is better)
    mce: float = 0.0

    # Brier Score (lower is better, 0 is perfect)
    brier_score: float = 0.0

    # Brier Score decomposition
    brier_reliability: float = 0.0  # Calibration component
    brier_resolution: float = 0.0   # Sharpness component
    brier_uncertainty: float = 0.0  # Inherent uncertainty

    # Reliability diagram data
    bin_confidences: List[float] = field(default_factory=list)
    bin_accuracies: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)

    # Calibration curve data (for plotting)
    mean_predicted_probs: List[float] = field(default_factory=list)
    fraction_of_positives: List[float] = field(default_factory=list)

    # Additional metrics
    overconfidence_error: float = 0.0  # Error when confident but wrong
    underconfidence_error: float = 0.0  # Error when uncertain but right

    # Per-class calibration (for multi-class)
    per_class_ece: Dict[int, float] = field(default_factory=dict)
    per_class_brier: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "brier_reliability": self.brier_reliability,
            "brier_resolution": self.brier_resolution,
            "brier_uncertainty": self.brier_uncertainty,
            "overconfidence_error": self.overconfidence_error,
            "underconfidence_error": self.underconfidence_error,
            "n_bins": len(self.bin_counts),
            "per_class_ece": self.per_class_ece,
            "per_class_brier": self.per_class_brier,
        }


class CalibrationComputer:
    """
    Computes calibration metrics for probability predictions.

    Calibration measures how well predicted probabilities match
    observed frequencies. A well-calibrated model predicting 80%
    probability should be correct 80% of the time.

    Args:
        n_bins: Number of bins for ECE/reliability diagrams
        strategy: Binning strategy - 'uniform' or 'quantile'
    """

    def __init__(
        self,
        n_bins: int = 15,
        strategy: str = "uniform"
    ):
        self.n_bins = n_bins
        self.strategy = strategy

    def compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Tuple[float, List[float], List[float], List[int]]:
        """
        Compute Expected Calibration Error.

        ECE = sum_{b=1}^{B} (n_b / N) * |acc(b) - conf(b)|

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            n_bins: Number of bins (overrides instance setting)

        Returns:
            ece: Expected Calibration Error
            bin_confidences: Mean confidence per bin
            bin_accuracies: Accuracy per bin
            bin_counts: Sample count per bin
        """
        n_bins = n_bins or self.n_bins
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        # Create bins based on strategy
        if self.strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        else:  # quantile
            bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(bin_edges)
            n_bins = len(bin_edges) - 1

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        ece = 0.0
        total_samples = len(y_true)

        for i in range(n_bins):
            # Find samples in this bin
            if i == n_bins - 1:
                # Include right edge for last bin
                in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            else:
                in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

            bin_size = in_bin.sum()

            if bin_size > 0:
                # Mean confidence in bin
                avg_confidence = y_prob[in_bin].mean()
                # Accuracy in bin
                accuracy = y_true[in_bin].mean()

                bin_confidences.append(float(avg_confidence))
                bin_accuracies.append(float(accuracy))
                bin_counts.append(int(bin_size))

                # Weighted contribution to ECE
                ece += (bin_size / total_samples) * abs(accuracy - avg_confidence)
            else:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
                bin_counts.append(0)

        return float(ece), bin_confidences, bin_accuracies, bin_counts

    def compute_mce(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Compute Maximum Calibration Error.

        MCE = max_{b} |acc(b) - conf(b)|

        The worst-case calibration error across all bins.
        """
        _, bin_confidences, bin_accuracies, bin_counts = self.compute_ece(
            y_true, y_prob, n_bins
        )

        mce = 0.0
        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
            if count > 0:
                mce = max(mce, abs(acc - conf))

        return float(mce)

    def compute_brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute Brier Score (mean squared error of probabilities).

        BS = (1/N) * sum((y_prob - y_true)^2)

        Range: [0, 1], lower is better.
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        return float(np.mean((y_prob - y_true) ** 2))

    def compute_brier_decomposition(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Compute Brier Score decomposition.

        BS = Reliability - Resolution + Uncertainty

        - Reliability (REL): Measures calibration (lower is better)
        - Resolution (RES): Measures discrimination (higher is better)
        - Uncertainty (UNC): Inherent uncertainty (constant for given labels)

        Returns:
            reliability, resolution, uncertainty
        """
        n_bins = n_bins or self.n_bins
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        N = len(y_true)
        base_rate = y_true.mean()

        # Uncertainty (inherent to the problem)
        uncertainty = base_rate * (1 - base_rate)

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)

        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            if i == n_bins - 1:
                in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            else:
                in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

            n_k = in_bin.sum()

            if n_k > 0:
                o_k = y_true[in_bin].mean()  # Observed frequency
                f_k = y_prob[in_bin].mean()  # Forecast probability

                # Reliability contribution
                reliability += (n_k / N) * (f_k - o_k) ** 2

                # Resolution contribution
                resolution += (n_k / N) * (o_k - base_rate) ** 2

        return float(reliability), float(resolution), float(uncertainty)

    def compute_confidence_errors(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Tuple[float, float]:
        """
        Compute over/under-confidence errors.

        Overconfidence: High confidence but wrong
        Underconfidence: Low confidence but right

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            confidence_threshold: Threshold for "high confidence"

        Returns:
            overconfidence_error, underconfidence_error
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # Max probability (confidence)
        confidence = np.maximum(y_prob, 1 - y_prob)

        # High confidence cases
        high_conf = confidence >= confidence_threshold
        # Low confidence cases
        low_conf = ~high_conf

        # Correct predictions
        correct = (y_pred == y_true)

        # Overconfidence: high confidence but wrong
        overconf_mask = high_conf & ~correct
        if overconf_mask.sum() > 0:
            overconfidence = (confidence[overconf_mask] - 0.5).mean()
        else:
            overconfidence = 0.0

        # Underconfidence: low confidence but right
        underconf_mask = low_conf & correct
        if underconf_mask.sum() > 0:
            underconfidence = (0.5 - (1 - confidence[underconf_mask])).mean()
        else:
            underconfidence = 0.0

        return float(overconfidence), float(underconfidence)

    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Compute calibration curve data for plotting.

        Returns:
            mean_predicted_probs: Mean predicted probability per bin
            fraction_of_positives: Fraction of positives per bin
        """
        n_bins = n_bins or self.n_bins
        _, bin_confidences, bin_accuracies, bin_counts = self.compute_ece(
            y_true, y_prob, n_bins
        )

        # Filter out empty bins
        mean_probs = []
        frac_positives = []

        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
            if count > 0:
                mean_probs.append(conf)
                frac_positives.append(acc)

        return mean_probs, frac_positives

    def compute_all(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> CalibrationMetrics:
        """
        Compute all calibration metrics.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            confidence_threshold: Threshold for confidence error analysis

        Returns:
            CalibrationMetrics object with all metrics
        """
        # ECE with reliability diagram data
        ece, bin_conf, bin_acc, bin_counts = self.compute_ece(y_true, y_prob)

        # MCE
        mce = self.compute_mce(y_true, y_prob)

        # Brier Score and decomposition
        brier = self.compute_brier_score(y_true, y_prob)
        rel, res, unc = self.compute_brier_decomposition(y_true, y_prob)

        # Confidence errors
        overconf, underconf = self.compute_confidence_errors(
            y_true, y_prob, confidence_threshold
        )

        # Calibration curve
        mean_probs, frac_pos = self.compute_calibration_curve(y_true, y_prob)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            brier_reliability=rel,
            brier_resolution=res,
            brier_uncertainty=unc,
            bin_confidences=bin_conf,
            bin_accuracies=bin_acc,
            bin_counts=bin_counts,
            mean_predicted_probs=mean_probs,
            fraction_of_positives=frac_pos,
            overconfidence_error=overconf,
            underconfidence_error=underconf,
        )


class MultiClassCalibration:
    """
    Calibration metrics for multi-class classification.

    Extends binary calibration to multi-class setting using:
    1. Class-wise calibration (one-vs-all per class)
    2. Top-label calibration (confidence of predicted class)
    3. Marginal calibration (averaged across classes)
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.binary_computer = CalibrationComputer(n_bins=n_bins)

    def compute_classwise_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute ECE for each class (one-vs-all).

        Args:
            y_true: True class labels (N,)
            y_prob: Predicted probabilities (N, C) where C is num classes

        Returns:
            Dict mapping class index to ECE
        """
        n_classes = y_prob.shape[1]
        class_ece = {}

        for c in range(n_classes):
            # One-vs-all for this class
            y_binary = (y_true == c).astype(int)
            y_prob_c = y_prob[:, c]

            ece, _, _, _ = self.binary_computer.compute_ece(y_binary, y_prob_c)
            class_ece[c] = ece

        return class_ece

    def compute_top_label_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute ECE based on top-label confidence.

        Uses the confidence of the predicted class only.
        """
        # Predicted class and its confidence
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)

        # Correctness
        correct = (y_pred == y_true).astype(int)

        ece, _, _, _ = self.binary_computer.compute_ece(correct, confidences)
        return ece

    def compute_marginal_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute marginal (averaged) ECE across all classes.
        """
        class_ece = self.compute_classwise_ece(y_true, y_prob)
        return float(np.mean(list(class_ece.values())))

    def compute_classwise_brier(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute Brier Score for each class.
        """
        n_classes = y_prob.shape[1]
        class_brier = {}

        for c in range(n_classes):
            y_binary = (y_true == c).astype(int)
            y_prob_c = y_prob[:, c]

            brier = self.binary_computer.compute_brier_score(y_binary, y_prob_c)
            class_brier[c] = brier

        return class_brier

    def compute_all(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> CalibrationMetrics:
        """
        Compute all multi-class calibration metrics.

        Args:
            y_true: True class labels (N,)
            y_prob: Predicted probabilities (N, C)

        Returns:
            CalibrationMetrics with multi-class specific values
        """
        # Convert to binary for top-label calibration
        y_pred = np.argmax(y_prob, axis=1)
        correct = (y_pred == y_true).astype(int)
        confidences = np.max(y_prob, axis=1)

        # Top-label calibration
        binary_metrics = self.binary_computer.compute_all(correct, confidences)

        # Add per-class metrics
        binary_metrics.per_class_ece = self.compute_classwise_ece(y_true, y_prob)
        binary_metrics.per_class_brier = self.compute_classwise_brier(y_true, y_prob)

        return binary_metrics


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.

    Post-hoc calibration technique that learns a single
    temperature parameter T to divide logits by.

    p_calibrated = softmax(logits / T)

    Reference: Guo et al., "On Calibration of Modern Neural Networks"
    """

    def __init__(self, init_temperature: float = 1.0):
        self.temperature = init_temperature
        self._fitted = False

    def fit(
        self,
        y_true: np.ndarray,
        logits: np.ndarray,
        max_iter: int = 50,
        lr: float = 0.01
    ) -> float:
        """
        Fit temperature parameter using NLL minimization.

        Args:
            y_true: True labels
            logits: Raw model outputs (before softmax)
            max_iter: Maximum optimization iterations
            lr: Learning rate

        Returns:
            Optimal temperature value
        """
        # Simple grid search for temperature
        # (Full implementation would use gradient descent)
        best_temp = 1.0
        best_nll = float('inf')

        for t in np.linspace(0.1, 5.0, 50):
            # Scale logits
            scaled = logits / t
            # Softmax
            exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)

            # Negative log-likelihood
            if probs.ndim == 1:
                nll = -np.log(probs[y_true] + 1e-10)
            else:
                nll = -np.log(probs[np.arange(len(y_true)), y_true] + 1e-10)
            nll = nll.mean()

            if nll < best_nll:
                best_nll = nll
                best_temp = t

        self.temperature = best_temp
        self._fitted = True
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model outputs

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise ValueError("Temperature not fitted. Call fit() first.")

        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)

        return probs


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.

    Non-parametric calibration that learns a monotonic
    mapping from predicted probabilities to calibrated ones.

    Works well for binary classification.
    """

    def __init__(self):
        self._mapping = None
        self._fitted = False

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> None:
        """
        Fit isotonic regression.

        Uses Pool Adjacent Violators (PAV) algorithm.
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        # Sort by predicted probability
        order = np.argsort(y_prob)
        y_true_sorted = y_true[order]
        y_prob_sorted = y_prob[order]

        # Pool Adjacent Violators algorithm
        n = len(y_true)
        result = y_true_sorted.astype(float).copy()

        # Merge adjacent violations
        changed = True
        while changed:
            changed = False
            i = 0
            while i < n - 1:
                if result[i] > result[i + 1]:
                    # Merge and take weighted average
                    merged = (result[i] + result[i + 1]) / 2
                    result[i] = merged
                    result[i + 1] = merged
                    changed = True
                i += 1

        # Store mapping
        self._mapping = {
            "prob": y_prob_sorted,
            "calibrated": result
        }
        self._fitted = True

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        """
        if not self._fitted:
            raise ValueError("Not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob).flatten()

        # Linear interpolation
        calibrated = np.interp(
            y_prob,
            self._mapping["prob"],
            self._mapping["calibrated"]
        )

        return np.clip(calibrated, 0, 1)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    is_multiclass: bool = False
) -> CalibrationMetrics:
    """
    Convenience function to compute all calibration metrics.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE
        is_multiclass: Whether this is multi-class classification

    Returns:
        CalibrationMetrics object
    """
    if is_multiclass and y_prob.ndim > 1:
        computer = MultiClassCalibration(n_bins=n_bins)
    else:
        computer = CalibrationComputer(n_bins=n_bins)

    return computer.compute_all(y_true, y_prob)
