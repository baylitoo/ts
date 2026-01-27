"""
Budgeted Detection Metrics for AML

Metrics that matter to compliance teams:
- Recall@K: How many frauds caught with K alerts?
- Precision@K: What % of K alerts are actual fraud?
- AUPR: Area under precision-recall curve (better than AUROC for imbalanced data)

These metrics reflect real-world constraints:
- Limited alert budget (compliance team can only investigate K cases/day)
- Extreme class imbalance (0.16% fraud rate)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)


class BudgetedMetrics:
    """
    Metrics for budgeted fraud detection.

    Example usage:
        metrics = BudgetedMetrics()

        # Predict risk scores for test set
        predictions = model.predict(X_test)  # Higher = more suspicious
        labels = y_test  # 0 = normal, 1 = fraud

        # How many frauds do we catch with K=100 alerts?
        recall = metrics.recall_at_k(predictions, labels, k=100)
        precision = metrics.precision_at_k(predictions, labels, k=100)

        print(f"With 100 alerts: catch {recall*100:.1f}% of fraud")
        print(f"Precision: {precision*100:.1f}% of alerts are fraud")
    """

    def recall_at_k(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Recall when flagging top-K highest-scored transactions.

        This answers: "If I can only investigate K transactions per day,
        what percentage of fraud will I catch?"

        Args:
            predictions: Risk scores (N,) - higher = more suspicious
            labels: True labels (N,) - 1 = fraud, 0 = normal
            k: Alert budget (number of transactions to flag)

        Returns:
            Recall@K: fraction of fraud caught (0.0 to 1.0)

        Example:
            predictions = [0.9, 0.1, 0.8, 0.05, 0.7]
            labels = [1, 0, 1, 0, 0]  # 2 frauds total
            k = 2

            Top-2 predictions: indices [0, 2] (scores 0.9, 0.8)
            Labels at [0, 2]: [1, 1] = 2 frauds caught
            Recall@2 = 2/2 = 100%
        """
        if len(predictions) != len(labels):
            raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")

        if k <= 0 or k > len(predictions):
            raise ValueError(f"k must be in range [1, {len(predictions)}], got {k}")

        # Get indices of top-K predictions (highest scores first)
        top_k_indices = np.argsort(predictions)[-k:]

        # Count frauds in top-K
        frauds_in_top_k = labels[top_k_indices].sum()

        # Count total frauds
        total_frauds = labels.sum()

        # Recall = caught / total
        if total_frauds == 0:
            return 0.0  # No fraud to catch

        return float(frauds_in_top_k / total_frauds)

    def precision_at_k(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Precision when flagging top-K highest-scored transactions.

        This answers: "Of the K transactions I flag, what percentage
        are actually fraud?"

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            k: Alert budget

        Returns:
            Precision@K: fraction of flagged alerts that are fraud (0.0 to 1.0)

        Example:
            predictions = [0.9, 0.1, 0.8, 0.05, 0.7]
            labels = [1, 0, 1, 0, 0]
            k = 3

            Top-3 predictions: indices [0, 2, 4] (scores 0.9, 0.8, 0.7)
            Labels at [0, 2, 4]: [1, 1, 0] = 2 frauds, 1 false alarm
            Precision@3 = 2/3 = 66.7%
        """
        if len(predictions) != len(labels):
            raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")

        if k <= 0 or k > len(predictions):
            raise ValueError(f"k must be in range [1, {len(predictions)}], got {k}")

        # Get top-K indices
        top_k_indices = np.argsort(predictions)[-k:]

        # Precision = frauds_in_top_k / k
        frauds_in_top_k = labels[top_k_indices].sum()

        return float(frauds_in_top_k / k)

    def f1_at_k(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        F1-score at K (harmonic mean of precision and recall).

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            k: Alert budget

        Returns:
            F1@K: 2 * (precision * recall) / (precision + recall)
        """
        precision = self.precision_at_k(predictions, labels, k)
        recall = self.recall_at_k(predictions, labels, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def recall_at_k_curve(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        max_k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Recall@K for all K from 1 to max_k.

        This shows the trade-off: "How does recall improve as I increase
        my alert budget?"

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            max_k: Maximum K to compute (default: all predictions)

        Returns:
            k_values: Array of K values [1, 2, ..., max_k]
            recalls: Corresponding Recall@K values
        """
        if max_k is None:
            max_k = len(predictions)

        max_k = min(max_k, len(predictions))

        k_values = np.arange(1, max_k + 1)
        recalls = np.array([self.recall_at_k(predictions, labels, k) for k in k_values])

        return k_values, recalls

    def aupr(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Area Under Precision-Recall Curve (AUPR).

        AUPR is better than AUROC for imbalanced datasets because:
        - Focuses on positive class (fraud) performance
        - Not affected by large number of true negatives
        - Ranges from baseline (fraud_rate) to 1.0

        For 0.16% fraud rate:
        - Random classifier: AUPR ≈ 0.0016
        - Perfect classifier: AUPR = 1.0
        - Good classifier: AUPR > 0.5

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)

        Returns:
            AUPR: Area under precision-recall curve (0.0 to 1.0)
        """
        if len(np.unique(labels)) < 2:
            # Only one class present
            return 0.0

        return average_precision_score(labels, predictions)

    def plot_recall_vs_budget(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        max_k: int = 1000,
        save_path: Optional[str] = None
    ):
        """
        Plot Recall@K vs Alert Budget (K).

        This visualization shows: "How many frauds can I catch with
        different alert budgets?"

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            max_k: Maximum K to plot
            save_path: If provided, save figure to this path
        """
        k_values, recalls = self.recall_at_k_curve(predictions, labels, max_k)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, recalls, linewidth=2, color='#2ecc71')
        plt.xlabel('Alert Budget (K)', fontsize=12)
        plt.ylabel('Recall@K (Fraud Detection Rate)', fontsize=12)
        plt.title('Budgeted Fraud Detection: Recall vs Alert Budget', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add reference lines
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% Recall Target')
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100% Recall')

        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_precision_recall_curve(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve.

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            save_path: If provided, save figure to this path
        """
        precision, recall, thresholds = precision_recall_curve(labels, predictions)
        aupr_score = self.aupr(predictions, labels)

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2, color='#3498db',
                 label=f'AUPR = {aupr_score:.3f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def summary_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        budget_levels: List[int] = [50, 100, 200, 500]
    ) -> Dict[str, float]:
        """
        Generate summary report at multiple budget levels.

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            budget_levels: List of K values to evaluate

        Returns:
            Dictionary with metrics at each budget level
        """
        report = {
            'aupr': self.aupr(predictions, labels),
            'total_samples': len(labels),
            'total_fraud': int(labels.sum()),
            'fraud_rate': float(labels.mean()),
        }

        for k in budget_levels:
            if k > len(predictions):
                continue

            recall = self.recall_at_k(predictions, labels, k)
            precision = self.precision_at_k(predictions, labels, k)
            f1 = self.f1_at_k(predictions, labels, k)

            report[f'recall@{k}'] = recall
            report[f'precision@{k}'] = precision
            report[f'f1@{k}'] = f1

        return report

    def print_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        budget_levels: List[int] = [50, 100, 200, 500]
    ):
        """
        Print formatted summary report.

        Args:
            predictions: Risk scores (N,)
            labels: True labels (N,)
            budget_levels: List of K values to evaluate
        """
        report = self.summary_report(predictions, labels, budget_levels)

        print("\n" + "=" * 70)
        print("BUDGETED FRAUD DETECTION REPORT")
        print("=" * 70)

        print("\nDataset Statistics:")
        print(f"  Total transactions: {report['total_samples']:,}")
        print(f"  Fraud cases: {report['total_fraud']:,} ({report['fraud_rate']*100:.2f}%)")
        print(f"  AUPR (overall): {report['aupr']:.3f}")

        print("\nPerformance at Different Alert Budgets:")
        print(f"  {'Budget (K)':<15} {'Recall@K':<12} {'Precision@K':<15} {'F1@K':<10}")
        print(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*10}")

        for k in budget_levels:
            if k > len(predictions):
                continue

            recall = report[f'recall@{k}']
            precision = report[f'precision@{k}']
            f1 = report[f'f1@{k}']

            print(f"  {k:<15} {recall:<12.1%} {precision:<15.1%} {f1:<10.3f}")

        print("=" * 70 + "\n")
