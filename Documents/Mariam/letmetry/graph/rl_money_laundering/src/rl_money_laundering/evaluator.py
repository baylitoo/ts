"""
Evaluation framework for temporal graph guardrail deployments.


Provides reusable metrics and comparison utilities. AML detection is a
featured case study, but the evaluators accept any binary risk labelling
problem on temporal graphs.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

MetricValue = Union[float, int, None]


class AMLEvaluator:
    """
    Comprehensive evaluator for temporal graph risk detection methods.

    Metrics:
    - Classification: Precision, Recall, F1, Accuracy
    - ROC/PR curves and AUC
    - Detection rate and False Positive Rate
    - Scheme completeness (for RL agents)
    """

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, MetricValue]] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray | None = None,
        method_name: str = "Method"
    ) -> Dict[str, MetricValue]:
        """
        Evaluate predictions

        Args:
            y_true: Ground truth labels (1 = fraud)
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities (optional)
            method_name: Name of method being evaluated

        Returns:
            Dictionary of metrics
        """
        metrics: Dict[str, MetricValue] = {}

        # Basic classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)

        # Detection rate and FPR
        metrics['detection_rate'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # AUC metrics (if scores provided)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))

                # PR-AUC (more informative for imbalanced data)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                metrics['pr_auc'] = float(auc(recall_curve, precision_curve))
            except (ValueError, IndexError):
                # Handle case where there aren't enough samples or classes
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None

        # Store results
        self.results[method_name] = metrics

        return metrics

    def compare_methods(
        self,
        results: Dict[str, Dict[str, MetricValue]] | None = None
    ) -> str:
        """
        Generate comparison table

        Args:
            results: Dictionary of {method_name: metrics}
                     If None, uses stored results

        Returns:
            Formatted comparison table
        """
        if results is None:
            results = self.results

        if not results:
            return "No results to compare"

        # Create table
        table = []
        table.append("=" * 100)
        table.append(f"{'Method':<20} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | "
                    f"{'Detection%':>10} | {'FPR%':>10} | {'ROC-AUC':>10}")
        table.append("=" * 100)

        for method_name, metrics in results.items():
            precision = float(metrics.get('precision') or 0.0)
            recall = float(metrics.get('recall') or 0.0)
            f1_score = float(metrics.get('f1_score') or 0.0)
            detection_rate = float(metrics.get('detection_rate') or 0.0)
            false_positive_rate = float(metrics.get('false_positive_rate') or 0.0)
            roc_auc = float(metrics.get('roc_auc') or 0.0)

            row = (
                f"{method_name:<20} | "
                f"{precision*100:>9.2f}% | "
                f"{recall*100:>9.2f}% | "
                f"{f1_score*100:>9.2f}% | "
                f"{detection_rate*100:>9.2f}% | "
                f"{false_positive_rate*100:>9.2f}% | "
                f"{roc_auc:>10.4f}"
            )
            table.append(row)

        table.append("=" * 100)

        return "\n".join(table)

    def print_report(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Print detailed classification report

        Args:
            method_name: Method name
            y_true: Ground truth
            y_pred: Predictions
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {method_name}")
        print(f"{'='*60}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print("                 Predicted")
        print("                Legit  Fraud")
        print(f"Actual Legit  {cm[0,0]:>6}  {cm[0,1]:>6}")
        print(f"       Fraud  {cm[1,0]:>6}  {cm[1,1]:>6}")

        # Get detailed metrics
        if method_name in self.results:
            metrics = self.results[method_name]
            print("\nDetailed Metrics:")
            detection_rate = float(metrics.get('detection_rate') or 0.0)
            false_positive_rate = float(metrics.get('false_positive_rate') or 0.0)
            roc_auc = float(metrics.get('roc_auc') or 0.0)
            pr_auc = float(metrics.get('pr_auc') or 0.0)

            print(f"  Detection Rate: {detection_rate*100:.2f}%")
            print(f"  False Positive Rate: {false_positive_rate*100:.2f}%")
            if roc_auc > 0.0:
                print(f"  ROC-AUC: {roc_auc:.4f}")
            if pr_auc > 0.0:
                print(f"  PR-AUC: {pr_auc:.4f}")

        print(f"{'='*60}\n")

    def evaluate_rl_agent(
        self,
        discovered_subgraphs: List[Set[str]],
        ground_truth_schemes: List[Set[str]],
        total_frauds: int
    ) -> Dict[str, float]:
        """
        Evaluate RL agent's subgraph discovery

        Args:
            discovered_subgraphs: List of node sets discovered by agent
            ground_truth_schemes: List of true fraud scheme node sets
            total_frauds: Total number of fraud nodes in graph

        Returns:
            Metrics specific to RL agent
        """
        metrics: Dict[str, float] = {}

        # Scheme-level metrics
        schemes_found = 0
        partial_schemes = 0

        for gt_scheme in ground_truth_schemes:
            max_overlap = 0.0
            for discovered in discovered_subgraphs:
                overlap = len(gt_scheme & discovered) / len(gt_scheme)
                max_overlap = max(max_overlap, overlap)

            if max_overlap >= 0.8:  # 80% overlap = found
                schemes_found += 1
            elif max_overlap >= 0.3:  # 30-80% = partial
                partial_schemes += 1

        metrics['scheme_detection_rate'] = schemes_found / len(ground_truth_schemes) if ground_truth_schemes else 0.0
        metrics['partial_scheme_rate'] = partial_schemes / len(ground_truth_schemes) if ground_truth_schemes else 0.0

        # Node-level coverage
        all_discovered_nodes: Set[str] = set()
        for subgraph in discovered_subgraphs:
            all_discovered_nodes.update(subgraph)

        all_fraud_nodes: Set[str] = set()
        for scheme in ground_truth_schemes:
            all_fraud_nodes.update(scheme)

        metrics['fraud_node_coverage'] = len(all_discovered_nodes & all_fraud_nodes) / total_frauds if total_frauds > 0 else 0.0

        # Efficiency metrics
        metrics['avg_subgraph_size'] = float(np.mean([len(sg) for sg in discovered_subgraphs])) if discovered_subgraphs else 0.0
        metrics['total_nodes_explored'] = float(len(all_discovered_nodes))

        return metrics

    def save_results(self, filepath: str) -> None:
        """Save evaluation results to file"""
        import json

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> None:
        """Load evaluation results from file"""
        import json

        with open(filepath, 'r') as f:
            self.results = json.load(f)

        print(f"Results loaded from {filepath}")
