# mypy: ignore-errors

"""
XGBoost Baseline for AML Detection

Traditional ML baseline using XGBoost on Algorithm-2 features
(amount + temporal + network features from AMLNet paper).

This represents what the AMLNet paper does:
- Isolation Forest + Random Forest on flat features
- No graph traversal, no RL

Used as comparison to show whether RL adds value.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """
    XGBoost classifier on Algorithm-2 features.

    Features extracted per transaction (edge):
    - Source node features (20-dim): account type, amounts, temporal, network
    - Destination node features (20-dim): same
    - Edge features (2-dim): amount, timestamp

    Total: 42 features per transaction

    This is the "flat" baseline - treats each transaction independently,
    no graph traversal or exploration like RL.
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize XGBoost baseline.

        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            scale_pos_weight: Weight for positive class (fraud)
                             If None, auto-computed from fraud rate
            random_state: Random seed
        """
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model = None
        self.feature_names = None

    def _extract_edge_features(
        self,
        graph: nx.DiGraph,
        src: str,
        dst: str,
        feature_extractor
    ) -> np.ndarray:
        """
        Extract features for a single edge (transaction).

        Returns 42-dimensional feature vector:
        - src_features (20): source node features
        - dst_features (20): destination node features
        - edge_features (2): transaction amount (log), timestamp
        """
        # Source node features (20-dim)
        src_features = feature_extractor.extract(graph.nodes[src])

        # Destination node features (20-dim)
        dst_features = feature_extractor.extract(graph.nodes[dst])

        # Edge features (2-dim)
        edge_data = graph.edges[src, dst]
        edge_amount = edge_data.get('amount', 0.0)
        edge_time = edge_data.get('step', 0)

        edge_features = np.array([
            np.log1p(edge_amount),  # Log-scaled amount
            edge_time / 1000.0      # Normalized timestamp
        ])

        # Concatenate all features
        features = np.concatenate([
            src_features,   # 20
            dst_features,   # 20
            edge_features   # 2
        ])  # Total: 42

        return features

    def prepare_features(
        self,
        graph: nx.DiGraph,
        feature_extractor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for all edges in graph.

        Args:
            graph: Transaction graph
            feature_extractor: NodeFeatureExtractor instance

        Returns:
            X: Feature matrix (num_edges, 42)
            y: Labels (num_edges,) - 0=normal, 1=fraud
        """
        from rl_money_laundering.features.extractors import NodeFeatureExtractor

        if feature_extractor is None:
            feature_extractor = NodeFeatureExtractor(
                include_temporal=True,
                include_network=True,
                node_feature_dim=20
            )

        X = []
        y = []

        logger.info(f"Extracting features from {graph.number_of_edges():,} edges...")

        for i, (src, dst) in enumerate(graph.edges()):
            # Extract features
            features = self._extract_edge_features(graph, src, dst, feature_extractor)
            X.append(features)

            # Extract label
            label = graph.edges[src, dst].get('isFraud', 0)
            y.append(label)

            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i+1:,} edges...")

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        logger.info(f"Feature extraction complete: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Fraud rate: {y.mean()*100:.2f}%")

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train XGBoost model.

        Args:
            X_train: Training features (N, 42)
            y_train: Training labels (N,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
        """
        # Auto-compute scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            fraud_rate = y_train.mean()
            if fraud_rate > 0:
                self.scale_pos_weight = (1 - fraud_rate) / fraud_rate
                logger.info(f"Auto-computed scale_pos_weight: {self.scale_pos_weight:.1f}")
            else:
                self.scale_pos_weight = 1.0
                logger.warning("No fraud in training set! Using scale_pos_weight=1.0")

        # Initialize model
        self.model = self.xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric='aucpr',  # AUPR (not AUROC) for imbalanced data
            random_state=self.random_state,
            tree_method='hist',   # Fast histogram-based
            enable_categorical=False
        )

        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # Train
        logger.info("Training XGBoost...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            verbose=verbose
        )

        logger.info("Training complete!")

        # Store feature names for interpretability
        self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities.

        Args:
            X: Features (N, 42)

        Returns:
            Probabilities (N,) - fraud probability for each transaction
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Return probability of fraud (class 1)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict fraud labels.

        Args:
            X: Features (N, 42)
            threshold: Classification threshold (default 0.5)

        Returns:
            Predictions (N,) - 0=normal, 1=fraud
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_k: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })

        df = df.sort_values('importance', ascending=False).head(top_k)
        return df

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        self.model = self.xgb.XGBClassifier()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")


def train_xgboost_baseline(
    train_graph: nx.DiGraph,
    test_graph: nx.DiGraph,
    feature_extractor = None,
    val_split: float = 0.2,
    verbose: bool = True
) -> Tuple[XGBoostBaseline, Dict[str, float]]:
    """
    Train XGBoost baseline and evaluate.

    Args:
        train_graph: Training graph
        test_graph: Test graph
        feature_extractor: NodeFeatureExtractor instance (or None to create)
        val_split: Fraction of train to use for validation
        verbose: Print progress

    Returns:
        model: Trained XGBoost model
        results: Dictionary with performance metrics
    """
    from rl_money_laundering.features.extractors import NodeFeatureExtractor
    from rl_money_laundering.evaluation import BudgetedMetrics

    # Initialize feature extractor
    if feature_extractor is None:
        feature_extractor = NodeFeatureExtractor(
            include_temporal=True,
            include_network=True,
            node_feature_dim=20
        )

    # Initialize baseline
    baseline = XGBoostBaseline()

    # Extract features
    logger.info("Extracting training features...")
    X_train_full, y_train_full = baseline.prepare_features(train_graph, feature_extractor)

    logger.info("Extracting test features...")
    X_test, y_test = baseline.prepare_features(test_graph, feature_extractor)

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_split,
        stratify=y_train_full,
        random_state=42
    )

    logger.info(f"Train: {len(X_train):,} samples")
    logger.info(f"Val: {len(X_val):,} samples")
    logger.info(f"Test: {len(X_test):,} samples")

    # Train
    baseline.train(X_train, y_train, X_val, y_val, verbose=verbose)

    # Evaluate
    logger.info("\nEvaluating on test set...")
    y_pred_proba = baseline.predict_proba(X_test)

    metrics = BudgetedMetrics()

    results = {
        'aupr': metrics.aupr(y_pred_proba, y_test),
        'recall@50': metrics.recall_at_k(y_pred_proba, y_test, 50),
        'recall@100': metrics.recall_at_k(y_pred_proba, y_test, 100),
        'recall@200': metrics.recall_at_k(y_pred_proba, y_test, 200),
        'precision@50': metrics.precision_at_k(y_pred_proba, y_test, 50),
        'precision@100': metrics.precision_at_k(y_pred_proba, y_test, 100),
        'precision@200': metrics.precision_at_k(y_pred_proba, y_test, 200),
    }

    # Print summary
    metrics.print_report(y_pred_proba, y_test, budget_levels=[50, 100, 200, 500])

    return baseline, results
