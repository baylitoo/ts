"""
Train and Evaluate Baseline Methods

Trains PageRank, Random Forest, and Static GNN baselines for comparison.

Usage:
    python scripts/train_baselines.py --dataset amlnet
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rl_money_laundering.baselines import (
    PageRankBaseline,
    RandomForestBaseline,
    StaticGNNBaseline,
)
from rl_money_laundering.data_loader import AMLNetDataLoader
from rl_money_laundering.evaluator import AMLEvaluator


def main():
    parser = argparse.ArgumentParser(description="Train baseline methods")

    parser.add_argument("--dataset", type=str, default="amlnet",
                       choices=["amlnet", "elliptic"])
    parser.add_argument("--data_path", type=str, default="data/amlnet/AMLNet_August_2025.csv")
    parser.add_argument("--nrows", type=int, default=None,
                       help="Number of rows (None = all)")
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    parser.add_argument("--methods", type=str, default="all",
                       help="Methods to run (comma-separated or 'all')")

    args = parser.parse_args()

    print("=" * 60)
    print("Baseline Methods Training & Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Methods: {args.methods}")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading dataset...")
    loader = AMLNetDataLoader(args.data_path)
    df = loader.load_data(nrows=args.nrows)
    loader.extract_features()
    graph = loader.build_transaction_graph()

    # Create splits
    print("\n[2/3] Creating train/test splits...")
    train_df, val_df, test_df = loader.create_train_test_split(
        test_size=0.2,
        val_size=0.1
    )

    # Get node lists and labels
    train_nodes = list(set(train_df['nameOrig'].unique()) | set(train_df['nameDest'].unique()))
    test_nodes = list(set(test_df['nameOrig'].unique()) | set(test_df['nameDest'].unique()))

    # Create labels (for nodes in graph)
    def get_node_label(node_id):
        """Get fraud label for a node"""
        # Check if node appears in fraud transactions
        fraud_txs = df[df['isMoneyLaundering'] == 1]
        is_fraud = (
            (fraud_txs['nameOrig'] == node_id).any() or
            (fraud_txs['nameDest'] == node_id).any()
        )
        return 1 if is_fraud else 0

    train_labels = np.array([get_node_label(node) for node in train_nodes])
    test_labels = np.array([get_node_label(node) for node in test_nodes])

    print(f"Train: {len(train_nodes)} nodes ({train_labels.sum()} fraud)")
    print(f"Test: {len(test_nodes)} nodes ({test_labels.sum()} fraud)")

    # Initialize evaluator
    evaluator = AMLEvaluator()

    # Determine which methods to run
    if args.methods == "all":
        methods_to_run = ["pagerank", "rf", "gnn_sage", "gnn_gat"]
    else:
        methods_to_run = args.methods.split(",")

    print("\n[3/3] Training baselines...")
    print("=" * 60)

    # 1. PageRank Baseline
    if "pagerank" in methods_to_run:
        print("\n>>> PageRank Baseline")
        print("-" * 60)

        pagerank = PageRankBaseline()
        pagerank.fit(graph)

        # Predict on test set
        test_pred = pagerank.predict(test_nodes, threshold=0.95)  # Top 5% = fraud
        test_scores = pagerank.get_scores(test_nodes)

        # Evaluate
        evaluator.evaluate(
            y_true=test_labels,
            y_pred=test_pred,
            y_scores=test_scores,
            method_name="PageRank"
        )

        evaluator.print_report("PageRank", test_labels, test_pred)

    # 2. Random Forest Baseline
    if "rf" in methods_to_run:
        print("\n>>> Random Forest Baseline")
        print("-" * 60)

        rf = RandomForestBaseline(n_estimators=100)
        rf.fit(graph, train_nodes, train_labels)

        # Predict on test set
        test_pred = rf.predict(graph, test_nodes)
        test_scores = rf.predict_proba(graph, test_nodes)

        # Evaluate
        evaluator.evaluate(
            y_true=test_labels,
            y_pred=test_pred,
            y_scores=test_scores,
            method_name="Random Forest"
        )

        evaluator.print_report("Random Forest", test_labels, test_pred)

    # 3. GraphSAGE Baseline
    if "gnn_sage" in methods_to_run:
        print("\n>>> GraphSAGE Baseline")
        print("-" * 60)

        # Simple node features for GNN
        def node_feature_extractor(node_data):
            return np.array([
                float(node_data.get('total_sent', 0.0)),
                float(node_data.get('total_received', 0.0)),
                float(node_data.get('num_transactions_sent', 0)),
                float(node_data.get('num_transactions_received', 0)),
                node_data.get('risk_score', 0.0)
            ], dtype=np.float32)

        gnn_sage = StaticGNNBaseline(
            in_channels=5,
            hidden_channels=64,
            num_classes=2,
            gnn_type="sage"
        )

        # Train
        gnn_sage.fit(
            graph=graph,
            train_nodes=train_nodes,
            train_labels=train_labels,
            val_nodes=test_nodes[:len(test_nodes)//2],
            val_labels=test_labels[:len(test_labels)//2],
            epochs=100,
            node_feature_extractor=node_feature_extractor
        )

        # Predict
        test_pred = gnn_sage.predict(graph, test_nodes, node_feature_extractor)

        # Evaluate
        evaluator.evaluate(
            y_true=test_labels,
            y_pred=test_pred,
            method_name="GraphSAGE"
        )

        evaluator.print_report("GraphSAGE", test_labels, test_pred)

    # 4. GAT Baseline
    if "gnn_gat" in methods_to_run:
        print("\n>>> GAT Baseline")
        print("-" * 60)

        def node_feature_extractor(node_data):
            return np.array([
                float(node_data.get('total_sent', 0.0)),
                float(node_data.get('total_received', 0.0)),
                float(node_data.get('num_transactions_sent', 0)),
                float(node_data.get('num_transactions_received', 0)),
                node_data.get('risk_score', 0.0)
            ], dtype=np.float32)

        gnn_gat = StaticGNNBaseline(
            in_channels=5,
            hidden_channels=64,
            num_classes=2,
            gnn_type="gat"
        )

        # Train
        gnn_gat.fit(
            graph=graph,
            train_nodes=train_nodes,
            train_labels=train_labels,
            val_nodes=test_nodes[:len(test_nodes)//2],
            val_labels=test_labels[:len(test_labels)//2],
            epochs=100,
            node_feature_extractor=node_feature_extractor
        )

        # Predict
        test_pred = gnn_gat.predict(graph, test_nodes, node_feature_extractor)

        # Evaluate
        evaluator.evaluate(
            y_true=test_labels,
            y_pred=test_pred,
            method_name="GAT"
        )

        evaluator.print_report("GAT", test_labels, test_pred)

    # Comparison table
    print("\n" + "=" * 100)
    print("COMPARISON OF ALL METHODS")
    print("=" * 100)
    print(evaluator.compare_methods())

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(str(output_path / "baseline_results.json"))

    print(f"\nResults saved to: {output_path / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
