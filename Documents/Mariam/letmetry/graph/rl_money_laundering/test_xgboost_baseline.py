"""
Test XGBoost Baseline on AMLNet Data

Demonstrates traditional ML baseline for comparison with RL approach.
"""

import sys
from pathlib import Path

import logging

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.features.extractors import NodeFeatureExtractor
from rl_money_laundering.baselines import XGBoostBaseline
from rl_money_laundering.evaluation import BudgetedMetrics
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

print("=" * 70)
print("XGBOOST BASELINE TEST")
print("=" * 70)

# 1. Load dataset (small sample for quick test)
print("\n[1/6] Loading AMLNet dataset...")
dataset_path = Path("..") / "AMLNet_August 2025.csv"
config = AMLNetConfig(csv_path=str(dataset_path))
loader = AMLNetLoader(config)

# Load 20K transactions (to ensure we have fraud cases in both train/test)
loader.load_raw(nrows=20000)
print(f"      Loaded {len(loader.transactions):,} transactions")
print(f"      Fraud: {loader.transactions['isFraud'].sum()} ({loader.transactions['isFraud'].mean()*100:.2f}%)")

# 2. Build graph
print("\n[2/6] Building graph with features...")
graph = loader.build_graph()
print(f"      Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# Add temporal and network features
graph = add_temporal_features_to_graph(graph, loader.transactions)
graph = add_network_features_to_graph(graph, compute_expensive=False)
print("      Temporal + network features added")

# 3. Split into train/test (70/30 split) with stratification
print("\n[3/6] Splitting train/test...")
edges = list(graph.edges())

# Get labels for stratification
labels = [graph.edges[e].get('isFraud', 0) for e in edges]

# Stratified split to ensure fraud in both train/test
train_edges, test_edges = train_test_split(
    edges,
    test_size=0.3,
    random_state=42,
    stratify=labels  # Ensure fraud distributed across train/test
)

train_graph = graph.edge_subgraph(train_edges).copy()
test_graph = graph.edge_subgraph(test_edges).copy()

print(f"      Train: {train_graph.number_of_edges():,} edges")
print(f"      Test: {test_graph.number_of_edges():,} edges")

# Verify fraud in both splits
train_fraud = sum(train_graph.edges[e].get('isFraud', 0) for e in train_graph.edges())
test_fraud = sum(test_graph.edges[e].get('isFraud', 0) for e in test_graph.edges())
print(f"      Train fraud: {train_fraud}")
print(f"      Test fraud: {test_fraud}")

# 4. Initialize feature extractor
print("\n[4/6] Initializing feature extractor...")
feature_extractor = NodeFeatureExtractor(
    include_temporal=True,
    include_network=True,
    node_feature_dim=20
)
print("      Feature extractor ready (20-dim vectors)")

# 5. Train XGBoost
print("\n[5/6] Training XGBoost baseline...")
baseline = XGBoostBaseline(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=50,  # Reduced for quick test
    random_state=42
)

# Extract features
X_train, y_train = baseline.prepare_features(train_graph, feature_extractor)
X_test, y_test = baseline.prepare_features(test_graph, feature_extractor)

print(f"      Train features: {X_train.shape}")
print(f"      Test features: {X_test.shape}")
print(f"      Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"      Test fraud rate: {y_test.mean()*100:.2f}%")

# Train model
baseline.train(X_train, y_train, verbose=False)
print("      Training complete!")

# 6. Evaluate
print("\n[6/6] Evaluating on test set...")
y_pred_proba = baseline.predict_proba(X_test)

metrics = BudgetedMetrics()
metrics.print_report(y_pred_proba, y_test, budget_levels=[10, 50, 100])

# Feature importance
print("\n" + "-" * 70)
print("Top 10 Most Important Features:")
print("-" * 70)
importance_df = baseline.feature_importance(top_k=10)
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

print("\n" + "=" * 70)
print("XGBOOST BASELINE TEST COMPLETE")
print("=" * 70)
print("\nKey Results:")

aupr = metrics.aupr(y_pred_proba, y_test)
recall_50 = metrics.recall_at_k(y_pred_proba, y_test, 50)
recall_100 = metrics.recall_at_k(y_pred_proba, y_test, 100)

print(f"  AUPR: {aupr:.3f}")
print(f"  Recall@50: {recall_50:.1%}")
print(f"  Recall@100: {recall_100:.1%}")

print("\nThis is the baseline your RL approach needs to beat!")
print("Advantages of RL over XGBoost:")
print("  1. Graph traversal (exploration)")
print("  2. Sequential decision-making")
print("  3. Can handle alert budgets directly")
print("  4. Learns from interaction (online learning)")
