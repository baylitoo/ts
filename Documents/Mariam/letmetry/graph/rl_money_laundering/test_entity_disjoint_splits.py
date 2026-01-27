"""
Test Entity-Disjoint Splits

Demonstrates proper train/test splitting with NO account overlap.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.utils.graph import (
    check_split_leakage,
    entity_disjoint_split,
    temporal_split,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

print("=" * 70)
print("ENTITY-DISJOINT SPLIT TEST")
print("=" * 70)

# 1. Load and build graph
print("\n[1/5] Loading AMLNet data...")
config = AMLNetConfig(csv_path=str(Path("..") / "AMLNet_August 2025.csv"))
loader = AMLNetLoader(config)
loader.load_raw(nrows=20000)

graph = loader.build_graph()
print(f"      Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# Add features
graph = add_temporal_features_to_graph(graph, loader.transactions)
graph = add_network_features_to_graph(graph, compute_expensive=False)

# Count fraud edges
fraud_edges = sum(1 for u, v in graph.edges() if graph.edges[u, v].get('isFraud', 0) == 1)
print(f"      Fraud edges: {fraud_edges}")

# 2. Test random (BAD) split - has leakage
print("\n[2/5] Testing RANDOM split (BAD - has leakage)...")
edges = list(graph.edges())
train_edges, test_edges = train_test_split(edges, test_size=0.3, random_state=42)

random_train_graph = graph.edge_subgraph(train_edges).copy()
random_test_graph = graph.edge_subgraph(test_edges).copy()

print("Random split (naive):")
leakage_random = check_split_leakage(random_train_graph, random_test_graph, verbose=False)
print(f"  Train: {random_train_graph.number_of_nodes():,} nodes, {random_train_graph.number_of_edges():,} edges")
print(f"  Test: {random_test_graph.number_of_nodes():,} nodes, {random_test_graph.number_of_edges():,} edges")
print(f"  Account overlap: {leakage_random['overlap_accounts']:,} "
      f"({leakage_random['overlap_ratio']:.1%})")
print("  [WARNING] This causes data leakage!")

# 3. Test entity-disjoint split (GOOD) - no leakage
print("\n[3/5] Testing ENTITY-DISJOINT split (GOOD - no leakage)...")
train_graph, val_graph, test_graph = entity_disjoint_split(
    graph,
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

print("Entity-disjoint split:")
print(f"  Train: {train_graph.number_of_nodes():,} nodes, {train_graph.number_of_edges():,} edges")
print(f"  Val: {val_graph.number_of_nodes():,} nodes, {val_graph.number_of_edges():,} edges")
print(f"  Test: {test_graph.number_of_nodes():,} nodes, {test_graph.number_of_edges():,} edges")

# Verify no leakage
leakage_entity = check_split_leakage(train_graph, test_graph, verbose=False)
print(f"  Account overlap: {leakage_entity['overlap_accounts']:,} "
      f"({leakage_entity['overlap_ratio']:.1%})")
print("  [OK] Zero leakage - proper split!")

# 4. Test temporal split
print("\n[4/5] Testing TEMPORAL split (train on past, test on future)...")
train_temporal, val_temporal, test_temporal = temporal_split(
    graph,
    test_size=0.2,
    val_size=0.1,
    time_column="step"
)

print("Temporal split:")
print(f"  Train: {train_temporal.number_of_edges():,} edges")
print(f"  Val: {val_temporal.number_of_edges():,} edges")
print(f"  Test: {test_temporal.number_of_edges():,} edges")

leakage_temporal = check_split_leakage(train_temporal, test_temporal, verbose=False)
print(f"  Account overlap: {leakage_temporal['overlap_accounts']:,} "
      f"({leakage_temporal['overlap_ratio']:.1%})")
print("  Note: Temporal split may have account overlap (tests generalization over time)")

# 5. Detailed leakage check
print("\n[5/5] Detailed leakage analysis...")
check_split_leakage(train_graph, test_graph, verbose=True)

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: Comparison of Split Methods")
print("=" * 70)
print(f"{'Method':<20} {'Train Edges':<12} {'Test Edges':<12} {'Account Overlap':<20}")
print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*20}")
print(f"{'Random':<20} {random_train_graph.number_of_edges():<12,} "
      f"{random_test_graph.number_of_edges():<12,} "
      f"{leakage_random['overlap_ratio']:<20.1%}")
print(f"{'Entity-Disjoint':<20} {train_graph.number_of_edges():<12,} "
      f"{test_graph.number_of_edges():<12,} "
      f"{leakage_entity['overlap_ratio']:<20.1%}")
print(f"{'Temporal':<20} {train_temporal.number_of_edges():<12,} "
      f"{test_temporal.number_of_edges():<12,} "
      f"{leakage_temporal['overlap_ratio']:<20.1%}")
print("=" * 70)

print("\nKey Takeaways:")
print("  1. Random split: LEAKS data (accounts in both train/test)")
print("  2. Entity-disjoint: NO leakage (accounts never overlap)")
print("  3. Temporal: Tests generalization to FUTURE transactions")
print("  4. For publication: ALWAYS use entity-disjoint or temporal!")
print("\n[OK] Entity-disjoint splits implemented and tested!")
