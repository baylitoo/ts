"""
MINIMAL PIPELINE TEST - Ultra-fast with tiny dataset
Tests: Load 1K rows -> Build graph -> Add features -> Run environment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.features.extractors import NodeFeatureExtractor

print("\n" + "=" * 60)
print("MINIMAL PIPELINE TEST (1K transactions)")
print("=" * 60)

# Load TINY dataset
print("\n[1/5] Loading 1,000 transactions...")
config = AMLNetConfig(csv_path=str(Path("..") / "AMLNet_August 2025.csv"))
loader = AMLNetLoader(config)
loader.load_raw(nrows=1000)
print(f"      ✓ Loaded {len(loader.transactions):,} transactions")
print(f"      Fraud: {loader.transactions['isFraud'].sum()} ({loader.transactions['isFraud'].mean()*100:.2f}%)")

# Build graph
print("\n[2/5] Building graph...")
graph = loader.build_graph()
print(f"      ✓ {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# Add features
print("\n[3/5] Adding temporal + network features...")
graph = add_temporal_features_to_graph(graph, loader.transactions)
graph = add_network_features_to_graph(graph, compute_expensive=False)
sample_node = list(graph.nodes())[0]
print("      ✓ Features added")
print(f"      Sample node '{sample_node}':")
print(f"        - tx_velocity: {graph.nodes[sample_node].get('tx_velocity', 0):.2f}")
print(f"        - business_hour_ratio: {graph.nodes[sample_node].get('business_hour_ratio', 0):.2f}")
print(f"        - degree_centrality: {graph.nodes[sample_node].get('degree_centrality', 0):.4f}")

# Test environment
print("\n[4/5] Testing environment (10 steps)...")
env = AMLDetectionEnv(graph=graph, max_steps=10, max_neighbors=3)
obs, info = env.reset()
print(f"      ✓ Environment initialized at node: {info['current_node']}")

for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        print(f"      Episode ended at step {i+1}")
        break
else:
    print("      Completed 10 steps")

print(f"      Final stats: {info['visited_nodes']} nodes visited, {info['fraud_edges_found']} fraud edges")

# Test feature extractor
print("\n[5/5] Testing feature extraction...")
extractor = NodeFeatureExtractor(include_temporal=True, include_network=True, node_feature_dim=20)
features = extractor.extract(graph.nodes[sample_node])
print(f"      ✓ Extracted {features.shape[0]}-dimensional feature vector")
print(f"      Features: min={features.min():.2f}, max={features.max():.2f}, mean={features.mean():.2f}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - System ready!")
print("=" * 60)
