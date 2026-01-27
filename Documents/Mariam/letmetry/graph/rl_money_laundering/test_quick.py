"""
Quick Pipeline Test - Streamlined for fast testing
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.features.extractors import NodeFeatureExtractor

print("=" * 80)
print("QUICK PIPELINE TEST")
print("=" * 80)

# 1. Load small dataset
print("\n[1/6] Loading dataset...")
dataset_path = Path("..") / "AMLNet_August 2025.csv"
config = AMLNetConfig(csv_path=str(dataset_path))
loader = AMLNetLoader(config)
loader.load_raw(nrows=10000)  # Only 10K rows for speed
print(f"✓ Loaded {len(loader.transactions):,} transactions")

# 2. Build graph
print("\n[2/6] Building graph...")
graph = loader.build_graph()
print(f"✓ Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# 3. Add features (skip expensive network features)
print("\n[3/6] Adding temporal features...")
graph = add_temporal_features_to_graph(graph, loader.transactions)
print("✓ Temporal features added")

print("\n[4/6] Adding network features (fast mode)...")
graph = add_network_features_to_graph(graph, compute_expensive=False)
print("✓ Network features added (skipped betweenness)")

# 4. Test environment
print("\n[5/6] Testing environment...")
env = AMLDetectionEnv(graph=graph, max_steps=20, max_neighbors=3)
obs, info = env.reset()
print("✓ Environment initialized")
print(f"  Starting node: {info['current_node']}")
print(f"  Observation shape: {obs.shape}")

# Test a few steps
print("\n  Running 5 environment steps...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"    Step {i+1}: action={action}, reward={reward:.2f}, nodes_visited={info['visited_nodes']}")
    if terminated or truncated:
        print("    Episode ended early")
        break

print("✓ Environment steps completed")

# 5. Test feature extraction
print("\n[6/6] Testing feature extraction...")
feature_extractor = NodeFeatureExtractor(
    include_temporal=True,
    include_network=True,
    node_feature_dim=20
)

sample_node = list(graph.nodes())[0]
node_data = graph.nodes[sample_node]
features = feature_extractor.extract(node_data)

print("✓ Feature extraction working")
print(f"  Sample node: {sample_node}")
print(f"  Feature shape: {features.shape}")
print(f"  Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")

# Summary
print("\n" + "=" * 80)
print("✅ ALL QUICK TESTS PASSED")
print("=" * 80)
print("\nComponents verified:")
print("  ✓ Data loading (10K transactions)")
print("  ✓ Graph construction")
print("  ✓ Temporal features (velocity, business hours)")
print("  ✓ Network features (degree, clustering)")
print("  ✓ Environment (initialization + steps)")
print("  ✓ Feature extraction (20-dim vectors)")
print("\nReady for full training!")
