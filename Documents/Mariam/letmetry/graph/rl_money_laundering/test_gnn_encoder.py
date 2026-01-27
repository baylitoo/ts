"""
Test GNN Encoder (StateEncoder) with Graph Data
Tests GraphSAGE and GAT encoders with actual graph
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from rl_money_laundering.datasets import AMLNetLoader, AMLNetConfig
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.features.extractors import NodeFeatureExtractor
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.environment import AMLDetectionEnv

print("\n" + "=" * 70)
print("GNN ENCODER TEST")
print("=" * 70)

# 1. Load small dataset and build graph
print("\n[1/5] Loading dataset and building graph...")
config = AMLNetConfig(csv_path=str(Path("..") / "AMLNet_August 2025.csv"))
loader = AMLNetLoader(config)
loader.load_raw(nrows=2000)  # 2K rows for reasonable graph size
graph = loader.build_graph()
print(f"      ✓ Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# 2. Add features
print("\n[2/5] Adding temporal and network features...")
graph = add_temporal_features_to_graph(graph, loader.transactions)
graph = add_network_features_to_graph(graph, compute_expensive=False)
print("      ✓ Features added to graph nodes")

# 3. Create feature extractor
print("\n[3/5] Creating node feature extractor...")
feature_extractor = NodeFeatureExtractor(
    include_temporal=True,
    include_network=True,
    node_feature_dim=20
)

def node_feature_fn(node_data):
    return feature_extractor.extract(node_data)

# Test extraction on a sample node
sample_node = list(graph.nodes())[0]
sample_features = node_feature_fn(graph.nodes[sample_node])
print("      ✓ Feature extractor created")
print(f"      Sample features shape: {sample_features.shape}")
print(f"      Sample features: {sample_features[:5]}...")

# 4. Test GraphSAGE encoder
print("\n[4/5] Testing GraphSAGE encoder...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"      Device: {device}")

state_encoder_sage = StateEncoder(
    node_feature_dim=20,
    gnn_type="sage",
    embedding_dim=64,
    history_dim=32,
    device=device
)
print("      ✓ GraphSAGE encoder created")
print("        - Node feature dim: 20")
print("        - Embedding dim: 64")
print("        - History dim: 32")
print(f"        - Total state dim: {state_encoder_sage.get_state_dim()}")

# Encode a state
current_node = list(graph.nodes())[10]
visited_nodes = {current_node}
visited_edges = []

print(f"\n      Encoding state for node: {current_node}")
state_embedding_sage = state_encoder_sage.encode_state(
    graph=graph,
    current_node=current_node,
    visited_nodes=visited_nodes,
    visited_edges=visited_edges,
    node_feature_extractor=node_feature_fn
)

print("      ✓ State encoded successfully")
print(f"        - Output shape: {state_embedding_sage.shape}")
print(f"        - Output dtype: {state_embedding_sage.dtype}")
print(f"        - Stats: min={state_embedding_sage.min():.4f}, max={state_embedding_sage.max():.4f}, mean={state_embedding_sage.mean():.4f}")

# 5. Test GAT encoder
print("\n[5/5] Testing GAT encoder...")
state_encoder_gat = StateEncoder(
    node_feature_dim=20,
    gnn_type="gat",
    embedding_dim=64,
    history_dim=32,
    device=device
)
print("      ✓ GAT encoder created")

state_embedding_gat = state_encoder_gat.encode_state(
    graph=graph,
    current_node=current_node,
    visited_nodes=visited_nodes,
    visited_edges=visited_edges,
    node_feature_extractor=node_feature_fn
)

print("      ✓ State encoded successfully")
print(f"        - Output shape: {state_embedding_gat.shape}")
print(f"        - Output dtype: {state_embedding_gat.dtype}")
print(f"        - Stats: min={state_embedding_gat.min():.4f}, max={state_embedding_gat.max():.4f}, mean={state_embedding_gat.mean():.4f}")

# 6. Test with environment
print("\n[BONUS] Testing with environment...")
env = AMLDetectionEnv(graph=graph, max_steps=10, max_neighbors=3)
obs, info = env.reset()
current_node = info['current_node']

print(f"      Environment reset at node: {current_node}")

# Encode initial state
initial_state = state_encoder_sage.encode_state(
    graph=graph,
    current_node=current_node,
    visited_nodes={current_node},
    visited_edges=[],
    node_feature_extractor=node_feature_fn
)
print(f"      ✓ Initial state encoded: {initial_state.shape}")

# Take a few steps and encode states
print("\n      Running 3 environment steps with state encoding...")
visited_nodes = {current_node}
visited_edges = []

for i in range(3):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    # Update visited tracking
    visited_nodes.add(env.current_node)
    if len(env.visited_edges) > len(visited_edges):
        visited_edges = list(env.visited_edges)

    # Encode new state
    state = state_encoder_sage.encode_state(
        graph=graph,
        current_node=env.current_node,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        node_feature_extractor=node_feature_fn
    )

    print(f"        Step {i+1}: node={env.current_node}, state_shape={state.shape}, reward={reward:.2f}")

    if done or truncated:
        print("        Episode ended")
        break

print("\n" + "=" * 70)
print("✅ GNN ENCODER TEST PASSED")
print("=" * 70)
print("\nVerified:")
print("  ✓ GraphSAGE encoder (sage)")
print("  ✓ GAT encoder (gat)")
print("  ✓ State encoding from graph")
print("  ✓ Feature extraction (20-dim)")
print("  ✓ Integration with environment")
print("  ✓ Multi-step state encoding")
print(f"\nState dimension: {state_encoder_sage.get_state_dim()}")
print("Ready for DQN training!")
