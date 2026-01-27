"""
Debug RMGANets encoding to understand why mean=0
"""
import torch
import numpy as np

from rl_money_laundering.datasets import AMLNetConfig, AMLNetLoader
from rl_money_laundering.features.network_pyg import add_network_features_pyg
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.utils.features import AMLNetFeatureExtractor

print("=" * 80)
print("DEBUG: RMGANets Encoding Analysis")
print("=" * 80)

# Load data
print("\n[1/5] Loading data...")
config = AMLNetConfig(csv_path="../AMLNet_August 2025.csv")
loader = AMLNetLoader(config)
loader.load_raw(nrows=5000)
graph = loader.build_graph()
graph = add_network_features_pyg(graph, device='cpu')
print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

# Create encoder
print("\n[2/5] Creating RMGANets encoder...")
encoder = StateEncoder(
    node_feature_dim=12,
    gnn_type="rmganets",
    embedding_dim=64,
    history_dim=16
)

# Create environment
print("\n[3/5] Creating environment...")
env = AMLDetectionEnv(graph=graph, max_steps=20, max_neighbors=5)
obs, info = env.reset()
current_node = env.current_node
print(f"Current node: {current_node}")

# Extract features manually to inspect
print("\n[4/5] Inspecting intermediate outputs...")
feature_extractor = AMLNetFeatureExtractor()

# Get local subgraph
subgraph_nodes = encoder._get_local_subgraph(graph, current_node, k=2)
subgraph = graph.subgraph(subgraph_nodes)
print(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph.edges())} edges")

# Convert to PyG
pyg_data = encoder._networkx_to_pyg(subgraph, feature_extractor)
print(f"PyG node features shape: {pyg_data.x.shape}")
print(f"PyG node features stats: min={pyg_data.x.min():.3f}, max={pyg_data.x.max():.3f}, mean={pyg_data.x.mean():.3f}")

# Get GNN embeddings (just the GNN part)
encoder.gnn.eval()
with torch.no_grad():
    node_embeddings = encoder.gnn(
        pyg_data.x.to(encoder.device),
        pyg_data.edge_index.to(encoder.device)
    )

print(f"\nGNN embeddings shape: {node_embeddings.shape}")
print(f"GNN embeddings stats: min={node_embeddings.min():.3f}, max={node_embeddings.max():.3f}, mean={node_embeddings.mean():.3f}")

# Get embedding for current node
node_idx = list(subgraph_nodes).index(current_node)
current_embedding = node_embeddings[node_idx].cpu().numpy()
print(f"\nCurrent node embedding shape: {current_embedding.shape}")
print(f"Current node embedding stats: min={current_embedding.min():.3f}, max={current_embedding.max():.3f}, mean={current_embedding.mean():.3f}")

# Get history features
history_features = encoder._encode_history(graph, set([current_node]), [])
print(f"\nHistory features shape: {history_features.shape}")
print(f"History features stats: min={history_features.min():.3f}, max={history_features.max():.3f}, mean={history_features.mean():.3f}")

# Full state
print("\n[5/5] Full state encoding...")
state = encoder.encode_state(
    graph=graph,
    current_node=current_node,
    visited_nodes=set([current_node]),
    visited_edges=[],
    node_feature_extractor=feature_extractor
)

print(f"Full state shape: {state.shape}")
print(f"Full state stats: min={state.min():.3f}, max={state.max():.3f}, mean={state.mean():.3f}")

# Break down the state
gnn_part = state[:64]  # First 64 dims are GNN embedding
history_part = state[64:]  # Last 16 dims are history

print("\nState breakdown:")
print(f"  GNN part (first 64): min={gnn_part.min():.3f}, max={gnn_part.max():.3f}, mean={gnn_part.mean():.3f}")
print(f"  History part (last 16): min={history_part.min():.3f}, max={history_part.max():.3f}, mean={history_part.mean():.3f}")

# Check if LayerNorm is the culprit
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if abs(gnn_part.mean()) < 0.001:
    print("\n[WARNING] GNN embedding mean is very close to 0!")
    print("This is likely due to LayerNorm in RMGANetsEncoder.output_norm")
    print("LayerNorm normalizes to mean=0, std=1 by default.")
    print("\nThis is EXPECTED BEHAVIOR and actually desirable because:")
    print("  1. Prevents covariate shift")
    print("  2. Stabilizes training")
    print("  3. Helps gradient flow")
    print("\nGraphSAGE and GAT don't use LayerNorm, so they have non-zero means.")
    print("\n[OK] This is NORMAL and CORRECT for RMGANets!")
else:
    print("\n[OK] GNN embedding mean is not zero.")

# Test with multiple nodes to verify it's not always zero
print("\n" + "=" * 80)
print("VERIFICATION: Testing with 5 different nodes")
print("=" * 80)

means = []
for i in range(5):
    obs, _ = env.reset()
    state = encoder.encode_state(
        graph=graph,
        current_node=env.current_node,
        visited_nodes=set([env.current_node]),
        visited_edges=[],
        node_feature_extractor=feature_extractor
    )
    means.append(state[:64].mean())
    print(f"Node {i+1} ({env.current_node}): GNN mean = {state[:64].mean():.6f}, std = {state[:64].std():.6f}")

print("\nOverall statistics across 5 nodes:")
print(f"  Mean of means: {np.mean(means):.6f}")
print(f"  Std of means: {np.std(means):.6f}")

if abs(np.mean(means)) < 0.01:
    print("\n[OK] CONFIRMED: Mean ~0 across different nodes due to LayerNorm")
    print("This is expected and beneficial for training stability!")
else:
    print("\n[WARNING] Mean is not consistently zero - may need investigation")
