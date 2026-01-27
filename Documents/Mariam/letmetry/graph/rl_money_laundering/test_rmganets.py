"""
Test RMGANets Implementation

Verifies that all RMGANets modules integrate correctly with existing pipeline.
"""
import torch

from rl_money_laundering.datasets import AMLNetConfig, AMLNetLoader
from rl_money_laundering.features.network_pyg import add_network_features_pyg
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.agent import DQNAgent
from rl_money_laundering.utils.features import AMLNetFeatureExtractor

print("=" * 80)
print("RMGANETS INTEGRATION TEST")
print("=" * 80)

# Test 1: Load Data
print("\n[1/8] Loading AMLNet data...")
config = AMLNetConfig(csv_path="../AMLNet_August 2025.csv")
loader = AMLNetLoader(config)
loader.load_raw(nrows=5000)
graph = loader.build_graph()
print(f"OK - Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

# Test 2: Add Network Features
print("\n[2/8] Adding network features (PyG)...")
graph = add_network_features_pyg(graph, device='cpu')
print("OK - Network features added")

# Test 3: Create RMGANets Encoder
print("\n[3/8] Creating RMGANets StateEncoder...")
try:
    encoder_rmganets = StateEncoder(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=64,
        history_dim=16
    )
    print("OK - RMGANets encoder created")
    print(f"  GNN type: {encoder_rmganets.gnn_type}")
    print(f"  State dim: {encoder_rmganets.state_dim}")
    print(f"  Device: {encoder_rmganets.device}")
except Exception as e:
    print(f"FAILED - {e}")
    raise

# Test 4: Compare with baseline encoders
print("\n[4/8] Creating baseline encoders for comparison...")
encoder_sage = StateEncoder(
    node_feature_dim=12,
    gnn_type="sage",
    embedding_dim=64,
    history_dim=16
)
print("OK - GraphSAGE encoder created")

encoder_gat = StateEncoder(
    node_feature_dim=12,
    gnn_type="gat",
    embedding_dim=64,
    history_dim=16
)
print("OK - GAT encoder created")

# Test 5: Environment Integration
print("\n[5/8] Creating environment...")
env = AMLDetectionEnv(
    graph=graph,
    max_steps=20,
    max_neighbors=5
)
obs, info = env.reset()
print("OK - Environment created and reset")
print(f"  Current node: {env.current_node}")

# Test 6: Encode State with RMGANets
print("\n[6/8] Encoding state with RMGANets...")
feature_extractor = AMLNetFeatureExtractor()

try:
    with torch.no_grad():
        state_rmganets = encoder_rmganets.encode_state(
            graph=graph,
            current_node=env.current_node,
            visited_nodes=set([env.current_node]),
            visited_edges=[],
            node_feature_extractor=feature_extractor
        )
    print("OK - State encoded with RMGANets")
    print(f"  State shape: {state_rmganets.shape}")
    print(f"  State stats: min={state_rmganets.min():.3f}, max={state_rmganets.max():.3f}, mean={state_rmganets.mean():.3f}")
except Exception as e:
    print(f"FAILED - {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 7: Compare encoding quality
print("\n[7/8] Comparing encoding quality across GNN types...")

with torch.no_grad():
    state_sage = encoder_sage.encode_state(
        graph=graph,
        current_node=env.current_node,
        visited_nodes=set([env.current_node]),
        visited_edges=[],
        node_feature_extractor=feature_extractor
    )

    state_gat = encoder_gat.encode_state(
        graph=graph,
        current_node=env.current_node,
        visited_nodes=set([env.current_node]),
        visited_edges=[],
        node_feature_extractor=feature_extractor
    )

print("OK - All encoders produce valid states")
print(f"  RMGANets: shape={state_rmganets.shape}, mean={state_rmganets.mean():.3f}")
print(f"  GraphSAGE: shape={state_sage.shape}, mean={state_sage.mean():.3f}")
print(f"  GAT: shape={state_gat.shape}, mean={state_gat.mean():.3f}")

# Test 8: DQN Agent Integration
print("\n[8/8] Testing DQN agent with RMGANets encoding...")
agent = DQNAgent(
    state_dim=encoder_rmganets.state_dim,
    action_dim=env.action_space.n,
    hidden_dims=[128, 128, 64],
    learning_rate=0.001
)

state_tensor = torch.FloatTensor(state_rmganets).unsqueeze(0)
with torch.no_grad():
    action = agent.select_action(state_tensor, epsilon=0.1)

print(f"OK - Agent selected action: {action}")

# Test RMGANets module architecture
print("\n" + "=" * 80)
print("RMGANETS MODULE ARCHITECTURE")
print("=" * 80)

rmganets_gnn = encoder_rmganets.gnn
print("\nRMGANetsEncoder:")
print(f"  Input dim: {rmganets_gnn.node_feature_dim}")
print(f"  Hidden dim: {rmganets_gnn.hidden_dim}")
print(f"  Embedding dim: {rmganets_gnn.embedding_dim}")
print("  Attention heads: 8 (from paper)")
print("  T1 (high threshold): 0.7")
print("  T2 (low threshold): 0.3")

print("\nModules:")
print("  [1] Att-GCM: Multi-head attention + subgraph splitting")
print("  [2] To-GCM: Adaptive topology convolution (medium/high)")
print("  [3] HyGCM: Hybrid convolution (high/low)")
print("  [4] Fusion: Feature concatenation (CRITICAL +11% F1)")

print("\n" + "=" * 80)
print("ALL RMGANETS TESTS PASSED!")
print("=" * 80)

print("\nSummary:")
print("  [OK] RMGANets modules created")
print("  [OK] Integration with StateEncoder")
print("  [OK] End-to-end state encoding")
print("  [OK] DQN agent integration")
print("  [OK] All 3 GNN types working (SAGE, GAT, RMGANets)")

print("\nNext steps:")
print("  1. Train agent with RMGANets encoder")
print("  2. Compare performance: SAGE vs GAT vs RMGANets")
print("  3. Monitor subgraph split statistics")
print("  4. Tune thresholds T1, T2 if needed")
