"""
Simple end-to-end pipeline test with GNN encoder
"""
import torch

from rl_money_laundering.datasets import AMLNetConfig, AMLNetLoader
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.features.network_pyg import add_network_features_pyg
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.utils.features import AMLNetFeatureExtractor

print("=" * 80)
print("SIMPLE PIPELINE TEST")
print("=" * 80)

# Test 1: Load data
print("\n[1/6] Loading data...")

config = AMLNetConfig(csv_path="../AMLNet_August 2025.csv")
loader = AMLNetLoader(config)
loader.load_raw(nrows=20000)
graph = loader.build_graph()
print(f"OK - Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

# Test 2: Add features
print("\n[2/6] Adding features...")
graph = add_network_features_pyg(graph, device='cpu')
print("OK - Features added")

sample_node = list(graph.nodes())[0]
node_data = graph.nodes[sample_node]
print(f"  Sample node '{sample_node}' has {len(node_data)} attributes")

# Test 3: Create GNN encoder
print("\n[3/6] Creating GNN encoder...")

encoder = StateEncoder(
    node_feature_dim=12,
    gnn_type="sage",
    embedding_dim=64,
    history_dim=16
)
print("OK - GNN encoder created")
print("  Input dim: 12")
print("  Embedding dim: 64")
print("  History dim: 16")

# Test 4: Create environment
print("\n[4/6] Creating environment...")

env = AMLDetectionEnv(
    graph=graph,
    max_steps=20,
    max_neighbors=5
)
print("OK - Environment created")
print(f"  Action space: {env.action_space}")
print(f"  Observation space shape: {env.observation_space.shape}")

# Test 5: Reset environment and encode state with GNN
print("\n[5/6] Resetting environment and encoding state...")
obs, info = env.reset()
print("OK - Environment reset")
print(f"  Observation shape: {obs.shape}")
print(f"  Current node: {env.current_node}")

# Create feature extractor for GNN
feature_extractor = AMLNetFeatureExtractor()

# Encode state with GNN
with torch.no_grad():
    encoded_state = encoder.encode_state(
        graph=graph,
        current_node=env.current_node,
        visited_nodes=set([env.current_node]),
        visited_edges=[],
        node_feature_extractor=feature_extractor
    )
print("OK - State encoded with GNN")
print(f"  Encoded state shape: {encoded_state.shape}")

# Test 6: Take steps
print("\n[6/6] Taking steps in environment...")
total_reward = 0
for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"  Step {step + 1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")

    if terminated or truncated:
        print("  Episode ended early")
        break

print(f"\nOK - Completed {step + 1} steps, total reward: {total_reward:.3f}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
