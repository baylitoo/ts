"""
Comprehensive test of all pipeline components
"""
import numpy as np
import torch

from rl_money_laundering.agent import DQNAgent
from rl_money_laundering.baselines.xgboost_baseline import XGBoostBaseline
from rl_money_laundering.datasets import AMLNetConfig, AMLNetLoader
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.evaluation.budgeted_metrics import BudgetedMetrics
from rl_money_laundering.features.network_pyg import add_network_features_pyg
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.utils.features import AMLNetFeatureExtractor
from rl_money_laundering.utils.graph import entity_disjoint_split

print("=" * 80)
print("COMPREHENSIVE PIPELINE TEST")
print("=" * 80)

# Test 1: Data Loading
print("\n[1/10] Data Loading (AMLNet)...")

config = AMLNetConfig(csv_path="../AMLNet_August 2025.csv")
loader = AMLNetLoader(config)
loader.load_raw(nrows=20000)
graph = loader.build_graph()
print(f"OK - Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

# Check fraud edges
fraud_edges = sum(1 for u, v in graph.edges() if graph.edges[u, v].get('isFraud', 0) == 1)
print(f"  Fraud edges: {fraud_edges}")

# Test 2: Network Features (PyG)
print("\n[2/10] Network Features (PyG)...")
graph = add_network_features_pyg(graph, device='cpu')
print("OK - Features added")

sample_node = list(graph.nodes())[0]
print(f"  Sample node '{sample_node}':")
print(f"    degree_centrality: {graph.nodes[sample_node].get('degree_centrality', 0):.4f}")
print(f"    clustering_coefficient: {graph.nodes[sample_node].get('clustering_coefficient', 0):.4f}")

# Test 3: Feature Extractor
print("\n[3/10] Feature Extractor...")
feature_extractor = AMLNetFeatureExtractor()
features = feature_extractor.extract(graph.nodes[sample_node])
print(f"OK - Extracted {len(features)} features")
print(f"  Feature vector shape: {features.shape}")

# Test 4: GNN Encoder
print("\n[4/10] GNN Encoder (GraphSAGE)...")
encoder = StateEncoder(
    node_feature_dim=12,
    gnn_type="sage",
    embedding_dim=64,
    history_dim=16
)
print("OK - GNN encoder created")

# Test 5: Environment
print("\n[5/10] Environment...")
env = AMLDetectionEnv(
    graph=graph,
    max_steps=20,
    max_neighbors=5
)
print("OK - Environment created")
print(f"  Action space: {env.action_space}")
print(f"  Observation space: {env.observation_space.shape}")

# Test 6: Environment Reset
print("\n[6/10] Environment Reset...")
obs, info = env.reset()
print("OK - Environment reset")
print(f"  Observation shape: {obs.shape}")
print(f"  Current node: {env.current_node}")

# Test 7: GNN State Encoding
print("\n[7/10] GNN State Encoding...")
with torch.no_grad():
    encoded_state = encoder.encode_state(
        graph=graph,
        current_node=env.current_node,
        visited_nodes=set([env.current_node]),
        visited_edges=[],
        node_feature_extractor=feature_extractor
    )
print("OK - State encoded")
print(f"  Encoded state shape: {encoded_state.shape}")

# Test 8: DQN Agent
print("\n[8/10] DQN Agent...")
agent = DQNAgent(
    state_dim=80,
    action_dim=env.action_space.n,
    hidden_dims=[128, 128, 64],
    learning_rate=0.001
)
print("OK - DQN agent created")
print("  State dim: 80")
print(f"  Action dim: {env.action_space.n}")
print("  Hidden dims: [128, 128, 64]")

# Test 9: Agent Action Selection
print("\n[9/10] Agent Action Selection...")
state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0)
with torch.no_grad():
    action = agent.select_action(state_tensor, epsilon=0.1)
print(f"OK - Action selected: {action}")

# Test 10: Environment Step
print("\n[10/10] Environment Step...")
obs, reward, terminated, truncated, info = env.step(action)
print("OK - Step executed")
print(f"  Reward: {reward:.3f}")
print(f"  Terminated: {terminated}")
print(f"  Truncated: {truncated}")

# Bonus: XGBoost Baseline
print("\n[BONUS] XGBoost Baseline...")
xgb_baseline = XGBoostBaseline()
print("OK - XGBoost baseline created")

# Bonus: Budgeted Metrics
print("\n[BONUS] Budgeted Metrics...")
metrics = BudgetedMetrics()
dummy_preds = np.random.rand(100)
dummy_labels = np.random.randint(0, 2, 100)
recall_10 = metrics.recall_at_k(dummy_preds, dummy_labels, k=10)
aupr = metrics.aupr(dummy_preds, dummy_labels)
print("OK - Metrics computed")
print(f"  Recall@10: {recall_10:.3f}")
print(f"  AUPR: {aupr:.3f}")

# Bonus: Entity-Disjoint Split
print("\n[BONUS] Entity-Disjoint Split...")
train_graph, val_graph, test_graph = entity_disjoint_split(graph, test_size=0.2, val_size=0.1)
print("OK - Split created")
print(f"  Train: {len(train_graph.nodes())} nodes, {len(train_graph.edges())} edges")
print(f"  Val: {len(val_graph.nodes())} nodes, {len(val_graph.edges())} edges")
print(f"  Test: {len(test_graph.nodes())} nodes, {len(test_graph.edges())} edges")

print("\n" + "=" * 80)
print("ALL COMPONENTS TESTED SUCCESSFULLY!")
print("=" * 80)

print("\nComponent Summary:")
print("  [OK] Data Loading (AMLNet)")
print("  [OK] Network Features (PyG)")
print("  [OK] Feature Extractor")
print("  [OK] GNN Encoder (GraphSAGE)")
print("  [OK] Environment (AMLDetectionEnv)")
print("  [OK] DQN Agent")
print("  [OK] XGBoost Baseline")
print("  [OK] Budgeted Metrics")
print("  [OK] Entity-Disjoint Split")
print("\nAll systems operational!")
