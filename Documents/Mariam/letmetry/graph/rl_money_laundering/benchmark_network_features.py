"""
Benchmark: PyG vs NetworkX Network Feature Extraction

Compares performance of:
1. NetworkX implementation (O(n^3) for betweenness)
2. PyG implementation (O(m) on GPU)
"""

import time

import numpy as np
import torch

from rl_money_laundering.datasets import AMLNetConfig, AMLNetLoader
from rl_money_laundering.features.network import add_network_features_to_graph
from rl_money_laundering.features.network_pyg import add_network_features_pyg

print("=" * 80)
print("NETWORK FEATURE EXTRACTION BENCHMARK")
print("=" * 80)

# Load data
print("\n[1/4] Loading AMLNet data...")

config = AMLNetConfig(csv_path="../AMLNet_August 2025.csv")
loader = AMLNetLoader(config)
loader.load_raw(nrows=50000)  # 50K transactions
graph = loader.build_graph()
print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

# Benchmark NetworkX
print("\n[2/4] NetworkX implementation (network.py)...")

graph_nx = graph.copy()  # Copy to avoid modifying original
start_time = time.time()
graph_nx = add_network_features_to_graph(graph_nx, compute_expensive=False)
networkx_time = time.time() - start_time
print(f"Time: {networkx_time:.2f} seconds")

# Get NetworkX results for comparison
sample_node = list(graph_nx.nodes())[100]
nx_features = {
    'degree_centrality': graph_nx.nodes[sample_node]['degree_centrality'],
    'clustering_coefficient': graph_nx.nodes[sample_node]['clustering_coefficient'],
    'in_degree': graph_nx.nodes[sample_node]['in_degree'],
    'out_degree': graph_nx.nodes[sample_node]['out_degree'],
}
print(f"Sample node '{sample_node}' features:")
for k, v in nx_features.items():
    print(f"  {k}: {v:.6f}")

# Benchmark PyG
print("\n[3/4] PyG implementation (network_pyg.py)...")

# Check if CUDA available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

graph_pyg = graph.copy()
start_time = time.time()
graph_pyg = add_network_features_pyg(graph_pyg, device=device)
pyg_time = time.time() - start_time
print(f"Time: {pyg_time:.2f} seconds")

# Get PyG results for comparison
pyg_features = {
    'degree_centrality': graph_pyg.nodes[sample_node]['degree_centrality'],
    'clustering_coefficient': graph_pyg.nodes[sample_node]['clustering_coefficient'],
    'in_degree': graph_pyg.nodes[sample_node]['in_degree'],
    'out_degree': graph_pyg.nodes[sample_node]['out_degree'],
}
print(f"Sample node '{sample_node}' features:")
for k, v in pyg_features.items():
    print(f"  {k}: {v:.6f}")

# Compare results
print("\n[4/4] Comparing results...")

# Check if results match (within numerical tolerance)
all_nodes = list(graph.nodes())
differences = []

for node in all_nodes[:100]:  # Check first 100 nodes
    for feature in ['degree_centrality', 'clustering_coefficient', 'in_degree', 'out_degree']:
        nx_val = graph_nx.nodes[node][feature]
        pyg_val = graph_pyg.nodes[node][feature]
        diff = abs(nx_val - pyg_val)
        differences.append(diff)

max_diff = max(differences)
mean_diff = np.mean(differences)

print("Numerical differences (first 100 nodes):")
print(f"  Max difference: {max_diff:.10f}")
print(f"  Mean difference: {mean_diff:.10f}")

if max_diff < 1e-5:
    print("  -> Results match! (within tolerance)")
else:
    print("  -> Results differ! Check implementation")

# Performance summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"NetworkX: {networkx_time:.2f}s")
print(f"PyG ({device}): {pyg_time:.2f}s")
speedup = networkx_time / pyg_time
print(f"Speedup: {speedup:.1f}x")
print("=" * 80)

if speedup > 1:
    print(f"\n-> PyG is {speedup:.1f}x FASTER!")
    print(f"   For 1M transactions (~110K nodes): {networkx_time * 10 / 60:.1f} min (NetworkX) vs {pyg_time * 10 / 60:.1f} min (PyG)")
else:
    print("\n-> NetworkX is faster for this graph size")
