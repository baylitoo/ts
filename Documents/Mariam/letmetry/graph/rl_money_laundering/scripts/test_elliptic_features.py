"""
Test NodeFeatureExtractor with Elliptic dataset.

This script loads Elliptic data and tests what happens when we try to
extract node features using the current NodeFeatureExtractor.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rl_money_laundering.datasets.elliptic import EllipticConfig, EllipticLoader
from rl_money_laundering.features.extractors import NodeFeatureExtractor


def main():
    print("=" * 80)
    print("Testing NodeFeatureExtractor with Elliptic Dataset")
    print("=" * 80)

    # 1. Load Elliptic dataset
    print("\n1. Loading Elliptic dataset...")
    config = EllipticConfig(data_dir="data/elliptic")
    loader = EllipticLoader(config)

    features_df, classes_df, edges_df = loader.load_raw()
    processed_df, target = loader.feature_pipeline()
    graph = loader.build_graph(include_unknown=False)

    print(f"   - Loaded {len(processed_df)} transactions")
    print(f"   - Graph has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"   - Features shape: {processed_df.shape}")

    # 2. Examine a sample node from the graph
    print("\n2. Examining sample node from graph...")
    sample_node = list(graph.nodes())[0]
    node_data = graph.nodes[sample_node]

    print(f"   - Sample node ID: {sample_node}")
    print(f"   - Node attributes: {node_data}")

    # 3. Check what features are available in processed_df for this node
    print("\n3. Checking processed features for this node...")
    node_row = processed_df[processed_df['txId'].astype(str) == sample_node]

    if len(node_row) > 0:
        node_row = node_row.iloc[0]
        print(f"   - Available columns: {node_row.index.tolist()}")
        print("   - Sample feature values:")
        feature_cols = [col for col in node_row.index if col.startswith('feature_')]
        print(f"     First 10 features: {node_row[feature_cols[:10]].values}")
        print(f"     Temporal features: time_tx_density={node_row['time_tx_density']:.4f}, time_illicit_ratio={node_row['time_illicit_ratio']:.4f}")
        print(f"     Local stats: mean={node_row['local_mean']:.4f}, std={node_row['local_std']:.4f}")
        print(f"     Agg stats: mean={node_row['agg_mean']:.4f}, std={node_row['agg_std']:.4f}")

    # 4. Try to use NodeFeatureExtractor (this will likely fail/return wrong dims)
    print("\n4. Testing NodeFeatureExtractor (current implementation)...")
    extractor = NodeFeatureExtractor(
        include_temporal=True,
        include_network=True,
        node_feature_dim=20  # AMLNet default
    )

    print(f"   - Expected output dim: {extractor.node_feature_dim}")

    try:
        features = extractor.extract(
            node_data=node_data,
            node_id=sample_node,
            graph=graph
        )
        print("   - ✓ Extraction succeeded")
        print(f"   - Output shape: {features.shape}")
        print(f"   - Output values: {features}")

        # Check if features are all zeros/defaults
        non_zero = np.count_nonzero(features)
        print(f"   - Non-zero values: {non_zero}/{len(features)}")

    except Exception as e:
        print(f"   - ✗ Extraction failed: {e}")

    # 5. Show what we ACTUALLY need
    print("\n5. What we actually need for Elliptic:")
    print("   - Elliptic has 166 raw features + 6 engineered features")
    print("   - Current NodeFeatureExtractor expects structured attributes:")
    print("     • node_type, total_sent, total_received, balance")
    print("     • num_transactions_sent, num_transactions_received")
    print("     • risk_score, is_suspicious")
    print("     • tx_velocity, business_hour_ratio, periodicity_score")
    print("     • degree_centrality, clustering_coefficient, in/out_degree")
    print(f"   - But Elliptic nodes only have: {list(node_data.keys())}")

    print("\n6. Proposed solutions:")
    print("   Option A: Create EllipticNodeFeatureExtractor that uses feature_1...feature_166")
    print("   Option B: Modify NodeFeatureExtractor to handle both datasets")
    print("   Option C: Store processed features in graph nodes during build_graph()")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
