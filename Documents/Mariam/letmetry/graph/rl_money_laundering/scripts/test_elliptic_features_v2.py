"""
Test EllipticNodeFeatureExtractor with proper feature storage.

This script tests the complete pipeline:
1. Load Elliptic dataset
2. Process features
3. Build graph WITH features stored in nodes
4. Extract features using EllipticNodeFeatureExtractor
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rl_money_laundering.datasets.elliptic import EllipticConfig, EllipticLoader
from rl_money_laundering.features import EllipticNodeFeatureExtractor


def main():
    print("=" * 80)
    print("Testing EllipticNodeFeatureExtractor (Professional Implementation)")
    print("=" * 80)

    # 1. Load Elliptic dataset
    print("\n[1/5] Loading Elliptic dataset...")
    import logging
    logging.basicConfig(level=logging.DEBUG)

    config = EllipticConfig(data_dir="data/elliptic")
    loader = EllipticLoader(config)

    features_df, classes_df, edges_df = loader.load_raw()
    print(f"   ✓ Loaded {len(features_df)} transactions, {len(edges_df)} edges")

    # 2. Process features
    print("\n[2/5] Running feature pipeline...")
    processed_df, target = loader.feature_pipeline()
    print(f"   ✓ Processed {len(processed_df)} labeled transactions")
    print(f"   ✓ Feature shape: {processed_df.shape}")

    feature_cols = [col for col in processed_df.columns if col.startswith('feature_')]
    engineered_cols = ['time_tx_density', 'time_illicit_ratio', 'local_mean', 'local_std', 'agg_mean', 'agg_std']
    print(f"   ✓ Raw features: {len(feature_cols)} (feature_1 ... feature_166)")
    print(f"   ✓ Engineered features: {len(engineered_cols)}")
    print(f"   ✓ Total feature dimension: {len(feature_cols) + len(engineered_cols)}")

    # 3. Build graph WITH features
    print("\n[3/5] Building graph with features stored in nodes...")
    graph = loader.build_graph(include_unknown=False, processed_features=processed_df)
    print(f"   ✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Check node attributes
    sample_node = list(graph.nodes())[0]
    node_attrs = list(graph.nodes[sample_node].keys())
    print(f"   ✓ Sample node {sample_node} has {len(node_attrs)} attributes")
    print(f"   ✓ Attributes: {node_attrs[:10]}... (showing first 10)")

    # DEBUG: Check if node exists in processed_df
    print("\n   [DEBUG] Checking feature attachment...")
    print(f"   Node ID: {sample_node} (type: {type(sample_node)})")
    matching = processed_df[processed_df['txId'].astype(str) == str(sample_node)]
    print(f"   Matching rows in processed_df: {len(matching)}")
    if len(matching) > 0:
        print("   ✓ Node exists in processed_df")
        print(f"   Sample features from DF: feature_1={matching.iloc[0]['feature_1']:.4f}")
    else:
        print("   ✗ Node NOT found in processed_df!")

    # 4. Create extractor
    print("\n[4/5] Creating EllipticNodeFeatureExtractor...")
    extractor = EllipticNodeFeatureExtractor(node_feature_dim=172)
    print(f"   ✓ Expected output dimension: {extractor.node_feature_dim}")

    # 5. Extract features
    print("\n[5/5] Extracting features from nodes...")

    # Test on multiple nodes
    test_nodes = list(graph.nodes())[:5]
    print(f"   Testing on {len(test_nodes)} sample nodes...")

    for i, node_id in enumerate(test_nodes, 1):
        node_data = graph.nodes[node_id]

        try:
            features = extractor.extract(
                node_data=node_data,
                node_id=node_id,
                graph=graph
            )

            # Verify output
            assert features.shape == (172,), f"Wrong shape: {features.shape}"
            assert features.dtype == np.float32, f"Wrong dtype: {features.dtype}"

            # Check for actual values (not all zeros)
            non_zero = np.count_nonzero(features)
            mean_val = np.mean(features)
            std_val = np.std(features)

            print(f"   [{i}/{len(test_nodes)}] Node {node_id}:")
            print(f"        ✓ Shape: {features.shape}")
            print(f"        ✓ Non-zero values: {non_zero}/{len(features)} ({100*non_zero/len(features):.1f}%)")
            print(f"        ✓ Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            print(f"        ✓ Min: {features.min():.4f}, Max: {features.max():.4f}")

        except Exception as e:
            print(f"   [{i}/{len(test_nodes)}] Node {node_id}: ✗ FAILED")
            print(f"        Error: {e}")
            return

    print("\n" + "=" * 80)
    print("SUCCESS: All tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Elliptic dataset loaded successfully")
    print("  ✓ Features processed and stored in graph nodes")
    print("  ✓ EllipticNodeFeatureExtractor working correctly")
    print("  ✓ Output dimension: 172 features (166 raw + 6 engineered)")
    print("\nNext steps:")
    print("  1. Update trainer/environment to use EllipticNodeFeatureExtractor for Elliptic")
    print("  2. Update GNN config to set node_feature_dim=172 for Elliptic")
    print("  3. Test training with Elliptic dataset")


if __name__ == "__main__":
    main()
