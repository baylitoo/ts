"""
Pre-compute and cache GNN embeddings for all nodes to accelerate training.

This script loads the graph, runs GNN encoding on all nodes once, and saves
the embeddings to disk. During training, instead of running GNN forward pass
at each step, the agent can simply load pre-computed embeddings.

Usage:
    python scripts/precompute_embeddings.py --config_file configs/amlnet_rmganets_simple.json --output embeddings.npz
    python scripts/precompute_embeddings.py --config amlnet_full --output embeddings.npz --max_edges 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_money_laundering.config import (
    ExperimentConfig,
    get_elliptic_config,
    get_full_amlnet_config,
    get_quick_test_config,
)
from rl_money_laundering.gnn_encoder import GNNStateEncoder


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load experiment configuration."""
    if args.config_file:
        config = ExperimentConfig.load(args.config_file)
    elif args.config == "quick_test":
        config = get_quick_test_config()
    elif args.config == "amlnet_full":
        config = get_full_amlnet_config()
    elif args.config == "elliptic":
        config = get_elliptic_config()
    else:
        raise ValueError(f"Unknown config preset: {args.config}")

    if args.nrows is not None:
        config.nrows = args.nrows
    if args.max_edges is not None:
        # Store for later use in pipeline
        config.max_edges = args.max_edges
    if args.fraud_ratio is not None:
        config.fraud_ratio = args.fraud_ratio

    return config


def precompute_all_embeddings(
    graph: nx.DiGraph,
    state_encoder: GNNStateEncoder,
    node_feature_extractor: callable,
    output_path: Path,
    batch_size: int = 100,
    device: str = "cpu"
) -> None:
    """
    Pre-compute GNN embeddings for all nodes in the graph.

    Args:
        graph: Full transaction graph
        state_encoder: GNN encoder to use
        node_feature_extractor: Function to extract features from node
        output_path: Where to save embeddings (.npz file)
        batch_size: Number of nodes to process at once
        device: Device for computation
    """
    print(f"\n{'='*80}")
    print("Pre-computing GNN Embeddings")
    print(f"{'='*80}")
    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"Total edges: {graph.number_of_edges()}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    state_encoder.eval()
    all_nodes = list(graph.nodes())
    embeddings_dict: Dict[str, np.ndarray] = {}

    # Process nodes in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(all_nodes), batch_size), desc="Computing embeddings"):
            batch_nodes = all_nodes[i:i + batch_size]

            for node in batch_nodes:
                try:
                    # Get embedding for this node
                    # Use empty visited sets since we're just caching node embeddings
                    embedding, history_features, _ = state_encoder.forward_state(
                        graph=graph,
                        current_node=node,
                        visited_nodes=set(),
                        visited_edges=[],
                        node_feature_extractor=node_feature_extractor,
                        return_auxiliary=False
                    )

                    # Store only the GNN embedding part (not history)
                    embedding_np = embedding.detach().cpu().numpy()
                    embeddings_dict[node] = embedding_np

                except Exception as e:
                    print(f"\nWarning: Failed to compute embedding for node {node}: {e}")
                    # Use zero embedding as fallback
                    embeddings_dict[node] = np.zeros(state_encoder.embedding_dim, dtype=np.float32)

    # Save embeddings
    print(f"\nSaving embeddings to {output_path}...")

    # Convert dict to arrays for efficient storage
    node_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[node] for node in node_ids], dtype=np.float32)

    np.savez_compressed(
        output_path,
        node_ids=np.array(node_ids, dtype=object),
        embeddings=embeddings,
        embedding_dim=state_encoder.embedding_dim,
        num_nodes=len(node_ids),
        gnn_type=state_encoder.gnn_type,
    )

    print(f"✓ Saved {len(node_ids)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("\nEmbeddings cache created successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute GNN embeddings for training acceleration")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", choices=["quick_test", "amlnet_full", "elliptic"], help="Preset configuration")
    group.add_argument("--config_file", type=str, help="Path to custom config JSON file")

    parser.add_argument("--output", type=str, required=True, help="Output path for embeddings (.npz)")
    parser.add_argument("--nrows", type=int, help="Limit number of rows loaded from dataset")
    parser.add_argument("--max_edges", type=int, help="Limit number of edges when constructing graph")
    parser.add_argument(
        "--fraud_ratio",
        type=float,
        help="Target fraud ratio when sampling AMLNet subsets (0-1)"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing nodes")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cpu", help="Computation device")

    args = parser.parse_args()

    # Load config
    config = load_config(args)

    # Override device if specified
    if args.device:
        config.agent.device = args.device
        config.gnn.device = args.device

    # Import pipeline after config is ready
    from rl_money_laundering.pipeline import build_pipeline

    print(f"{'='*80}")
    print("Loading Dataset and Building Graph")
    print(f"{'='*80}")
    print(f"Config: {config.name}")
    print(f"Dataset: {config.dataset_type}")
    print(f"Device: {config.gnn.device}")
    print(f"GNN Type: {config.gnn.gnn_type}")
    print(f"{'='*80}\n")

    # Build pipeline to get dataset and state encoder
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = build_pipeline(
        config,
        output_dir=output_path,
        nrows=config.nrows,
        max_edges=getattr(args, 'max_edges', None),
        fraud_ratio=getattr(args, 'fraud_ratio', None),
        device=args.device,
    )

    dataset = artifacts.dataset
    state_encoder = artifacts.state_encoder

    # Node feature extractor
    def node_feature_extractor(node_id: str):
        return dataset.graph.nodes[node_id]

    print("\n✓ Dataset loaded:")
    print(f"  Transactions: {len(dataset.transactions)}")
    print(f"  Graph: {dataset.graph.number_of_nodes()} nodes, {dataset.graph.number_of_edges()} edges")
    print(f"  Fraudulent: {int(dataset.transactions[dataset.target_column].sum())} "
          f"({dataset.transactions[dataset.target_column].mean()*100:.2f}%)")

    # Pre-compute embeddings
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    precompute_all_embeddings(
        graph=dataset.graph,
        state_encoder=state_encoder,
        node_feature_extractor=node_feature_extractor,
        output_path=output_file,
        batch_size=args.batch_size,
        device=args.device
    )

    print(f"\n{'='*80}")
    print("Usage in training:")
    print(f"{'='*80}")
    print("# Load embeddings in your training script:")
    print(f"embeddings_cache = np.load('{args.output}')")
    print("node_ids = embeddings_cache['node_ids']")
    print("embeddings = embeddings_cache['embeddings']")
    print("# Create lookup dict:")
    print("embedding_lookup = {nid: emb for nid, emb in zip(node_ids, embeddings)}")
    print("# Use in training:")
    print("node_embedding = embedding_lookup[current_node]")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
