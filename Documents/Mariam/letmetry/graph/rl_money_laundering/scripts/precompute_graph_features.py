"""
Precompute graph features for AML datasets.

This utility loads the raw dataset, builds the transaction graph once,
computes temporal + network features, and persists the enriched graph to disk.
Downstream training can then load the cached graph directly to avoid repeated
CPU-bound preprocessing.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch

from rl_money_laundering.datasets import AMLNetConfig, AMLNetLoader
from rl_money_laundering.features.network_pyg import add_network_features_pyg
from rl_money_laundering.features.temporal import add_temporal_features_to_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute graph features and cache the processed graph."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the raw AMLNet CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the cached graph (e.g. data/amlnet/cache/graph.gpickle).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for PyG feature extraction (default: auto-detect).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit when loading the CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--save-transactions-csv",
        type=Path,
        default=None,
        help="Optional path to export the (possibly subset) transactions as CSV for reuse.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    device = resolve_device(args.device)
    print("=" * 80)
    print("GRAPH FEATURE PRECOMPUTATION")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Output : {output_path}")
    print(f"Device : {device}")
    if args.nrows:
        print(f"Row cap: {args.nrows:,}")
    print("=" * 80)

    loader = AMLNetLoader(AMLNetConfig(csv_path=dataset_path))

    t0 = time.perf_counter()
    transactions = loader.load_raw(nrows=args.nrows)
    load_time = time.perf_counter() - t0
    print(f"[1/4] Loaded {len(transactions):,} rows in {load_time:.2f}s")

    t1 = time.perf_counter()
    graph = loader.build_graph()
    graph_time = time.perf_counter() - t1
    print(
        f"[2/4] Graph: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges (built in {graph_time:.2f}s)"
    )

    t2 = time.perf_counter()
    graph = add_temporal_features_to_graph(graph, transactions)
    temporal_time = time.perf_counter() - t2
    print(f"[3/4] Temporal features added in {temporal_time:.2f}s")

    t3 = time.perf_counter()
    graph = add_network_features_pyg(graph, device=device)
    network_time = time.perf_counter() - t3
    print(f"[4/4] Network features added in {network_time:.2f}s")

    nx.write_gpickle(graph, output_path)
    print(f"\nCached graph written to: {output_path}")

    if args.save_transactions_csv:
        csv_path = args.save_transactions_csv.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        transactions.to_csv(csv_path, index=False)
        print(f"Transactions CSV saved to: {csv_path}")

    meta = {
        "dataset": str(dataset_path),
        "output": str(output_path),
        "nrows": args.nrows,
        "device": device,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "load_time_sec": round(load_time, 3),
        "graph_time_sec": round(graph_time, 3),
        "temporal_time_sec": round(temporal_time, 3),
        "network_time_sec": round(network_time, 3),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata saved to: {meta_path}")
    print("\nPrecomputation complete.")


if __name__ == "__main__":
    main()
