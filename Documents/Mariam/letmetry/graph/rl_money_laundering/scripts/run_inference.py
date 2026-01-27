"""
Inference script for trained RL-AML detection models.

Loads a trained model checkpoint and runs inference on a dataset (AMLNet or Elliptic),
producing detailed detection reports with confusion matrices, precision, recall, F1 scores,
and per-node predictions.

Usage:
    # Run on AMLNet dataset
    python scripts/run_inference.py --checkpoint outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt \
                                     --config configs/amlnet_rmganets_fixed.json \
                                     --dataset amlnet \
                                     --output results/inference_amlnet.json

    # Run on Elliptic dataset
    python scripts/run_inference.py --checkpoint outputs/best_model.pt \
                                     --config configs/elliptic_config.json \
                                     --dataset elliptic \
                                     --elliptic-dir data/elliptic \
                                     --output results/inference_elliptic.json

    # Run with specific episode length
    python scripts/run_inference.py --checkpoint outputs/checkpoint_ep1500.pt \
                                     --config configs/amlnet_rmganets_fixed.json \
                                     --max-steps 200 \
                                     --num-episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_money_laundering.agent import QRDQNAgent
from rl_money_laundering.config import ExperimentConfig
from rl_money_laundering.datasets.amlnet import AMLNetLoader
from rl_money_laundering.datasets.elliptic import EllipticConfig, EllipticLoader
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.features.extractors import NodeFeatureExtractor
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.utils.checkpoint import CheckpointManager


def load_amlnet_dataset(config: ExperimentConfig) -> Tuple[nx.DiGraph, List[nx.DiGraph]]:
    """Load AMLNet dataset and fraud subgraphs."""
    print(f"Loading AMLNet dataset from {config.dataset_path}...")

    from rl_money_laundering.datasets.amlnet import AMLNetConfig

    aml_config = AMLNetConfig(csv_path=str(config.dataset_path))
    loader = AMLNetLoader(config=aml_config)

    df = loader.load_raw()
    graph = loader.build_graph(df)
    fraud_subgraphs = loader.extract_fraud_subgraphs(graph, min_size=5)

    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Found {len(fraud_subgraphs)} fraud-containing subgraphs")

    return graph, fraud_subgraphs


def load_elliptic_dataset(data_dir: str) -> Tuple[nx.DiGraph, List[nx.DiGraph]]:
    """
    Load Elliptic Bitcoin dataset with ALL 166 features.

    Features breakdown:
    - Feature 1: Time step (1-49, 2-week intervals)
    - Features 2-94: Local transaction features (93 features)
      * inputs/outputs, fees, volumes, averages
    - Features 95-166: Aggregated neighbor features (72 features)
      * max, min, std, correlation of 1-hop neighbors

    Our loader already extracts:
    - Time step encoding
    - Local feature statistics (mean, std)
    - Aggregate feature statistics (mean, std)
    - Temporal context (tx density, illicit ratio per time step)
    """
    print(f"Loading Elliptic dataset from {data_dir}...")
    print("Dataset info: 203k nodes, 234k edges, 166 features per node")
    print("  - 94 local features (time, inputs/outputs, fees, volumes)")
    print("  - 72 aggregated features (neighbor statistics)")
    print("  - 49 time steps (2-week intervals)")
    print("  - 2% fraud rate (4,545 illicit nodes)")

    elliptic_config = EllipticConfig(
        data_dir=data_dir,
        include_unlabeled=False,  # Only use labeled data (4,545 illicit + 42,019 licit)
    )

    loader = EllipticLoader(elliptic_config)
    features_df, classes_df, edges_df = loader.load_raw()

    # Apply feature engineering (adds temporal context, standardizes all 166 features)
    processed_features, targets = loader.feature_pipeline(scale_features=True)

    print("Feature pipeline complete:")
    print(f"  - Original features: {features_df.shape[1] - 2} (excluding txId, time)")
    print(f"  - Processed features: {processed_features.shape[1] - 2} (with temporal context)")

    # Build graph with rich node features
    graph = loader.build_graph(include_unknown=False)

    # Attach processed features to graph nodes
    feature_cols = [col for col in processed_features.columns if col not in ['txId', 'time_step', 'is_labeled']]
    for idx, row in processed_features.iterrows():
        node_id = str(row['txId'])
        if node_id in graph:
            # Store all features as node attributes
            for col in feature_cols:
                graph.nodes[node_id][col] = row[col]

    # Extract fraud-containing subgraphs using temporal locality
    fraud_nodes = [n for n, d in graph.nodes(data=True) if d.get("label", 0) == 1]
    fraud_subgraphs = []

    # Group by time step for better subgraph coherence
    time_step_groups = {}
    for node in fraud_nodes:
        time_step = graph.nodes[node].get('time', 0)
        if time_step not in time_step_groups:
            time_step_groups[time_step] = []
        time_step_groups[time_step].append(node)

    visited = set()
    for time_step, fraud_nodes_in_step in sorted(time_step_groups.items()):
        for fraud_node in fraud_nodes_in_step:
            if fraud_node in visited:
                continue

            # Get connected component containing this fraud node
            component = nx.node_connected_component(graph.to_undirected(), fraud_node)
            visited.update(component)

            subgraph = graph.subgraph(component).copy()
            if subgraph.number_of_nodes() >= 5:
                fraud_subgraphs.append(subgraph)

    print(f"\nLoaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Found {len(fraud_subgraphs)} fraud-containing subgraphs")
    print(f"Total fraud nodes: {len(fraud_nodes)}")
    print(f"Time steps represented: {len(time_step_groups)}")

    return graph, fraud_subgraphs


def create_agent_and_encoder(
    config: ExperimentConfig,
    graph: nx.DiGraph,
    device: str
) -> Tuple[QRDQNAgent, StateEncoder]:
    """Create agent and state encoder from config."""
    print("Initializing state encoder and agent...")

    # Create feature extractor
    feature_extractor = NodeFeatureExtractor(
        include_temporal=True,
        include_network=True,
        node_feature_dim=20,
    )

    # Detect node feature dimension from graph
    sample_node = list(graph.nodes())[0]
    sample_features = feature_extractor.extract_node_features(sample_node, graph)
    node_feature_dim = len(sample_features)

    print(f"Detected node feature dimension: {node_feature_dim}")

    # Create state encoder
    state_encoder = StateEncoder(
        in_channels=node_feature_dim,
        hidden_channels=config.gnn.hidden_channels,
        num_layers=config.gnn.num_layers,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
        multi_branch_enabled=config.multi_branch.enabled,
    ).to(device)

    # Create agent
    agent = QRDQNAgent(
        state_dim=config.gnn.hidden_channels,
        action_dim=4,  # MOVE_FORWARD, MOVE_BACKWARD, TELEPORT, FLAG
        config=config.agent,
        device=device,
    )

    return agent, state_encoder


def run_episode_inference(
    env: AMLDetectionEnv,
    agent: QRDQNAgent,
    state_encoder: StateEncoder,
    max_steps: int = 200,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run a single episode and collect detailed statistics.

    Returns:
        Dictionary with episode results including:
        - visited_nodes: List of visited node IDs
        - actions_taken: List of action names
        - rewards: List of rewards received
        - fraud_detected: Whether fraud was flagged
        - flag_node: Node ID where FLAG was used (if any)
        - true_fraud_nodes: List of actual fraud nodes in graph
        - confusion: TP/FP/TN/FN counts
    """
    state = env.reset()
    done = False
    step = 0

    visited_nodes = [env.current_node]
    actions_taken = []
    rewards = []

    # Track fraud nodes in graph
    true_fraud_nodes = [
        n for n in env.graph.nodes()
        if env.graph.nodes[n].get("is_fraud", False)
    ]

    while not done and step < max_steps:
        # Encode state
        with torch.no_grad():
            state_tensor = state_encoder(
                env.graph,
                env.current_node,
                env.visited
            ).unsqueeze(0).to(device)

            # Get action from agent (greedy, no exploration)
            action = agent.select_action(state_tensor, epsilon=0.0)

        next_state, reward, done, info = env.step(action)

        # Record step
        action_names = ["MOVE_FORWARD", "MOVE_BACKWARD", "TELEPORT", "FLAG"]
        actions_taken.append(action_names[action])
        rewards.append(reward)
        visited_nodes.append(env.current_node)

        state = next_state
        step += 1

    # Compute confusion matrix
    flagged = info.get("flagged", False)
    flag_node = info.get("flag_node", None)

    fraud_present = len(true_fraud_nodes) > 0

    if fraud_present and flagged:
        tp = 1
        fp = 0
        fn = 0
        tn = 0
    elif fraud_present and not flagged:
        tp = 0
        fp = 0
        fn = 1
        tn = 0
    elif not fraud_present and flagged:
        tp = 0
        fp = 1
        fn = 0
        tn = 0
    else:  # not fraud_present and not flagged
        tp = 0
        fp = 0
        fn = 0
        tn = 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "visited_nodes": visited_nodes,
        "actions_taken": actions_taken,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "steps": step,
        "fraud_detected": flagged,
        "flag_node": flag_node,
        "true_fraud_nodes": true_fraud_nodes,
        "confusion": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    }


def run_inference(
    checkpoint_path: str,
    config_path: str,
    dataset_type: str,
    output_path: str,
    elliptic_dir: str | None = None,
    num_episodes: int = 100,
    max_steps: int = 200,
) -> None:
    """
    Run inference on a dataset using a trained checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        config_path: Path to experiment config (.json file)
        dataset_type: Either "amlnet" or "elliptic"
        output_path: Where to save inference results
        elliptic_dir: Directory containing Elliptic dataset (if dataset_type="elliptic")
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    print(f"Loading config from {config_path}...")
    config = ExperimentConfig.load(config_path)

    # Load dataset
    if dataset_type == "amlnet":
        graph, fraud_subgraphs = load_amlnet_dataset(config)
    elif dataset_type == "elliptic":
        if elliptic_dir is None:
            raise ValueError("Must provide --elliptic-dir when using Elliptic dataset")
        graph, fraud_subgraphs = load_elliptic_dataset(elliptic_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Create agent and encoder
    agent, state_encoder = create_agent_and_encoder(config, graph, device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_manager = CheckpointManager(Path(checkpoint_path).parent)

    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    agent.q_network.load_state_dict(checkpoint_data["q_network_state_dict"])
    agent.epsilon = 0.0  # Greedy inference

    episode_num = checkpoint_data.get("episode", "unknown")
    print(f"Loaded checkpoint from episode {episode_num}")

    # Set agent to eval mode
    agent.q_network.eval()
    state_encoder.eval()

    # Create environment
    env = AMLDetectionEnv(
        graph=graph,
        max_steps=max_steps,
        intrinsic_reward_weight=config.agent.intrinsic_reward_weight,
    )

    # Run inference episodes
    print(f"\nRunning {num_episodes} inference episodes...")
    results = []

    # Split episodes between fraud-seeded and random starts
    fraud_episodes = num_episodes // 2
    random_episodes = num_episodes - fraud_episodes

    # Fraud-seeded episodes
    print(f"\nRunning {fraud_episodes} fraud-seeded episodes...")
    for i in range(fraud_episodes):
        if i < len(fraud_subgraphs):
            # Use specific fraud subgraph
            env.graph = fraud_subgraphs[i]
        else:
            # Cycle through fraud subgraphs
            env.graph = fraud_subgraphs[i % len(fraud_subgraphs)]

        episode_result = run_episode_inference(env, agent, state_encoder, max_steps, device)
        episode_result["episode_id"] = i
        episode_result["episode_type"] = "fraud_seeded"
        results.append(episode_result)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{fraud_episodes} fraud-seeded episodes")

    # Random-start episodes
    print(f"\nRunning {random_episodes} random-start episodes...")
    env.graph = graph  # Use full graph for random starts

    for i in range(random_episodes):
        episode_result = run_episode_inference(env, agent, state_encoder, max_steps, device)
        episode_result["episode_id"] = fraud_episodes + i
        episode_result["episode_type"] = "random_start"
        results.append(episode_result)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{random_episodes} random-start episodes")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)

    fraud_seeded_results = [r for r in results if r["episode_type"] == "fraud_seeded"]
    random_start_results = [r for r in results if r["episode_type"] == "random_start"]

    for episode_type, episode_results in [
        ("Fraud-Seeded", fraud_seeded_results),
        ("Random-Start", random_start_results)
    ]:
        print(f"\n{episode_type} Episodes ({len(episode_results)} episodes)")
        print("-" * 80)

        # Aggregate confusion matrix
        total_tp = sum(r["confusion"]["TP"] for r in episode_results)
        total_fp = sum(r["confusion"]["FP"] for r in episode_results)
        total_tn = sum(r["confusion"]["TN"] for r in episode_results)
        total_fn = sum(r["confusion"]["FN"] for r in episode_results)

        # Calculate aggregate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

        print("Confusion Matrix:")
        print(f"  TP: {total_tp}  FP: {total_fp}")
        print(f"  FN: {total_fn}  TN: {total_tn}")
        print("\nMetrics:")
        print(f"  Precision: {precision * 100:.1f}%")
        print(f"  Recall:    {recall * 100:.1f}%")
        print(f"  F1 Score:  {f1 * 100:.1f}%")
        print(f"  Specificity: {specificity * 100:.1f}%")

        # Episode statistics
        avg_reward = np.mean([r["total_reward"] for r in episode_results])
        avg_steps = np.mean([r["steps"] for r in episode_results])
        flag_rate = sum(r["fraud_detected"] for r in episode_results) / len(episode_results)

        print("\nEpisode Statistics:")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Steps:  {avg_steps:.1f}")
        print(f"  Flag Rate:  {flag_rate * 100:.1f}%")

    # Save detailed results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "dataset": dataset_type,
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "summary": {
            "fraud_seeded": {
                "total_tp": sum(r["confusion"]["TP"] for r in fraud_seeded_results),
                "total_fp": sum(r["confusion"]["FP"] for r in fraud_seeded_results),
                "total_tn": sum(r["confusion"]["TN"] for r in fraud_seeded_results),
                "total_fn": sum(r["confusion"]["FN"] for r in fraud_seeded_results),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "random_start": {
                "total_tp": sum(r["confusion"]["TP"] for r in random_start_results),
                "total_fp": sum(r["confusion"]["FP"] for r in random_start_results),
                "total_tn": sum(r["confusion"]["TN"] for r in random_start_results),
                "total_fn": sum(r["confusion"]["FN"] for r in random_start_results),
                "precision": sum(r["metrics"]["precision"] for r in random_start_results) / len(random_start_results),
                "recall": sum(r["metrics"]["recall"] for r in random_start_results) / len(random_start_results),
                "f1": sum(r["metrics"]["f1"] for r in random_start_results) / len(random_start_results),
            }
        },
        "episodes": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Detailed results saved to {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained RL-AML detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config (.json file)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amlnet", "elliptic"],
        required=True,
        help="Dataset to run inference on"
    )

    parser.add_argument(
        "--elliptic-dir",
        type=str,
        default=None,
        help="Directory containing Elliptic dataset (required if --dataset=elliptic)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/inference_results.json",
        help="Where to save inference results (default: results/inference_results.json)"
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        dataset_type=args.dataset,
        output_path=args.output,
        elliptic_dir=args.elliptic_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
