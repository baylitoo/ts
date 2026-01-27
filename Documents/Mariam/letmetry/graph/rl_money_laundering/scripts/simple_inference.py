"""
Simple inference script - loads existing trained model and runs on dataset.

Usage:
    python scripts/simple_inference.py outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt
"""

import numpy as np
import random
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_money_laundering.pipeline import _build_amlnet_dataset, _build_elliptic_dataset, _extract_fraud_subgraphs
from rl_money_laundering.config import ExperimentConfig
from rl_money_laundering.agent import QRDQNAgent
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.features.extractors import NodeFeatureExtractor


def run_simple_inference(checkpoint_path: str, num_episodes: int = 20):
    """Run inference using existing checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    output_dir = checkpoint_path.parent.parent
    config_path = output_dir / "config.json"

    print(f"Loading config from: {config_path}")
    config = ExperimentConfig.load(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load graph
    print(f"\nLoading dataset: {config.dataset_path}")
    if config.dataset_type == "amlnet":
        dataset_artifacts = _build_amlnet_dataset(
            config,
            nrows=config.nrows,
            max_edges=None,
            fraud_ratio=None
        )
    elif config.dataset_type == "elliptic":
        dataset_artifacts = _build_elliptic_dataset(config, nrows=config.nrows)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")

    graph = dataset_artifacts.graph
    fraud_subgraphs = _extract_fraud_subgraphs(graph, k_hop=1)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Fraud subgraphs: {len(fraud_subgraphs)}")

    # Use config parameters (same as training)
    print(f"Node feature dim: {config.gnn.node_feature_dim}")

    # Create node feature extractor (same as trainer.py lines 70-74)
    node_feature_extractor = NodeFeatureExtractor(
        include_temporal=True,
        include_network=True,
        node_feature_dim=config.gnn.node_feature_dim,
    )

    # Create encoder (exactly as in pipeline.py lines 428-437)
    state_encoder = StateEncoder(
        node_feature_dim=config.gnn.node_feature_dim,
        gnn_type=config.gnn.gnn_type,
        embedding_dim=config.gnn.embedding_dim,
        history_dim=config.gnn.history_dim,
        device=device,
        multi_branch=getattr(config.gnn, "multi_branch", False),
        use_dqn_enhancement=getattr(config.gnn, "use_dqn_enhancement", False),
        num_classes=getattr(config.gnn, "num_classes", 2),
    )

    # Create agent (use state_dim from encoder, just like pipeline.py lines 439-443)
    state_dim = state_encoder.get_state_dim()

    # Build agent exactly as in pipeline.py lines 190-211
    agent = QRDQNAgent(
        state_dim=state_dim,
        action_dim=config.agent.action_dim,
        num_quantiles=config.agent.num_quantiles,
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        n_step=config.agent.n_step,
        epsilon_start=config.agent.epsilon_start,
        epsilon_end=config.agent.epsilon_end,
        epsilon_decay=config.agent.epsilon_decay,
        buffer_capacity=config.agent.buffer_capacity,
        hidden_dims=config.agent.hidden_dims,
        risk_measure=config.agent.risk_measure,
        device=device,
        prioritized_alpha=config.agent.prioritized_alpha,
        prioritized_beta=config.agent.prioritized_beta,
        prioritized_beta_annealing=config.agent.prioritized_beta_annealing,
        positive_fraction=config.agent.positive_fraction,
        use_double_dqn=config.agent.use_double_dqn,
        use_guided_exploration=config.agent.use_guided_exploration,
        exploration_temperature=config.agent.exploration_temperature,
    )

    # Load checkpoint (using agent.load() method from agent.py lines 1871-1878)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    agent.load(str(checkpoint_path))
    agent.epsilon = 0.0  # Force greedy inference

    # Get episode number from filename
    episode_num = checkpoint_path.stem.replace("checkpoint_ep", "")
    print(f"Checkpoint episode: {episode_num}")

    # Set to eval mode
    agent.q_network.eval()
    state_encoder.eval()

    # Create environment (pipeline.py lines 419-424)
    env = AMLDetectionEnv(
        graph=graph,
        max_steps=200,
        intrinsic_reward_weight=config.environment.intrinsic_reward_weight,
    )

    print(f"\n{'='*80}")
    print(f"Running {num_episodes} inference episodes...")
    print(f"{'='*80}\n")

    # Run episodes
    results = {"fraud_seeded": [], "random": []}

    # Fraud-seeded episodes (first half)
    # Use start_node from fraud subgraphs, env.graph stays as main graph (trainer.py lines 208-222)
    fraud_episodes = num_episodes // 2
    for i in range(fraud_episodes):
        # Pick a random node from a fraud subgraph as start_node
        subgraph = fraud_subgraphs[i % len(fraud_subgraphs)]
        nodes = [str(node) for node in subgraph.nodes()]
        start_node = random.choice(nodes) if nodes else None

        # Reset with start_node option (trainer.py lines 233-234)
        reset_options = {"start_node": start_node} if start_node else None
        state, info = env.reset(options=reset_options)
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 200:
            with torch.no_grad():
                # Use forward_state() exactly like trainer.py lines 260-266
                embedding_tensor, history_features, _ = state_encoder.forward_state(
                    graph=env.graph,
                    current_node=env.current_node,
                    visited_nodes=env.visited_nodes,
                    visited_edges=list(env.visited_edges),
                    node_feature_extractor=node_feature_extractor,
                    return_auxiliary=False,
                )
                # Combine embedding and history into state vector (trainer.py line 269)
                state_vector = np.concatenate([
                    embedding_tensor.detach().cpu().numpy().reshape(-1),
                    history_features
                ])
                valid_actions = list(range(len(env.current_neighbors))) + [env.FLAG_ACTION]
                action = agent.select_action(state_vector, valid_actions=valid_actions, epsilon=0.0)

            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        flag_used = bool(info.get("flag_used", False))
        flag_correct = bool(info.get("flag_correct", False))
        episode_contains_fraud = bool(info.get("episode_contains_fraud", False))

        tp = 1 if flag_used and flag_correct else 0
        fp = 1 if flag_used and not flag_correct else 0
        fn = 1 if (not flag_used) and episode_contains_fraud else 0
        tn = 1 if (not flag_used) and (not episode_contains_fraud) else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results["fraud_seeded"].append({
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "reward": total_reward, "steps": steps
        })

        if (i + 1) % 5 == 0:
            print(f"Fraud-seeded {i+1}/{fraud_episodes}: F1={f1*100:.1f}% (P={precision*100:.1f}%, R={recall*100:.1f}%)")

    # Random episodes (second half)
    random_episodes = num_episodes - fraud_episodes

    for i in range(random_episodes):
        # Reset without specifying start_node (random start, trainer.py line 234)
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 200:
            with torch.no_grad():
                # Use forward_state() exactly like trainer.py lines 260-266
                embedding_tensor, history_features, _ = state_encoder.forward_state(
                    graph=env.graph,
                    current_node=env.current_node,
                    visited_nodes=env.visited_nodes,
                    visited_edges=list(env.visited_edges),
                    node_feature_extractor=node_feature_extractor,
                    return_auxiliary=False,
                )
                # Combine embedding and history into state vector (trainer.py line 269)
                state_vector = np.concatenate([
                    embedding_tensor.detach().cpu().numpy().reshape(-1),
                    history_features
                ])
                valid_actions = list(range(len(env.current_neighbors))) + [env.FLAG_ACTION]
                action = agent.select_action(state_vector, valid_actions=valid_actions, epsilon=0.0)

            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        flag_used = bool(info.get("flag_used", False))
        flag_correct = bool(info.get("flag_correct", False))
        episode_contains_fraud = bool(info.get("episode_contains_fraud", False))

        tp = 1 if flag_used and flag_correct else 0
        fp = 1 if flag_used and not flag_correct else 0
        fn = 1 if (not flag_used) and episode_contains_fraud else 0
        tn = 1 if (not flag_used) and (not episode_contains_fraud) else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results["random"].append({
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "reward": total_reward, "steps": steps
        })

        if (i + 1) % 5 == 0:
            print(f"Random {i+1}/{random_episodes}: F1={f1*100:.1f}% (P={precision*100:.1f}%, R={recall*100:.1f}%)")

    # Print summary
    print(f"\n{'='*80}")
    print("INFERENCE SUMMARY")
    print(f"{'='*80}")

    for name, episodes in [("Fraud-Seeded", results["fraud_seeded"]), ("Random-Start", results["random"])]:
        total_tp = sum(e["tp"] for e in episodes)
        total_fp = sum(e["fp"] for e in episodes)
        total_fn = sum(e["fn"] for e in episodes)
        total_tn = sum(e["tn"] for e in episodes)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_reward = sum(e["reward"] for e in episodes) / len(episodes)
        avg_steps = sum(e["steps"] for e in episodes) / len(episodes)

        print(f"\n{name} ({len(episodes)} episodes):")
        print(f"  TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall:    {recall*100:.1f}%")
        print(f"  F1 Score:  {f1*100:.1f}%")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Steps:  {avg_steps:.1f}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/simple_inference.py <checkpoint_path> [num_episodes]")
        print("Example: python scripts/simple_inference.py outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt 20")
        sys.exit(1)

    checkpoint = sys.argv[1]
    num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    run_simple_inference(checkpoint, num_eps)
