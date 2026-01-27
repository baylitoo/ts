"""
RLlib Training Script for AML Detection

Demonstrates Phase 1 integration of RLlib with GNN-based RL agent.
Follows Ray 2.40+ new API stack patterns.
"""

import argparse
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rl_money_laundering.rllib_integration import (
    register_aml_env,
    GNNDQNModule,
    FraudAwareReplayBuffer,
    create_fraud_callbacks,
)
from rl_money_laundering.features.network_pyg import NetworkFeatureExtractor
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.data.elliptic import load_elliptic_data


def create_env_config(args):
    """
    Create environment configuration dict for RLlib.

    Args:
        args: Command line arguments

    Returns:
        Dict with environment configuration
    """
    print("Loading Elliptic dataset...")
    graph, illicit_nodes = load_elliptic_data(
        node_features_path=args.node_features,
        edges_path=args.edges,
        classes_path=args.classes,
        include_unknown=args.include_unknown
    )

    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Illicit nodes: {len(illicit_nodes)}")

    # Initialize feature extractor
    node_feature_extractor = NetworkFeatureExtractor(
        graph=graph,
        node_feature_dim=args.node_feature_dim,
        categorical_features=[]
    )

    # Initialize GNN encoder
    gnn_encoder = StateEncoder(
        node_feature_dim=args.node_feature_dim,
        gnn_type=args.gnn_type,
        embedding_dim=args.embedding_dim,
        history_dim=args.history_dim,
        device="cpu",  # Workers use CPU, learner uses GPU
        multi_branch=args.multi_branch,
        use_dqn_enhancement=False
    )

    # Select start node (random illicit node)
    import random
    start_node = random.choice(list(illicit_nodes))

    return {
        "graph": graph,
        "start_node": start_node,
        "node_feature_extractor": node_feature_extractor,
        "gnn_encoder": gnn_encoder,
        "max_steps": args.max_steps,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
    }


def train(args):
    """
    Train DQN agent with RLlib.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True
    )

    try:
        # Register environment
        register_aml_env("aml_detection")

        # Create environment config
        env_config = create_env_config(args)

        # Configure DQN algorithm with new API stack
        config = (
            DQNConfig()
            .environment(
                env="aml_detection",
                env_config=env_config,
            )
            .framework("torch")
            .rl_module(
                # Use new RLModule API (not deprecated model config)
                rl_module_spec=RLModuleSpec(
                    module_class=GNNDQNModule,
                    model_config_dict={
                        "node_feature_dim": args.node_feature_dim,
                        "gnn_type": args.gnn_type,
                        "embedding_dim": args.embedding_dim,
                        "history_dim": args.history_dim,
                        "multi_branch": args.multi_branch,
                        "use_dqn_enhancement": False,
                        "hidden_dims": [128, 128, 64],
                    }
                )
            )
            .training(
                # DQN-specific training config
                lr=args.lr,
                gamma=args.gamma,
                train_batch_size_per_learner=args.batch_size,  # New API name
                replay_buffer_config={
                    # Phase 2: Use FraudAwareReplayBuffer for better fraud sampling
                    "type": FraudAwareReplayBuffer if args.use_fraud_buffer else "PrioritizedEpisodeReplayBuffer",
                    "capacity": args.buffer_capacity,
                    "alpha": args.per_alpha,
                    "beta": args.per_beta,
                    "fraud_boost_factor": args.fraud_boost_factor,  # Custom parameter
                },
                double_q=True,  # Double DQN
                dueling=True,   # Dueling architecture
                n_step=args.n_step,
                target_network_update_freq=args.target_update_freq,
            )
            .callbacks(
                # Phase 2: Add fraud-specific metrics tracking
                create_fraud_callbacks(detailed=args.detailed_callbacks)
            )
            .env_runners(
                # Parallel environment rollout workers
                num_env_runners=args.num_workers,
                num_envs_per_env_runner=1,  # One env per worker (graph env not vectorizable)
                rollout_fragment_length=args.rollout_fragment_length,
            )
            .learners(
                # New API: separate learners from env_runners
                num_learners=1,  # Start with single learner
                num_gpus_per_learner=1 if args.num_gpus > 0 else 0,
            )
            .resources(
                # Resource allocation for workers
                num_cpus_per_env_runner=args.cpus_per_worker,
            )
            .reporting(
                # Logging and checkpointing
                min_sample_timesteps_per_iteration=args.min_sample_timesteps,
                min_time_s_per_iteration=0,
            )
            .debugging(
                # Set seed for reproducibility
                seed=args.seed,
            )
        )

        # Build algorithm
        print("\nBuilding DQN algorithm...")
        algo = config.build()

        print(f"\nStarting training for {args.num_iterations} iterations...")
        print(f"Workers: {args.num_workers}, Batch size: {args.batch_size}")
        print(f"GNN type: {args.gnn_type}, Multi-branch: {args.multi_branch}")
        print(f"Phase 2 optimizations:")
        print(f"  - Batch graph processing: ENABLED (PyG Batch)")
        print(f"  - Fraud-aware replay: {'ENABLED' if args.use_fraud_buffer else 'DISABLED'}")
        print(f"  - Fraud callbacks: {'DETAILED' if args.detailed_callbacks else 'BASIC'}")
        print()

        # Training loop
        best_reward = float('-inf')
        for iteration in range(args.num_iterations):
            result = algo.train()

            # Extract key metrics (new API uses nested structure)
            episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
            episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)
            num_env_steps = result.get("num_env_steps_sampled_lifetime", 0)

            # Loss from learners
            learner_results = result.get("learners", {})
            loss = learner_results.get("default_policy", {}).get("total_loss", 0)

            print(
                f"Iter {iteration + 1:3d}: "
                f"reward={episode_reward_mean:7.2f}, "
                f"len={episode_len_mean:5.1f}, "
                f"steps={num_env_steps:7d}, "
                f"loss={loss:7.4f}"
            )

            # Save best model
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                checkpoint_dir = algo.save(args.checkpoint_dir)
                print(f"  → Saved checkpoint to {checkpoint_dir}")

            # Periodic evaluation (optional)
            if (iteration + 1) % args.eval_freq == 0:
                print(f"\n[Evaluation at iteration {iteration + 1}]")
                # TODO: Add evaluation logic
                print()

        print(f"\nTraining completed! Best reward: {best_reward:.2f}")
        print(f"Final checkpoint: {checkpoint_dir}")

    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent with RLlib on AML detection")

    # Data arguments
    parser.add_argument("--node-features", type=str, required=True, help="Path to node features CSV")
    parser.add_argument("--edges", type=str, required=True, help="Path to edges CSV")
    parser.add_argument("--classes", type=str, required=True, help="Path to classes CSV")
    parser.add_argument("--include-unknown", action="store_true", help="Include unknown class nodes")

    # Model arguments
    parser.add_argument("--gnn-type", type=str, default="rmganets", choices=["sage", "gat", "rmganets", "tgat", "tgn"])
    parser.add_argument("--node-feature-dim", type=int, default=166, help="Node feature dimension")
    parser.add_argument("--embedding-dim", type=int, default=32, help="GNN embedding dimension")
    parser.add_argument("--history-dim", type=int, default=16, help="History feature dimension")
    parser.add_argument("--multi-branch", action="store_true", help="Use multi-branch RMGANets")

    # Environment arguments
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--max-nodes", type=int, default=30, help="Max nodes in subgraph observation")
    parser.add_argument("--max-edges", type=int, default=200, help="Max edges in subgraph observation")

    # Training arguments
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size per learner")
    parser.add_argument("--buffer-capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="PER alpha (prioritization)")
    parser.add_argument("--per-beta", type=float, default=0.4, help="PER beta (importance sampling)")
    parser.add_argument("--n-step", type=int, default=3, help="N-step returns")
    parser.add_argument("--target-update-freq", type=int, default=500, help="Target network update frequency")

    # Phase 2 arguments
    parser.add_argument("--use-fraud-buffer", action="store_true", help="Use FraudAwareReplayBuffer")
    parser.add_argument("--fraud-boost-factor", type=float, default=2.0, help="Fraud episode priority multiplier")
    parser.add_argument("--detailed-callbacks", action="store_true", help="Use detailed fraud callbacks")

    # Parallelization arguments
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel env runners")
    parser.add_argument("--num-cpus", type=int, default=4, help="Total CPUs for Ray")
    parser.add_argument("--num-gpus", type=int, default=1, help="Total GPUs for Ray")
    parser.add_argument("--cpus-per-worker", type=int, default=1, help="CPUs per env runner")
    parser.add_argument("--rollout-fragment-length", type=int, default=50, help="Rollout fragment length")
    parser.add_argument("--min-sample-timesteps", type=int, default=1000, help="Min timesteps per iteration")

    # Misc arguments
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/rllib_dqn", help="Checkpoint directory")
    parser.add_argument("--eval-freq", type=int, default=10, help="Evaluation frequency (iterations)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
