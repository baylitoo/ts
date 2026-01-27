"""
Main Training Script for RL-based AML Detection

Usage:
    python scripts/train_agent.py --dataset amlnet --episodes 1000
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from rl_money_laundering.agent import DQNAgent
from rl_money_laundering.data_loader import AMLNetDataLoader
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.trainer import AMLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for AML detection")

    # Data arguments
    parser.add_argument("--dataset", type=str, default="amlnet",
                       choices=["amlnet", "elliptic"],
                       help="Dataset to use")
    parser.add_argument("--data_path", type=str, default="data/amlnet/AMLNet_August_2025.csv",
                       help="Path to dataset")
    parser.add_argument("--nrows", type=int, default=None,
                       help="Number of rows to load (None = all)")

    # Training arguments
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for DQN training")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")

    # Model arguments
    parser.add_argument("--gnn_type", type=str, default="sage",
                       choices=["sage", "gat"],
                       help="Type of GNN for state encoding")
    parser.add_argument("--embedding_dim", type=int, default=32,
                       help="GNN embedding dimension")
    parser.add_argument("--hidden_dims", type=str, default="128,128,64",
                       help="DQN hidden layer dimensions (comma-separated)")
    parser.add_argument("--use_prioritized_replay", action="store_true",
                       help="Use prioritized experience replay")

    # Environment arguments
    parser.add_argument("--max_steps", type=int, default=20,
                       help="Maximum steps per episode")
    parser.add_argument("--intrinsic_reward_weight", type=float, default=0.1,
                       help="Weight for intrinsic curiosity reward")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/rl_agent",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--checkpoint_freq", type=int, default=100,
                       help="Save checkpoint every N episodes")
    parser.add_argument("--eval_freq", type=int, default=50,
                       help="Evaluate every N episodes")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device for training")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("RL-based AML Detection - Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {device}")
    print(f"GNN Type: {args.gnn_type}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading dataset...")
    if args.dataset == "amlnet":
        loader = AMLNetDataLoader(args.data_path)
        loader.load_data(nrows=args.nrows)
        loader.extract_features()
        graph = loader.build_transaction_graph()
        fraud_subgraphs = loader.get_fraud_subgraphs(k_hop=2)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported")

    # Create environment
    print("\n[2/5] Creating RL environment...")
    env = AMLDetectionEnv(
        graph=graph,
        max_steps=args.max_steps,
        intrinsic_reward_weight=args.intrinsic_reward_weight
    )

    # Create state encoder
    print("\n[3/5] Initializing GNN state encoder...")
    node_feature_dim = 12  # Based on feature extraction in trainer
    state_encoder = StateEncoder(
        node_feature_dim=node_feature_dim,
        gnn_type=args.gnn_type,
        embedding_dim=args.embedding_dim,
        device=device
    )

    # Create DQN agent
    print("\n[4/5] Initializing DQN agent...")
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    print(f"  Hidden dimensions: {hidden_dims}")

    agent = DQNAgent(
        state_dim=state_encoder.get_state_dim(),
        action_dim=env.action_space.n,
        learning_rate=args.lr,
        gamma=args.gamma,
        use_prioritized_replay=args.use_prioritized_replay,
        hidden_dims=hidden_dims,
        device=device
    )

    # Create trainer
    print("\n[5/5] Creating trainer...")
    trainer = AMLTrainer(
        graph=graph,
        agent=agent,
        state_encoder=state_encoder,
        env=env,
        fraud_subgraphs=fraud_subgraphs,
        output_dir=args.output_dir,
        device=device
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")

    history = trainer.train(
        num_episodes=args.episodes,
        eval_frequency=args.eval_freq,
        checkpoint_frequency=args.checkpoint_freq,
        batch_size=args.batch_size
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final average reward: {np.mean(history['episode_rewards'][-100:]):.2f}")
    print(f"Final detection rate: {history['detection_rates'][-1]*100:.1f}%")
    print(f"Final FPR: {history['false_positive_rates'][-1]*100:.1f}%")
    print(f"\nCheckpoints saved to: {args.output_dir}/checkpoints/")
    print(f"Training stats saved to: {args.output_dir}/training_stats.npz")


if __name__ == "__main__":
    import numpy as np
    main()
