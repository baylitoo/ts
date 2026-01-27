"""
Train RL agent for AML detection using configuration presets or external config files.

Examples
--------
  # Quick test preset
  python scripts/train_agent.py --config quick_test

  # Full AMLNet training
  python scripts/train_agent.py --config amlnet_full

  # Custom configuration JSON
  python scripts/train_agent.py --config_file configs/amlnet_rmganets_multibranch_improved.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rl_money_laundering.config import (
    ExperimentConfig,
    get_elliptic_config,
    get_full_amlnet_config,
    get_quick_test_config,
)
from rl_money_laundering.pipeline import build_pipeline


class _Tee(io.TextIOBase):
    """Duplication helper that writes to multiple streams (e.g., stdout + log file)."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except ValueError:
                # Stream might already be closed; ignore.
                continue
        self.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except ValueError:
                continue


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load experiment configuration based on CLI arguments."""
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
    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir)
    if args.episodes is not None:
        config.trainer.num_episodes = args.episodes

    return config


def _parse_hidden_dims(value: str) -> list[int]:
    """Parse comma-separated hidden layer dimensions."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer (e.g., 256,128).")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid hidden dims '{value}'. Use comma-separated integers (e.g., 256,128)."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agent for AML detection (config driven)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", choices=["quick_test", "amlnet_full", "elliptic"], help="Preset configuration")
    group.add_argument("--config_file", type=str, help="Path to custom config JSON file")

    parser.add_argument("--nrows", type=int, help="Limit number of rows loaded from dataset")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--episodes", type=int, help="Override total training episodes")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Force computation device")
    parser.add_argument("--max_edges", type=int, help="Limit number of edges when constructing graph (AMLNet)")
    parser.add_argument(
        "--fraud_ratio",
        type=float,
        help="Target fraud ratio when sampling AMLNet subsets (0-1). Ignored for other datasets.",
    )
    parser.add_argument(
        "--include-unlabeled",
        dest="include_unlabeled",
        action="store_true",
        help="Elliptic-only: keep unknown nodes when building the graph (restores sparse class prior).",
    )

    parser.add_argument("--agent-type", choices=["dqn", "qrdqn"], help="Override agent type (default from config)")
    parser.add_argument("--num-quantiles", type=int, help="QR-DQN: number of quantiles")
    parser.add_argument("--n-step", type=int, dest="n_step", help="QR-DQN: N-step horizon")
    parser.add_argument("--risk-measure", choices=["mean", "cvar_90", "cvar_95"], help="QR-DQN: risk measure")
    parser.add_argument("--prioritized-alpha", type=float, dest="prioritized_alpha", help="PER alpha exponent")
    parser.add_argument("--prioritized-beta", type=float, dest="prioritized_beta", help="PER importance-sampling beta")
    parser.add_argument("--prioritized-beta-annealing", type=float, dest="prioritized_beta_annealing", help="PER beta annealing rate")
    parser.add_argument("--positive-fraction", type=float, dest="positive_fraction", help="Target fraction of fraud-positive samples per batch")
    parser.add_argument("--disable-double-dqn", action="store_true", help="Disable Double DQN target updates")
    parser.add_argument(
        "--hidden-dims",
        type=_parse_hidden_dims,
        help="Comma-separated hidden layer dimensions for the agent network (e.g., 512,256,128).",
    )
    parser.add_argument("--buffer-capacity", type=int, dest="buffer_capacity", help="Experience replay buffer capacity")

    # Exploration parameters
    parser.add_argument("--epsilon-end", type=float, dest="epsilon_end", help="Minimum epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=float, dest="epsilon_decay", help="Epsilon decay rate per episode")
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="Maximum steps per episode")

    # Hint-guided exploration
    parser.add_argument("--guided-exploration", action="store_true", help="Enable hint-guided exploration")
    parser.add_argument("--no-guided-exploration", action="store_true", help="Disable hint-guided exploration")
    parser.add_argument("--exploration-temperature", type=float, dest="exploration_temperature", help="Temperature for hint-based sampling")

    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Training batch size")

    return parser.parse_args()


def _apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> list[str]:
    """Apply CLI overrides to the loaded configuration and collect any warnings."""
    warnings: list[str] = []

    # Device overrides
    if args.device:
        config.agent.device = args.device
        config.gnn.device = args.device

    if args.agent_type:
        config.agent.agent_type = args.agent_type

    agent_type = config.agent.agent_type
    qr_specific_overrides = [
        ("--num-quantiles", args.num_quantiles, "num_quantiles"),
        ("--n-step", getattr(args, "n_step", None), "n_step"),
        ("--risk-measure", args.risk_measure, "risk_measure"),
        ("--prioritized-alpha", args.prioritized_alpha, "prioritized_alpha"),
        ("--prioritized-beta", args.prioritized_beta, "prioritized_beta"),
        ("--prioritized-beta-annealing", args.prioritized_beta_annealing, "prioritized_beta_annealing"),
    ]
    provided_qr_flags = [flag for flag, value, _ in qr_specific_overrides if value is not None]

    if provided_qr_flags and agent_type != "qrdqn":
        joined = ", ".join(provided_qr_flags)
        warnings.append(
            f"[WARN] Ignoring QR-DQN specific options ({joined}) because agent_type={agent_type.upper()}."
            " Use --agent-type qrdqn or update your config to apply these overrides."
        )
    elif agent_type == "qrdqn":
        for _flag, value, attr in qr_specific_overrides:
            if value is not None:
                setattr(config.agent, attr, value)

    if args.positive_fraction is not None:
        config.agent.positive_fraction = args.positive_fraction
    if args.buffer_capacity is not None:
        config.agent.buffer_capacity = args.buffer_capacity
    if args.disable_double_dqn:
        config.agent.use_double_dqn = False
    if args.hidden_dims:
        config.agent.hidden_dims = args.hidden_dims

    # Exploration parameters
    if args.epsilon_end is not None:
        config.agent.epsilon_end = args.epsilon_end
    if args.epsilon_decay is not None:
        config.agent.epsilon_decay = args.epsilon_decay
    if args.max_steps is not None:
        config.environment.max_steps = args.max_steps

    # Hint-guided exploration
    if args.guided_exploration:
        config.agent.use_guided_exploration = True
    if args.no_guided_exploration:
        config.agent.use_guided_exploration = False
    if args.exploration_temperature is not None:
        config.agent.exploration_temperature = args.exploration_temperature

    # Training parameters
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size

    if args.include_unlabeled:
        config.include_unlabeled = True
    return warnings


def _run_training(
    config: ExperimentConfig,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    print("=" * 80)
    print("RL-based AML Detection - Training")
    print("=" * 80)
    print(f"Experiment: {config.name}")
    print(f"Dataset: {config.dataset_type} ({config.dataset_path})")
    print(f"Episodes: {config.trainer.num_episodes}")
    print(f"Device: {config.agent.device}")
    print(f"Agent Type: {config.agent.agent_type}")
    print(f"GNN Type: {config.gnn.gnn_type}")
    print(f"Use Prioritized Replay: {config.agent.use_prioritized_replay}")
    print(f"Output: {config.output_dir}")
    print("=" * 80)

    artifacts = build_pipeline(
        config,
        output_dir=output_path,
        nrows=config.nrows,
        max_edges=args.max_edges,
        fraud_ratio=args.fraud_ratio,
        device=args.device,
    )

    dataset = artifacts.dataset
    fraud_total = int(dataset.transactions[dataset.target_column].sum())
    fraud_rate = float(dataset.transactions[dataset.target_column].mean())
    print("\n[1/5] Loading dataset...")
    print(f"  Transactions: {len(dataset.transactions)} | Fraudulent: {fraud_total} ({fraud_rate*100:.2f}%)")
    print(f"  Graph: {dataset.graph.number_of_nodes()} nodes, {dataset.graph.number_of_edges()} edges")

    env = artifacts.env
    action_space_size = int(getattr(env.action_space, "n", config.environment.max_neighbors + 1))
    print("\n[2/5] Creating RL environment...")
    print(f"  Action space: {action_space_size} actions")

    state_encoder = artifacts.state_encoder
    print("\n[3/5] Initializing GNN state encoder...")
    print(f"  State dimension: {state_encoder.get_state_dim()}")

    agent = artifacts.agent
    print(f"\n[4/5] Initializing agent ({config.agent.agent_type.upper()})...")
    if config.agent.agent_type == "qrdqn":
        print(f"  Quantiles: {config.agent.num_quantiles} | N-step: {config.agent.n_step} | Risk: {config.agent.risk_measure}")
    print(f"  Hidden dims: {config.agent.hidden_dims}")
    print(f"  Replay buffer: {config.agent.buffer_capacity} capacity")

    trainer = artifacts.trainer
    print("\n[5/5] Creating trainer...")

    checkpoint_manager = artifacts.checkpoint_manager
    config.save(output_path / "config.json")
    print(f"\nConfiguration saved to: {output_path / 'config.json'}")

    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")

    history = trainer.train(
        num_episodes=config.trainer.num_episodes,
        batch_size=config.trainer.batch_size,
        eval_frequency=config.trainer.eval_frequency,
        checkpoint_frequency=config.trainer.checkpoint_frequency,
        target_update_frequency=config.trainer.target_update_frequency,
        curriculum_schedule=config.trainer.curriculum_schedule,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    final_model_path = checkpoint_manager.save_final_model(agent)
    print(f"\nFinal model saved to: {final_model_path}")

    episode_rewards = history.get("episode_rewards", [])
    detection_rates = history.get("detection_rates", [])
    false_positive_rates = history.get("false_positive_rates", [])
    loss_history = history.get("losses", [])

    final_avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
    final_detection_rate = float(detection_rates[-1]) if detection_rates else 0.0
    final_fpr = float(false_positive_rates[-1]) if false_positive_rates else 0.0
    final_loss = float(np.mean(loss_history[-100:])) if loss_history else None

    print("\nFinal Performance:")
    print(f"  Avg Reward (last 100 episodes): {final_avg_reward:.2f}")
    print(f"  Detection Rate: {final_detection_rate*100:.1f}%")
    print(f"  False Positive Rate: {final_fpr*100:.1f}%")
    if final_loss is not None:
        print(f"  Avg Loss (last 100 updates): {final_loss:.4f}")

    stats_path = output_path / "training_stats.npz"
    np.savez(
        stats_path,
        episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
        episode_lengths=np.asarray(history.get("episode_lengths", []), dtype=np.int32),
        detection_rates=np.asarray(detection_rates, dtype=np.float32),
        false_positive_rates=np.asarray(false_positive_rates, dtype=np.float32),
        losses=np.asarray(loss_history, dtype=np.float32),
    )
    print(f"\nTraining stats saved to: {stats_path}")

    print(f"\nAll outputs saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    warning_messages = _apply_cli_overrides(config, args)

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = output_path / "train.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _Tee(sys.stdout, log_file)
        tee_stderr = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            for message in warning_messages:
                print(message)
            _run_training(config, args, output_path)
            print(f"\nFull log written to: {log_path}")

