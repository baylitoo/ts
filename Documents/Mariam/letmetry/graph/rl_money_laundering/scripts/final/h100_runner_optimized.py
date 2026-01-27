"""
Optimized H100 experiment runner with shared dataset loading.

Instead of reloading the dataset for each experiment, this script:
1. Loads the dataset ONCE into GPU memory
2. Runs all experiments sequentially sharing the same dataset
3. Saves ~10-15min per experiment by avoiding redundant data loading

Usage:
    python scripts/final/h100_runner_optimized.py --max_edges 500000 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rl_money_laundering.config import ExperimentConfig
from rl_money_laundering.pipeline import build_pipeline


def run_experiment_with_shared_data(
    exp_name: str,
    config: ExperimentConfig,
    artifacts,
    output_base: Path
):
    """
    Run a single experiment using pre-loaded dataset artifacts.

    Args:
        exp_name: Experiment name
        config: Experiment configuration
        artifacts: Pre-loaded pipeline artifacts (dataset, env, agent, etc.)
        output_base: Base output directory
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    print(f"Agent: {config.agent.agent_type.upper()}")
    print(f"GNN: {config.gnn.gnn_type}")
    print(f"Episodes: {config.trainer.num_episodes}")
    print(f"Max steps: {config.environment.max_steps}")
    print(f"{'='*80}\n")

    # Create output directory for this experiment
    exp_output = output_base / exp_name
    exp_output.mkdir(parents=True, exist_ok=True)

    # Save config
    config.output_dir = exp_output
    config.save(exp_output / "config.json")

    # Get artifacts
    dataset = artifacts.dataset
    env = artifacts.env
    state_encoder = artifacts.state_encoder
    agent = artifacts.agent
    trainer = artifacts.trainer
    checkpoint_manager = artifacts.checkpoint_manager

    # Update checkpoint manager output directory
    checkpoint_manager.checkpoint_dir = exp_output / "checkpoints"
    checkpoint_manager.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {len(dataset.transactions)} transactions")
    print(f"Graph: {dataset.graph.number_of_nodes()} nodes, {dataset.graph.number_of_edges()} edges")
    print(f"Fraud rate: {dataset.transactions[dataset.target_column].mean()*100:.2f}%\n")

    # Train
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%H:%M:%S')}\n")

    history = trainer.train(
        num_episodes=config.trainer.num_episodes,
        batch_size=config.trainer.batch_size,
        eval_frequency=config.trainer.eval_frequency,
        checkpoint_frequency=config.trainer.checkpoint_frequency,
        target_update_frequency=config.trainer.target_update_frequency,
        curriculum_schedule=config.trainer.curriculum_schedule,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"Training Complete - {exp_name}")
    print(f"{'='*80}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"{'='*80}\n")

    # Save final model
    final_model_path = checkpoint_manager.save_final_model(agent)
    print(f"Final model saved: {final_model_path}")

    # Save training stats
    import numpy as np
    stats_path = exp_output / "training_stats.npz"
    np.savez(
        stats_path,
        episode_rewards=np.asarray(history.get("episode_rewards", []), dtype=np.float32),
        episode_lengths=np.asarray(history.get("episode_lengths", []), dtype=np.int32),
        detection_rates=np.asarray(history.get("detection_rates", []), dtype=np.float32),
        false_positive_rates=np.asarray(history.get("false_positive_rates", []), dtype=np.float32),
        losses=np.asarray(history.get("losses", []), dtype=np.float32),
    )
    print(f"Training stats saved: {stats_path}")

    # Save summary
    summary = {
        "experiment": exp_name,
        "duration_minutes": duration / 60,
        "final_detection_rate": float(history["detection_rates"][-1]) if history.get("detection_rates") else 0.0,
        "final_avg_reward": float(np.mean(history["episode_rewards"][-100:])) if history.get("episode_rewards") else 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    with open(exp_output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run H100 experiments with optimized data loading")
    parser.add_argument("--max_edges", type=int, default=500000, help="Max edges in graph")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/h100_final"), help="Base output directory")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("H100 Optimized Experiment Runner")
    print(f"{'='*80}")
    print(f"Max edges: {args.max_edges}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create base output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================
    # LOAD DATASET ONCE
    # ===========================================
    print("="*80)
    print("LOADING DATASET (ONE TIME ONLY)")
    print("="*80)

    # Use a base config to load the dataset
    base_config = ExperimentConfig.load("configs/amlnet_rmganets_multibranch_improved.json")
    base_config.agent.device = args.device
    base_config.gnn.device = args.device

    # Build pipeline (this loads the dataset)
    artifacts = build_pipeline(
        base_config,
        output_dir=args.output_dir / "shared",
        nrows=None,  # Load full dataset
        max_edges=args.max_edges,
        fraud_ratio=None,
        device=args.device,
    )

    print("\n✓ Dataset loaded and cached in GPU memory")
    print(f"  Graph: {artifacts.dataset.graph.number_of_nodes()} nodes, {artifacts.dataset.graph.number_of_edges()} edges\n")

    # ===========================================
    # DEFINE EXPERIMENTS
    # ===========================================
    experiments = [
        # Baseline
        {
            "name": "exp01_dqn_baseline",
            "agent_type": "dqn",
            "episodes": 2000,
            "max_steps": 100,
            "epsilon_decay": 0.999,
            "guided_exploration": False,
        },

        # DQN Variants
        {
            "name": "exp02_dqn_hints",
            "agent_type": "dqn",
            "episodes": 2000,
            "max_steps": 100,
            "epsilon_decay": 0.999,
            "guided_exploration": True,
            "exploration_temperature": 1.0,
        },
        {
            "name": "exp03_dqn_hints_temp0.5",
            "agent_type": "dqn",
            "episodes": 2000,
            "max_steps": 100,
            "epsilon_decay": 0.999,
            "guided_exploration": True,
            "exploration_temperature": 0.5,
        },

        # QR-DQN Variants
        {
            "name": "exp04_qrdqn_mean",
            "agent_type": "qrdqn",
            "num_quantiles": 64,
            "n_step": 5,
            "risk_measure": "mean",
            "episodes": 2500,
            "max_steps": 100,
            "epsilon_decay": 0.999,
            "prioritized_alpha": 0.6,
            "prioritized_beta": 0.4,
        },
        {
            "name": "exp05_qrdqn_cvar90",
            "agent_type": "qrdqn",
            "num_quantiles": 64,
            "n_step": 5,
            "risk_measure": "cvar_90",
            "episodes": 2500,
            "max_steps": 100,
            "epsilon_decay": 0.999,
            "prioritized_alpha": 0.6,
            "prioritized_beta": 0.4,
        },
        {
            "name": "exp06_qrdqn_hints",
            "agent_type": "qrdqn",
            "num_quantiles": 64,
            "n_step": 5,
            "risk_measure": "mean",
            "episodes": 2500,
            "max_steps": 100,
            "epsilon_decay": 0.999,
            "guided_exploration": True,
            "exploration_temperature": 1.0,
            "prioritized_alpha": 0.6,
            "prioritized_beta": 0.4,
        },

        # RMGANETs (Heavy)
        {
            "name": "exp07_rmganets_multibranch",
            "agent_type": "qrdqn",
            "num_quantiles": 64,
            "n_step": 5,
            "risk_measure": "mean",
            "episodes": 3000,
            "max_steps": 150,
            "epsilon_decay": 0.999,
            "guided_exploration": True,
            "exploration_temperature": 1.0,
            "prioritized_alpha": 0.6,
            "prioritized_beta": 0.4,
            "gnn_type": "rmganets",
            "multi_branch": True,
        },
    ]

    # ===========================================
    # RUN ALL EXPERIMENTS
    # ===========================================
    all_summaries = []
    total_start = datetime.now()

    for i, exp_params in enumerate(experiments, 1):
        exp_name = exp_params.pop("name")

        print(f"\n\n{'#'*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp_name}")
        print(f"{'#'*80}\n")

        # Create config for this experiment
        exp_config = ExperimentConfig.load("configs/amlnet_rmganets_multibranch_improved.json")
        exp_config.name = exp_name
        exp_config.agent.device = args.device
        exp_config.gnn.device = args.device

        # Apply experiment-specific parameters
        if "agent_type" in exp_params:
            exp_config.agent.agent_type = exp_params["agent_type"]
        if "episodes" in exp_params:
            exp_config.trainer.num_episodes = exp_params["episodes"]
        if "max_steps" in exp_params:
            exp_config.environment.max_steps = exp_params["max_steps"]
        if "epsilon_decay" in exp_params:
            exp_config.agent.epsilon_decay = exp_params["epsilon_decay"]
        if "guided_exploration" in exp_params:
            exp_config.agent.use_guided_exploration = exp_params["guided_exploration"]
        if "exploration_temperature" in exp_params:
            exp_config.agent.exploration_temperature = exp_params["exploration_temperature"]

        # QR-DQN specific
        if "num_quantiles" in exp_params:
            exp_config.agent.num_quantiles = exp_params["num_quantiles"]
        if "n_step" in exp_params:
            exp_config.agent.n_step = exp_params["n_step"]
        if "risk_measure" in exp_params:
            exp_config.agent.risk_measure = exp_params["risk_measure"]
        if "prioritized_alpha" in exp_params:
            exp_config.agent.prioritized_alpha = exp_params["prioritized_alpha"]
        if "prioritized_beta" in exp_params:
            exp_config.agent.prioritized_beta = exp_params["prioritized_beta"]

        # GNN specific
        if "gnn_type" in exp_params:
            exp_config.gnn.gnn_type = exp_params["gnn_type"]
        if "multi_branch" in exp_params:
            exp_config.gnn.multi_branch = exp_params["multi_branch"]
            exp_config.trainer.multi_branch["enabled"] = exp_params["multi_branch"]

        # Run experiment
        summary = run_experiment_with_shared_data(
            exp_name=exp_name,
            config=exp_config,
            artifacts=artifacts,
            output_base=args.output_dir
        )
        all_summaries.append(summary)

    # ===========================================
    # FINAL SUMMARY
    # ===========================================
    total_duration = (datetime.now() - total_start).total_seconds()

    print(f"\n\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Number of experiments: {len(experiments)}")
    print("\nResults Summary:")
    print(f"{'-'*80}")

    for summary in all_summaries:
        print(f"{summary['experiment']:30s} | Detection: {summary['final_detection_rate']*100:5.1f}% | "
              f"Reward: {summary['final_avg_reward']:6.2f} | Time: {summary['duration_minutes']:5.1f}m")

    print(f"{'='*80}\n")

    # Save master summary
    with open(args.output_dir / "master_summary.json", "w") as f:
        json.dump({
            "experiments": all_summaries,
            "total_duration_hours": total_duration / 3600,
            "num_experiments": len(experiments),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"Master summary saved to: {args.output_dir / 'master_summary.json'}\n")


if __name__ == "__main__":
    main()
