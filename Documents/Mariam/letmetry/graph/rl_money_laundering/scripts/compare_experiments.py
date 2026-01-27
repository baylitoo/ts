"""
Compare results from different training runs (e.g., CPU vs GPU, different hyperparameters).

Usage:
    python scripts/compare_experiments.py <path_to_run1> <path_to_run2> [<path_to_run3> ...]

Example:
    python scripts/compare_experiments.py outputs/cpu_tests/09_AGGRESSIVE_CVAR outputs/h100_scaled_training
"""

import argparse
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def load_experiment(exp_path: Path) -> Dict[str, Any]:
    """Load experiment data (config + training stats)."""
    config_path = exp_path / "config.json"
    stats_path = exp_path / "training_stats.npz"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load stats
    stats = np.load(stats_path)

    return {
        "name": exp_path.name,
        "path": exp_path,
        "config": config,
        "stats": stats,
    }


def print_comparison_table(experiments: List[Dict[str, Any]]):
    """Print comparison table of key metrics."""
    print("\n" + "=" * 100)
    print("Experiment Comparison")
    print("=" * 100)

    # Extract key info
    rows = []
    for exp in experiments:
        config = exp["config"]
        stats = exp["stats"]

        # Calculate final metrics (last 100 episodes)
        rewards = stats["episode_rewards"]
        detection_rates = stats["detection_rates"]
        fp_rates = stats["false_positive_rates"]

        avg_reward = float(np.mean(rewards[-100:])) if len(rewards) > 0 else 0.0
        final_detection = float(detection_rates[-1]) if len(detection_rates) > 0 else 0.0
        final_fp = float(fp_rates[-1]) if len(fp_rates) > 0 else 0.0

        rows.append({
            "Name": exp["name"],
            "Device": config["agent"]["device"],
            "Agent": config["agent"]["agent_type"].upper(),
            "Episodes": config["trainer"]["num_episodes"],
            "Batch Size": config["trainer"]["batch_size"],
            "Hidden Dims": str(config["agent"]["hidden_dims"]),
            "Avg Reward": avg_reward,
            "Detection %": final_detection * 100,
            "FP %": final_fp * 100,
        })

    # Print table
    headers = ["Name", "Device", "Agent", "Episodes", "Batch Size", "Hidden Dims", "Avg Reward", "Detection %", "FP %"]
    col_widths = [max(len(str(row[h])) for row in rows + [{"Name": h}]) for h in headers]

    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for row in rows:
        row_line = " | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
        print(row_line)

    print("=" * 100)


def plot_training_curves(experiments: List[Dict[str, Any]], output_dir: Path):
    """Plot training curves for comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Comparison", fontsize=16, fontweight="bold")

    # 1. Episode Rewards
    ax = axes[0, 0]
    for exp in experiments:
        rewards = exp["stats"]["episode_rewards"]
        # Moving average
        window = min(50, len(rewards) // 10)
        if window > 0:
            rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(rewards_smooth, label=exp["name"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Episode Rewards Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Detection Rate
    ax = axes[0, 1]
    for exp in experiments:
        detection = exp["stats"]["detection_rates"] * 100
        ax.plot(detection, label=exp["name"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Fraud Detection Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. False Positive Rate
    ax = axes[1, 0]
    for exp in experiments:
        fp_rates = exp["stats"]["false_positive_rates"] * 100
        ax.plot(fp_rates, label=exp["name"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Episode Length
    ax = axes[1, 1]
    for exp in experiments:
        lengths = exp["stats"]["episode_lengths"]
        # Moving average
        window = min(50, len(lengths) // 10)
        if window > 0:
            lengths_smooth = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(lengths_smooth, label=exp["name"], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per Episode (smoothed)")
    ax.set_title("Episode Length Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "comparison_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Training curves saved to: {save_path}")

    plt.close()


def plot_final_performance_bars(experiments: List[Dict[str, Any]], output_dir: Path):
    """Create bar charts comparing final performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Final Performance Comparison (Last 100 Episodes)", fontsize=14, fontweight="bold")

    names = [exp["name"] for exp in experiments]
    x = np.arange(len(names))

    # 1. Average Reward
    ax = axes[0]
    rewards = [np.mean(exp["stats"]["episode_rewards"][-100:]) for exp in experiments]
    bars = ax.bar(x, rewards, color=['#2ecc71' if r > 0 else '#e74c3c' for r in rewards])
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, rewards)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom' if val > 0 else 'top')

    # 2. Detection Rate
    ax = axes[1]
    detection = [exp["stats"]["detection_rates"][-1] * 100 for exp in experiments]
    bars = ax.bar(x, detection, color='#3498db')
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Fraud Detection Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, detection)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom')

    # 3. False Positive Rate
    ax = axes[2]
    fp_rates = [exp["stats"]["false_positive_rates"][-1] * 100 for exp in experiments]
    bars = ax.bar(x, fp_rates, color='#e67e22')
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, fp_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    save_path = output_dir / "comparison_bars.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Performance bars saved to: {save_path}")

    plt.close()


def print_hyperparameter_diff(experiments: List[Dict[str, Any]]):
    """Print differences in hyperparameters between experiments."""
    print("\n" + "=" * 100)
    print("Hyperparameter Differences")
    print("=" * 100)

    if len(experiments) < 2:
        print("Need at least 2 experiments to compare")
        return

    # Compare each experiment with the first one
    base_exp = experiments[0]
    base_config = base_exp["config"]

    for exp in experiments[1:]:
        print(f"\n{exp['name']} vs {base_exp['name']}:")
        print("-" * 80)

        config = exp["config"]

        # Check agent config
        for key in base_config["agent"]:
            base_val = base_config["agent"][key]
            exp_val = config["agent"][key]
            if base_val != exp_val:
                print(f"  agent.{key}: {base_val} → {exp_val}")

        # Check GNN config
        for key in base_config["gnn"]:
            base_val = base_config["gnn"][key]
            exp_val = config["gnn"][key]
            if base_val != exp_val:
                print(f"  gnn.{key}: {base_val} → {exp_val}")

        # Check trainer config
        for key in base_config["trainer"]:
            if key == "multi_branch":
                continue  # Skip nested config
            base_val = base_config["trainer"][key]
            exp_val = config["trainer"][key]
            if base_val != exp_val:
                print(f"  trainer.{key}: {base_val} → {exp_val}")


def main():
    parser = argparse.ArgumentParser(description="Compare training experiments")
    parser.add_argument("experiments", nargs="+", help="Paths to experiment directories")
    parser.add_argument("--output", default="outputs/comparison", help="Output directory for plots")
    args = parser.parse_args()

    # Load experiments
    print("Loading experiments...")
    experiments = []
    for exp_path_str in args.experiments:
        exp_path = Path(exp_path_str)
        if not exp_path.exists():
            print(f"⚠️  Skipping non-existent path: {exp_path}")
            continue

        try:
            exp = load_experiment(exp_path)
            experiments.append(exp)
            print(f"✅ Loaded: {exp['name']}")
        except Exception as e:
            print(f"❌ Failed to load {exp_path}: {e}")

    if len(experiments) == 0:
        print("\n❌ No valid experiments to compare!")
        return

    # Print comparison
    print_comparison_table(experiments)

    # Print hyperparameter differences
    print_hyperparameter_diff(experiments)

    # Create plots
    output_dir = Path(args.output)
    print("\nGenerating comparison plots...")

    try:
        plot_training_curves(experiments, output_dir)
        plot_final_performance_bars(experiments, output_dir)
    except Exception as e:
        print(f"⚠️  Failed to generate plots: {e}")
        print("   (matplotlib may not be available)")

    print("\n" + "=" * 100)
    print("Comparison Complete!")
    print("=" * 100)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
