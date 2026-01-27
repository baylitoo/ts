#!/usr/bin/env python3
"""
Analyze and compare results from H100 final experiment suite.

Usage:
    python scripts/final/analyze_experiments.py outputs/final/YYYYMMDD_HHMMSS/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_experiment_stats(exp_dir: Path) -> Dict[str, Any]:
    """Load training stats from experiment directory."""
    stats_file = exp_dir / "training_stats.npz"
    config_file = exp_dir / "config.json"

    if not stats_file.exists():
        return None

    stats = dict(np.load(stats_file))

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return {
        "name": exp_dir.name,
        "stats": stats,
        "config": config
    }


def plot_learning_curves(experiments: List[Dict], output_dir: Path):
    """Plot reward and detection rate learning curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Reward curve
    ax = axes[0, 0]
    for exp in experiments:
        if exp and "rewards" in exp["stats"]:
            rewards = exp["stats"]["rewards"]
            ax.plot(rewards, label=exp["name"], alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Detection rate curve
    ax = axes[0, 1]
    for exp in experiments:
        if exp and "detection_rates" in exp["stats"]:
            rates = exp["stats"]["detection_rates"]
            ax.plot(rates, label=exp["name"], alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss curve
    ax = axes[1, 0]
    for exp in experiments:
        if exp and "losses" in exp["stats"]:
            losses = exp["stats"]["losses"]
            # Smooth with moving average
            window = min(50, len(losses) // 10)
            if window > 1:
                losses_smooth = pd.Series(losses).rolling(window, min_periods=1).mean()
                ax.plot(losses_smooth, label=exp["name"], alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # False positive rate
    ax = axes[1, 1]
    for exp in experiments:
        if exp and "false_positive_rates" in exp["stats"]:
            fp_rates = exp["stats"]["false_positive_rates"]
            ax.plot(fp_rates, label=exp["name"], alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved learning curves to {output_dir / 'learning_curves.png'}")


def plot_final_comparison(experiments: List[Dict], output_dir: Path):
    """Plot final performance comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = []
    detection_rates = []
    fp_rates = []

    for exp in experiments:
        if exp and "detection_rates" in exp["stats"]:
            names.append(exp["name"].replace("exp", "").replace("_", " ").title())
            detection_rates.append(exp["stats"]["detection_rates"][-1] * 100)
            if "false_positive_rates" in exp["stats"]:
                fp_rates.append(exp["stats"]["false_positive_rates"][-1] * 100)
            else:
                fp_rates.append(0)

    # Detection rate
    ax = axes[0]
    bars = ax.barh(names, detection_rates, color=sns.color_palette("viridis", len(names)))
    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("Final Detection Rate Comparison")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, detection_rates)):
        ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=9)

    # False positive rate
    ax = axes[1]
    bars = ax.barh(names, fp_rates, color=sns.color_palette("rocket_r", len(names)))
    ax.set_xlabel("False Positive Rate (%)")
    ax.set_title("Final False Positive Rate Comparison")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, fp_rates)):
        ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "final_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved final comparison to {output_dir / 'final_comparison.png'}")


def generate_summary_table(experiments: List[Dict], output_dir: Path):
    """Generate summary statistics table."""
    rows = []

    for exp in experiments:
        if not exp:
            continue

        stats = exp["stats"]
        config = exp["config"]

        # Extract final metrics
        final_reward = stats["rewards"][-1] if "rewards" in stats else 0
        final_detection = stats["detection_rates"][-1] * 100 if "detection_rates" in stats else 0
        final_fp = stats["false_positive_rates"][-1] * 100 if "false_positive_rates" in stats else 0

        # Extract config params
        agent_type = config.get("agent", {}).get("agent_type", "unknown")
        episodes = config.get("trainer", {}).get("num_episodes", 0)
        epsilon_end = config.get("agent", {}).get("epsilon_end", 0)

        rows.append({
            "Experiment": exp["name"],
            "Agent": agent_type.upper(),
            "Episodes": episodes,
            "ε_end": epsilon_end,
            "Final Reward": f"{final_reward:.2f}",
            "Detection (%)": f"{final_detection:.1f}",
            "FP Rate (%)": f"{final_fp:.1f}"
        })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary table to {csv_path}")

    # Save as markdown
    md_path = output_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"✓ Saved markdown summary to {md_path}")

    # Print to console
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze H100 experiment results")
    parser.add_argument("run_dir", type=Path, help="Run directory (outputs/final/YYYYMMDD_HHMMSS/)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        return

    print(f"Analyzing experiments in {run_dir}")

    # Load all experiments
    experiments = []
    for exp_dir in sorted(run_dir.iterdir()):
        if exp_dir.is_dir() and exp_dir.name.startswith("exp"):
            print(f"  Loading {exp_dir.name}...", end=" ")
            exp_data = load_experiment_stats(exp_dir)
            if exp_data:
                experiments.append(exp_data)
                print("✓")
            else:
                print("✗ (no stats)")

    if not experiments:
        print("Error: No valid experiments found")
        return

    print(f"\nFound {len(experiments)} experiments")

    # Create analysis output directory
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Generate plots and tables
    print("\nGenerating visualizations...")
    plot_learning_curves(experiments, analysis_dir)
    plot_final_comparison(experiments, analysis_dir)
    generate_summary_table(experiments, analysis_dir)

    print(f"\n✓ Analysis complete! Results saved to {analysis_dir}")
    print("\nView results:")
    print(f"  - Learning curves: {analysis_dir / 'learning_curves.png'}")
    print(f"  - Final comparison: {analysis_dir / 'final_comparison.png'}")
    print(f"  - Summary table: {analysis_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
