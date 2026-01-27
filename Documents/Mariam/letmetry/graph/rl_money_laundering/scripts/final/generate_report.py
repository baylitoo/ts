#!/usr/bin/env python3
"""
Generate PDF report from H100 experiment results.

Usage:
    python scripts/final/generate_report.py --run-dir outputs/final/YYYYMMDD_HHMMSS/ --output report.pdf
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_experiment_stats(exp_dir: Path):
    """Load training stats from experiment directory."""
    stats_file = exp_dir / "training_stats.npz"
    config_file = exp_dir / "config.json"

    if not stats_file.exists():
        return None

    stats = dict(np.load(stats_file))
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return {"name": exp_dir.name, "stats": stats, "config": config}


def create_title_page(pdf, run_dir):
    """Create title page."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.7, "H100 Final Experiment Suite", ha="center", fontsize=24, weight="bold")
    fig.text(0.5, 0.6, "AML Detection with RL + GNNs", ha="center", fontsize=16)
    fig.text(0.5, 0.5, f"Run: {run_dir.name}", ha="center", fontsize=12)
    fig.text(0.5, 0.45, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha="center", fontsize=10)

    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def create_summary_page(pdf, experiments):
    """Create summary statistics page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    summary_text = "EXPERIMENT SUMMARY\n\n"
    summary_text += f"Total Experiments: {len(experiments)}\n\n"

    for exp in experiments:
        if not exp:
            continue

        name = exp["name"]
        stats = exp["stats"]

        final_reward = stats["rewards"][-1] if "rewards" in stats else 0
        final_detection = stats["detection_rates"][-1] * 100 if "detection_rates" in stats else 0

        summary_text += f"{name}:\n"
        summary_text += f"  Final Reward: {final_reward:.2f}\n"
        summary_text += f"  Detection Rate: {final_detection:.1f}%\n\n"

    ax.text(0.1, 0.9, summary_text, fontsize=10, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def create_plots_page(pdf, experiments):
    """Create plots page."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Reward curves
    ax = axes[0, 0]
    for exp in experiments:
        if exp and "rewards" in exp["stats"]:
            rewards = exp["stats"]["rewards"]
            ax.plot(rewards, label=exp["name"][:15], alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Learning Curves - Reward")
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)

    # Detection rate
    ax = axes[0, 1]
    for exp in experiments:
        if exp and "detection_rates" in exp["stats"]:
            rates = exp["stats"]["detection_rates"]
            ax.plot(rates * 100, label=exp["name"][:15], alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Learning Curves - Detection Rate")
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)

    # Loss curves
    ax = axes[1, 0]
    for exp in experiments:
        if exp and "losses" in exp["stats"]:
            losses = exp["stats"]["losses"]
            # Downsample for clarity
            if len(losses) > 1000:
                indices = np.linspace(0, len(losses)-1, 1000, dtype=int)
                losses = [losses[i] for i in indices]
            ax.plot(losses, label=exp["name"][:15], alpha=0.7, linewidth=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Final comparison
    ax = axes[1, 1]
    names = []
    values = []
    for exp in experiments:
        if exp and "detection_rates" in exp["stats"]:
            names.append(exp["name"][:15])
            values.append(exp["stats"]["detection_rates"][-1] * 100)

    bars = ax.barh(names, values)
    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("Final Performance Comparison")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from experiment results")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        return

    print(f"Generating report from {run_dir}")

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
                print("✗")

    if not experiments:
        print("Error: No valid experiments found")
        return

    print(f"\nFound {len(experiments)} experiments")
    print("Generating PDF report...")

    # Create PDF
    with PdfPages(args.output) as pdf:
        create_title_page(pdf, run_dir)
        create_summary_page(pdf, experiments)
        create_plots_page(pdf, experiments)

        # Add metadata
        d = pdf.infodict()
        d["Title"] = "H100 Final Experiment Suite Report"
        d["Author"] = "AML Detection Team"
        d["Subject"] = "Reinforcement Learning for AML Detection"
        d["CreationDate"] = datetime.now()

    print(f"\n✓ Report saved to {args.output}")


if __name__ == "__main__":
    main()
