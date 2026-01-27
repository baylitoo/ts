"""
Plot training results from multiple experiment configurations.

Usage:
    python scripts/plot_training_results.py
    python scripts/plot_training_results.py --configs 05_CVAR_RISK_AVERSE 06_STRONG_PER
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def load_training_stats(output_dir: Path) -> Dict:
    """Load training statistics from NPZ file."""
    stats_file = output_dir / "training_stats.npz"

    if not stats_file.exists():
        raise FileNotFoundError(f"No training_stats.npz found in {output_dir}")

    data = np.load(stats_file, allow_pickle=True)

    # Convert to dict for easier access
    stats = {key: data[key] for key in data.files}

    # Load config for metadata
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        stats['config'] = config

    return stats


def plot_learning_curves(configs: Dict[str, Dict], save_path: Path = None):
    """Plot episode rewards over time for multiple configs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, stats in configs.items():
        if 'episode_rewards' in stats:
            episodes = np.arange(len(stats['episode_rewards']))
            rewards = stats['episode_rewards']

            # Plot raw rewards
            ax1.plot(episodes, rewards, alpha=0.3, label=f"{name} (raw)")

            # Plot smoothed rewards (moving average)
            window = 50
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(episodes[window-1:], smoothed, linewidth=2, label=f"{name} (smoothed)")

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot detection rates (recall from eval, every 50 episodes)
    for name, stats in configs.items():
        if 'detection_rates' in stats:
            # detection_rates are evaluated every 50 episodes
            eval_points = np.arange(len(stats['detection_rates'])) * 50
            detection_rates = np.array(stats['detection_rates']) * 100

            ax2.plot(eval_points, detection_rates, marker='o', linewidth=2, label=name)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Detection Rate (%)')
    ax2.set_title('Fraud Detection Rate Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    else:
        plt.show()


def plot_evaluation_metrics(configs: Dict[str, Dict], save_path: Path = None):
    """Plot evaluation metrics (recall and FP rate) over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, stats in configs.items():
        if 'detection_rates' in stats:
            # detection_rates are recall evaluated every 50 episodes
            eval_points = np.arange(len(stats['detection_rates'])) * 50
            recall = np.array(stats['detection_rates']) * 100

            ax1.plot(eval_points, recall, marker='o', linewidth=2, label=name)

        if 'false_positive_rates' in stats:
            # FP rate evaluated every 50 episodes
            eval_points = np.arange(len(stats['false_positive_rates'])) * 50
            fp_rate = np.array(stats['false_positive_rates']) * 100

            ax2.plot(eval_points, fp_rate, marker='s', linewidth=2, label=name)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Recall / Detection Rate (%)')
    ax1.set_title('Evaluation Recall Over Time (Random-Start)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('False Positive Rate (%)')
    ax2.set_title('Evaluation False Positive Rate Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved evaluation metrics to {save_path}")
    else:
        plt.show()


def plot_exploration_metrics(configs: Dict[str, Dict], save_path: Path = None):
    """Plot training losses and episode lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training losses
    for name, stats in configs.items():
        if 'losses' in stats:
            losses = stats['losses']
            steps = np.arange(len(losses))

            # Smooth losses with larger window since they're per training step
            window = 100
            if len(losses) >= window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax1.plot(steps[window-1:], smoothed, label=name, linewidth=2, alpha=0.7)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time (Smoothed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot episode lengths
    for name, stats in configs.items():
        if 'episode_lengths' in stats:
            episodes = np.arange(len(stats['episode_lengths']))
            lengths = stats['episode_lengths']

            # Smooth
            window = 50
            if len(lengths) >= window:
                smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
                ax2.plot(episodes[window-1:], smoothed, label=name, linewidth=2)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Length Over Time (Smoothed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training metrics to {save_path}")
    else:
        plt.show()


def create_comparison_table(configs: Dict[str, Dict]):
    """Create a comparison table of final metrics."""
    print("\n" + "="*100)
    print("FINAL METRICS COMPARISON")
    print("="*100)

    metrics = []

    for name, stats in configs.items():
        config_info = stats.get('config', {})
        agent_config = config_info.get('agent', {})

        # Get final evaluation metrics (from available data)
        final_recall = stats.get('detection_rates', [0])[-1] * 100 if 'detection_rates' in stats else 0
        final_fp_rate = stats.get('false_positive_rates', [0])[-1] * 100 if 'false_positive_rates' in stats else 0

        # Get best metrics
        best_recall = max(stats.get('detection_rates', [0])) * 100 if 'detection_rates' in stats else 0
        min_fp_rate = min(stats.get('false_positive_rates', [100])) * 100 if 'false_positive_rates' in stats else 100

        # Average reward over last 100 episodes
        final_rewards = stats.get('episode_rewards', [])[-100:] if 'episode_rewards' in stats else []
        avg_reward = float(np.mean(final_rewards)) if len(final_rewards) > 0 else 0.0

        # Get config parameters
        risk_measure = agent_config.get('risk_measure', 'N/A')
        epsilon_end = agent_config.get('epsilon_end', 'N/A')
        alpha = agent_config.get('prioritized_alpha', 'N/A')

        metrics.append({
            'name': name,
            'risk_measure': risk_measure,
            'epsilon_end': epsilon_end,
            'alpha': alpha,
            'final_recall': final_recall,
            'best_recall': best_recall,
            'final_fp_rate': final_fp_rate,
            'min_fp_rate': min_fp_rate,
            'avg_reward': avg_reward,
        })

    # Print table
    header = f"{'Config':<25} {'Risk':<10} {'eps_end':<8} {'alpha':<6} {'Final Recall':<14} {'Best Recall':<13} {'Final FP%':<12} {'Min FP%':<10} {'Avg Reward':<12}"
    print(header)
    print("-" * len(header))

    for m in sorted(metrics, key=lambda x: x['best_recall'], reverse=True):
        print(f"{m['name']:<25} {m['risk_measure']:<10} {m['epsilon_end']:<8} {m['alpha']:<6} "
              f"{m['final_recall']:>12.1f}% {m['best_recall']:>11.1f}% "
              f"{m['final_fp_rate']:>10.1f}% {m['min_fp_rate']:>8.1f}% {m['avg_reward']:>10.2f}")

    print("="*100 + "\n")


def inspect_stats_file(stats_file: Path):
    """Print contents of training_stats.npz file."""
    print(f"\nInspecting: {stats_file}")
    print("="*80)

    data = np.load(stats_file, allow_pickle=True)

    print("\nAvailable arrays:")
    for key in sorted(data.files):
        arr = data[key]
        if isinstance(arr, np.ndarray):
            if arr.size == 1 and arr.dtype == object:
                print(f"  {key:<30} - object (possibly dict/list)")
            else:
                print(f"  {key:<30} - shape: {arr.shape}, dtype: {arr.dtype}")
        else:
            print(f"  {key:<30} - type: {type(arr)}")

    # Show sample values for key metrics
    print("\nSample values:")
    for key in ['episode_rewards', 'detection_rates', 'false_positive_rates', 'episode_lengths']:
        if key in data:
            arr = data[key]
            if len(arr) > 0:
                print(f"  {key}:")
                print(f"    Length: {len(arr)}")
                print(f"    Last 5: {arr[-5:]}")
                if key == 'detection_rates':
                    print(f"    Best recall: {arr.max():.3f} ({arr.max()*100:.1f}%) at episode {arr.argmax() * 50}")
                elif key == 'false_positive_rates':
                    print(f"    Min FP rate: {arr.min():.3f} ({arr.min()*100:.1f}%) at episode {arr.argmin() * 50}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot training results from experiments")
    parser.add_argument('--configs', nargs='+', help='Specific config directories to plot')
    parser.add_argument('--inspect', action='store_true', help='Just inspect stats files without plotting')
    parser.add_argument('--output-dir', default='plots', help='Directory to save plots')

    args = parser.parse_args()

    # Find all config directories
    base_dir = Path("outputs/cpu_tests")

    if args.configs:
        config_dirs = [base_dir / name for name in args.configs]
    else:
        # Auto-detect all directories with training_stats.npz
        config_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "training_stats.npz").exists()]

    if not config_dirs:
        print("No training results found!")
        print(f"Looking in: {base_dir}")
        return

    print(f"Found {len(config_dirs)} experiment(s):")
    for d in config_dirs:
        print(f"  - {d.name}")

    # Inspect mode
    if args.inspect:
        for config_dir in config_dirs:
            stats_file = config_dir / "training_stats.npz"
            if stats_file.exists():
                inspect_stats_file(stats_file)
        return

    # Load all training stats
    configs = {}
    for config_dir in config_dirs:
        try:
            stats = load_training_stats(config_dir)
            configs[config_dir.name] = stats
            print(f"[OK] Loaded {config_dir.name}")
        except Exception as e:
            print(f"[FAIL] Failed to load {config_dir.name}: {e}")

    if not configs:
        print("No valid training data loaded!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    plot_learning_curves(configs, save_path=output_dir / "learning_curves.png")
    plot_evaluation_metrics(configs, save_path=output_dir / "evaluation_metrics.png")
    plot_exploration_metrics(configs, save_path=output_dir / "exploration_metrics.png")

    # Print comparison table
    create_comparison_table(configs)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
