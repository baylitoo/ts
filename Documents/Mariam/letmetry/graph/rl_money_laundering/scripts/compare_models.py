"""
Model Comparison and Ablation Study Script

Systematically compares different model configurations:
1. Baseline (GraphSAGE, no multi-branch)
2. GAT (no multi-branch)
3. RMGANets (paper loss)
4. RMGANets (improved loss)
5. RMGANets (improved loss + all enhancements)

Tracks: F1, Precision, Recall, Training time, Convergence speed
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from rl_money_laundering.config import ExperimentConfig, GNNConfig, MultiBranchLossConfig
from rl_money_laundering.pipeline import build_pipeline


def create_baseline_sage_config(output_dir: Path) -> ExperimentConfig:
    """Baseline: GraphSAGE without multi-branch"""
    config = ExperimentConfig(
        name="baseline_sage",
        dataset_type="simple",
        dataset_path=None
    )

    config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="sage",
        embedding_dim=64,
        multi_branch=False
    )

    config.trainer.multi_branch = MultiBranchLossConfig(enabled=False)

    return config


def create_baseline_gat_config(output_dir: Path) -> ExperimentConfig:
    """Baseline: GAT without multi-branch"""
    config = ExperimentConfig(
        name="baseline_gat",
        dataset_type="simple",
        dataset_path=None
    )

    config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="gat",
        embedding_dim=64,
        multi_branch=False
    )

    config.trainer.multi_branch = MultiBranchLossConfig(enabled=False)

    return config


def create_rmganets_paper_config(output_dir: Path) -> ExperimentConfig:
    """RMGANets with paper loss (no improvements)"""
    config = ExperimentConfig(
        name="rmganets_paper",
        dataset_type="simple",
        dataset_path=None
    )

    config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=64,
        multi_branch=True,
        num_classes=2,
        use_dqn_enhancement=False
    )

    config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="paper",
        lambda_branch=0.25,
        epsilon_dqn=0.4
    )

    return config


def create_rmganets_improved_config(output_dir: Path) -> ExperimentConfig:
    """RMGANets with improved loss (all enhancements)"""
    config = ExperimentConfig(
        name="rmganets_improved",
        dataset_type="simple",
        dataset_path=None
    )

    config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=64,
        multi_branch=True,
        num_classes=2,
        use_dqn_enhancement=False
    )

    config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="improved",
        lambda_branch=0.25,
        epsilon_dqn=0.4,
        beta_reg=0.1,
        temporal_decay=0.1,
        adaptive_weighting=True
    )

    return config


def create_rmganets_full_config(output_dir: Path) -> ExperimentConfig:
    """RMGANets with all enhancements (improved loss + DQN enhancement)"""
    config = ExperimentConfig(
        name="rmganets_full",
        dataset_type="simple",
        dataset_path=None
    )

    config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=64,
        multi_branch=True,
        num_classes=2,
        use_dqn_enhancement=True  # Enable DQN enhancement
    )

    config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="improved",
        lambda_branch=0.25,
        epsilon_dqn=0.4,
        beta_reg=0.1,
        temporal_decay=0.1,
        adaptive_weighting=True
    )

    return config


def train_and_evaluate(
    config: ExperimentConfig,
    num_episodes: int,
    nrows: int,
    max_edges: int,
    output_dir: Path,
    device: str
) -> Dict[str, Any]:
    """
    Train a model and return evaluation metrics

    Returns:
        Dict with keys:
        - name: Model name
        - training_time: Total training time (seconds)
        - episode_rewards: List of episode rewards
        - detection_rates: List of detection rates
        - false_positive_rates: List of false positive rates
        - final_detection_rate: Final detection rate
        - final_fp_rate: Final false positive rate
        - convergence_episode: Episode where model converged (reward stable)
    """
    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"{'='*60}")

    model_output_dir = output_dir / config.name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    print("Building pipeline...")
    artifacts = build_pipeline(
        config,
        nrows=nrows,
        max_edges=max_edges,
        output_dir=model_output_dir,
        device=device
    )

    # Train
    print(f"Training for {num_episodes} episodes...")
    start_time = time.time()

    history = artifacts.trainer.train(
        num_episodes=num_episodes,
        eval_frequency=max(10, num_episodes // 10),
        checkpoint_frequency=max(50, num_episodes // 5),
        batch_size=64
    )

    training_time = time.time() - start_time

    # Calculate convergence episode (when reward variance stabilizes)
    convergence_episode = detect_convergence(history['episode_rewards'])

    # Get final metrics
    final_detection_rate = history['detection_rates'][-1] if history['detection_rates'] else 0.0
    final_fp_rate = history['false_positive_rates'][-1] if history['false_positive_rates'] else 1.0

    # Calculate F1 score (approximation)
    precision = final_detection_rate / (final_detection_rate + final_fp_rate + 1e-6)
    recall = final_detection_rate
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    results = {
        'name': config.name,
        'gnn_type': config.gnn.gnn_type,
        'multi_branch': config.trainer.multi_branch.enabled,
        'variant': config.trainer.multi_branch.variant if config.trainer.multi_branch.enabled else None,
        'training_time': training_time,
        'episode_rewards': history['episode_rewards'],
        'detection_rates': history['detection_rates'],
        'false_positive_rates': history['false_positive_rates'],
        'final_detection_rate': final_detection_rate,
        'final_fp_rate': final_fp_rate,
        'final_f1': f1,
        'final_precision': precision,
        'final_recall': recall,
        'convergence_episode': convergence_episode,
        'avg_reward_last_100': np.mean(history['episode_rewards'][-100:]) if history['episode_rewards'] else 0.0,
        'multi_branch_metrics': artifacts.trainer.multi_branch_metrics if artifacts.trainer.multi_branch_enabled else None
    }

    print(f"\nResults for {config.name}:")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Final F1: {f1:.2%}")
    print(f"  Final Detection Rate: {final_detection_rate:.2%}")
    print(f"  Final FP Rate: {final_fp_rate:.2%}")
    print(f"  Convergence Episode: {convergence_episode}/{num_episodes}")

    return results


def detect_convergence(rewards: List[float], window: int = 50, threshold: float = 0.1) -> int:
    """
    Detect convergence episode based on reward variance

    Args:
        rewards: List of episode rewards
        window: Window size for variance calculation
        threshold: Variance threshold for convergence

    Returns:
        Episode number where convergence detected (or total episodes if never converged)
    """
    if len(rewards) < window * 2:
        return len(rewards)

    for i in range(window, len(rewards) - window):
        window_rewards = rewards[i:i+window]
        variance = np.var(window_rewards)

        if variance < threshold:
            return i

    return len(rewards)


def create_comparison_plots(results: List[Dict[str, Any]], output_dir: Path):
    """
    Create comparison plots and tables

    Args:
        results: List of result dicts from train_and_evaluate
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt

    # Create results directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Bar chart: Final F1 scores
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r['name'] for r in results]
    f1_scores = [r['final_f1'] for r in results]

    bars = ax.bar(names, f1_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Comparison: F1 Score')
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / "f1_comparison.png", dpi=300)
    plt.close()

    # 2. Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [r['training_time'] for r in results]

    bars = ax.bar(names, times, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Model Comparison: Training Time')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / "training_time_comparison.png", dpi=300)
    plt.close()

    # 3. Learning curves (all models)
    fig, ax = plt.subplots(figsize=(12, 6))

    for r in results:
        rewards = r['episode_rewards']
        # Smooth with moving average
        window = min(50, len(rewards) // 10)
        if window > 0:
            smoothed = pd.Series(rewards).rolling(window=window).mean()
            ax.plot(smoothed, label=r['name'], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Learning Curves (Moving Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "learning_curves.png", dpi=300)
    plt.close()

    print(f"\nPlots saved to: {plots_dir}")


def save_results_table(results: List[Dict[str, Any]], output_dir: Path):
    """
    Save results as CSV and markdown table
    """
    # Create DataFrame
    df = pd.DataFrame([
        {
            'Model': r['name'],
            'GNN Type': r['gnn_type'],
            'Multi-Branch': r['multi_branch'],
            'Variant': r['variant'] or 'N/A',
            'F1 Score': f"{r['final_f1']:.2%}",
            'Precision': f"{r['final_precision']:.2%}",
            'Recall': f"{r['final_recall']:.2%}",
            'FP Rate': f"{r['final_fp_rate']:.2%}",
            'Training Time (s)': f"{r['training_time']:.1f}",
            'Convergence Episode': r['convergence_episode']
        }
        for r in results
    ])

    # Save CSV
    csv_path = output_dir / "results_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved to: {csv_path}")

    # Save Markdown
    md_path = output_dir / "results_comparison.md"
    with open(md_path, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")

        best_f1 = max(results, key=lambda r: r['final_f1'])
        fastest = min(results, key=lambda r: r['training_time'])

        f.write(f"- **Best F1 Score**: {best_f1['name']} ({best_f1['final_f1']:.2%})\n")
        f.write(f"- **Fastest Training**: {fastest['name']} ({fastest['training_time']:.1f}s)\n")

    print(f"Results Markdown saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare different model configurations")
    parser.add_argument("--num-episodes", type=int, default=200, help="Training episodes per model")
    parser.add_argument("--nrows", type=int, default=1000, help="Dataset size (rows)")
    parser.add_argument("--max-edges", type=int, default=5000, help="Max edges in graph")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if True else "cpu", help="Training device")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", "baseline_sage", "baseline_gat", "rmganets_paper", "rmganets_improved", "rmganets_full"],
                        help="Which models to train")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define model configs
    config_builders = {
        "baseline_sage": create_baseline_sage_config,
        "baseline_gat": create_baseline_gat_config,
        "rmganets_paper": create_rmganets_paper_config,
        "rmganets_improved": create_rmganets_improved_config,
        "rmganets_full": create_rmganets_full_config
    }

    # Determine which models to run
    if "all" in args.models:
        models_to_run = list(config_builders.keys())
    else:
        models_to_run = args.models

    print("="*60)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Models to compare: {', '.join(models_to_run)}")
    print(f"Episodes per model: {args.num_episodes}")
    print(f"Dataset size: {args.nrows} rows, {args.max_edges} edges")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)

    # Train each model
    results = []
    for model_name in models_to_run:
        config = config_builders[model_name](args.output_dir)

        result = train_and_evaluate(
            config=config,
            num_episodes=args.num_episodes,
            nrows=args.nrows,
            max_edges=args.max_edges,
            output_dir=args.output_dir,
            device=args.device
        )

        results.append(result)

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save raw results as JSON
    results_json = args.output_dir / "results_raw.json"
    with open(results_json, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            r_copy['episode_rewards'] = [float(x) for x in r_copy['episode_rewards']]
            r_copy['detection_rates'] = [float(x) for x in r_copy['detection_rates']]
            r_copy['false_positive_rates'] = [float(x) for x in r_copy['false_positive_rates']]
            if r_copy['multi_branch_metrics']:
                r_copy['multi_branch_metrics'] = [
                    {k: float(v) for k, v in m.items()} for m in r_copy['multi_branch_metrics']
                ]
            serializable_results.append(r_copy)

        json.dump(serializable_results, f, indent=2)

    print(f"Raw results saved to: {results_json}")

    # Create comparison tables
    save_results_table(results, args.output_dir)

    # Create plots (requires matplotlib)
    try:
        create_comparison_plots(results, args.output_dir)
    except ImportError:
        print("\nWarning: matplotlib not available, skipping plots")

    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
