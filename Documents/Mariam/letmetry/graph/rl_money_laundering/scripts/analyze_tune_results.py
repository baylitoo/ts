"""
Hyperparameter Tuning Results Analysis

Analyzes Ray Tune results and creates visualizations for:
- Parameter importance
- Parallel coordinates plot of top trials
- Learning curves comparison
- Best hyperparameter recommendations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from ray.tune import ExperimentAnalysis


def load_results(results_dir: str, experiment_name: str) -> ExperimentAnalysis:
    """
    Load Ray Tune experiment results.

    Args:
        results_dir: Directory containing Ray Tune results
        experiment_name: Name of the experiment

    Returns:
        ExperimentAnalysis object
    """
    experiment_path = Path(results_dir) / experiment_name
    print(f"Loading results from: {experiment_path}")

    try:
        analysis = ExperimentAnalysis(str(experiment_path))
        print(f"Loaded {len(analysis.trials)} trials")
        return analysis
    except Exception as e:
        print(f"Error loading results: {e}")
        print("Make sure the experiment directory contains valid Ray Tune results")
        raise


def get_results_dataframe(analysis: ExperimentAnalysis) -> pd.DataFrame:
    """
    Convert analysis to pandas DataFrame with all metrics.

    Args:
        analysis: Ray Tune analysis object

    Returns:
        DataFrame with trial results
    """
    df = analysis.dataframe()

    # Clean up column names (remove nested structure prefixes if any)
    df.columns = [col.replace("config/", "") for col in df.columns]

    return df


def plot_parameter_importance(
    df: pd.DataFrame,
    metric: str = "fraud_f1_score",
    top_n: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot parameter importance using correlation analysis.

    Args:
        df: Results dataframe
        metric: Metric to analyze
        top_n: Number of top parameters to show
        save_path: Optional path to save plot
    """
    # Get hyperparameter columns (exclude metrics and metadata)
    exclude_cols = [
        "trial_id", "experiment_tag", "status", "time_this_iter_s",
        "done", "timesteps_total", "episodes_total", "training_iteration",
        "time_total_s", "date", "timestamp", "hostname", "node_ip", "pid",
        "iterations_since_restore", "time_since_restore", "warmup_time",
        "logdir", "checkpoint_dir_name"
    ]

    param_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and not col.startswith("episode_")
        and not col.startswith("fraud_")
        and metric in df.columns
    ]

    # Calculate correlation with target metric
    correlations = {}
    for col in param_cols:
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            # Numeric parameters
            corr = df[col].corr(df[metric])
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        else:
            # Categorical parameters - use ANOVA
            try:
                groups = df.groupby(col)[metric].apply(list)
                if len(groups) > 1:
                    # Calculate variance ratio as proxy for importance
                    between_var = df.groupby(col)[metric].mean().var()
                    within_var = df[metric].var()
                    if within_var > 0:
                        correlations[col] = between_var / within_var
            except:
                pass

    if not correlations:
        print(f"No valid correlations found for {metric}")
        return

    # Sort by importance
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    params, importance = zip(*sorted_params)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
    bars = ax.barh(range(len(params)), importance, color=colors)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_xlabel("Importance (Absolute Correlation)")
    ax.set_title(f"Top {top_n} Most Important Hyperparameters\n(for {metric})")
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val, i, f" {val:.3f}", va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter importance plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_parallel_coordinates(
    df: pd.DataFrame,
    metric: str = "fraud_f1_score",
    top_n: int = 10,
    params: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot parallel coordinates of top trials.

    Args:
        df: Results dataframe
        metric: Metric to use for selecting top trials
        top_n: Number of top trials to show
        params: List of parameters to plot (if None, auto-select)
        save_path: Optional path to save plot
    """
    # Select top trials
    top_df = df.nlargest(top_n, metric)

    # Auto-select parameters if not provided
    if params is None:
        exclude_cols = [
            "trial_id", "experiment_tag", "status", "time_this_iter_s",
            "done", "timesteps_total", "episodes_total", "training_iteration",
            "time_total_s", "date", "timestamp", "hostname", "node_ip", "pid"
        ]
        params = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            and col != metric
        ][:8]  # Limit to 8 for readability

    # Prepare data for parallel coordinates
    plot_df = top_df[params + [metric]].copy()

    # Normalize all columns to [0, 1] for comparison
    for col in params:
        min_val = plot_df[col].min()
        max_val = plot_df[col].max()
        if max_val > min_val:
            plot_df[col] = (plot_df[col] - min_val) / (max_val - min_val)

    # Normalize metric for coloring
    metric_normalized = (plot_df[metric] - plot_df[metric].min()) / (plot_df[metric].max() - plot_df[metric].min())

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create parallel coordinates
    x = np.arange(len(params))
    for idx, row in plot_df.iterrows():
        y = [row[p] for p in params]
        color = plt.cm.viridis(metric_normalized[idx])
        ax.plot(x, y, '-o', alpha=0.6, linewidth=2, markersize=6, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.set_ylabel("Normalized Value")
    ax.set_title(f"Parallel Coordinates Plot - Top {top_n} Trials by {metric}")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=plot_df[metric].min(), vmax=plot_df[metric].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(metric)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parallel coordinates plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_curves(
    analysis: ExperimentAnalysis,
    metric: str = "fraud_f1_score",
    top_n: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot learning curves for top trials.

    Args:
        analysis: Ray Tune analysis object
        metric: Metric to plot
        top_n: Number of top trials to show
        save_path: Optional path to save plot
    """
    # Get top trials
    df = analysis.dataframe()
    top_trials = df.nlargest(top_n, metric).index

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, top_n))

    for idx, (trial_idx, color) in enumerate(zip(top_trials, colors)):
        # Get trial dataframe (history)
        trial_df = analysis.trial_dataframes[trial_idx]

        if metric in trial_df.columns:
            iterations = trial_df["training_iteration"]
            values = trial_df[metric]

            final_value = values.iloc[-1]
            label = f"Trial {idx+1} (final: {final_value:.3f})"

            ax.plot(iterations, values, '-o', color=color, label=label, linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning Curves - Top {top_n} Trials")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metric_distribution(
    df: pd.DataFrame,
    metric: str = "fraud_f1_score",
    save_path: Optional[str] = None
):
    """
    Plot distribution of metric across trials.

    Args:
        df: Results dataframe
        metric: Metric to plot
        save_path: Optional path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(df[metric], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[metric].mean():.3f}')
    ax1.axvline(df[metric].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[metric].median():.3f}')
    ax1.set_xlabel(metric)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Distribution of {metric}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot([df[metric]], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel(metric)
    ax2.set_title(f"Box Plot of {metric}")
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    stats_text = f"Mean: {df[metric].mean():.3f}\nStd: {df[metric].std():.3f}\nMin: {df[metric].min():.3f}\nMax: {df[metric].max():.3f}"
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric distribution to: {save_path}")
    else:
        plt.show()

    plt.close()


def export_best_config(
    analysis: ExperimentAnalysis,
    metric: str = "fraud_f1_score",
    output_path: Optional[str] = None
) -> Dict:
    """
    Export best hyperparameter configuration.

    Args:
        analysis: Ray Tune analysis object
        metric: Metric to optimize
        output_path: Optional path to save config JSON

    Returns:
        Best config dict
    """
    best_config = analysis.get_best_config(metric=metric, mode="max")

    # Clean up config (remove Ray internal keys)
    clean_config = {
        k: v for k, v in best_config.items()
        if not k.startswith("_")
    }

    # Get best trial results
    best_trial = analysis.get_best_trial(metric=metric, mode="max")
    best_result = best_trial.last_result

    output = {
        "best_hyperparameters": clean_config,
        "best_metrics": {
            "fraud_f1_score": best_result.get("fraud_f1_score", 0),
            "episode_reward_mean": best_result.get("episode_reward_mean", 0),
            "fraud_edges_per_episode": best_result.get("fraud_edges_per_episode", 0),
        },
        "trial_id": best_trial.trial_id,
        "checkpoint_path": str(best_trial.checkpoint.path) if best_trial.checkpoint else None,
    }

    # Print summary
    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND")
    print("="*70)
    print(f"\nBest {metric}: {output['best_metrics']['fraud_f1_score']:.4f}")
    print(f"Episode Reward: {output['best_metrics']['episode_reward_mean']:.2f}")
    print(f"\nHyperparameters:")
    for key, value in clean_config.items():
        print(f"  {key}: {value}")
    print()

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved best config to: {output_path}")

    return output


def create_comparison_table(
    df: pd.DataFrame,
    metric: str = "fraud_f1_score",
    top_n: int = 10,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create comparison table of top trials.

    Args:
        df: Results dataframe
        metric: Metric to use for ranking
        top_n: Number of top trials to include
        save_path: Optional path to save CSV

    Returns:
        Comparison dataframe
    """
    # Select top trials
    top_df = df.nlargest(top_n, metric).copy()

    # Select relevant columns
    key_params = ["lr", "gamma", "embedding_dim", "gnn_type", "batch_size", "fraud_boost_factor"]
    key_metrics = ["fraud_f1_score", "episode_reward_mean", "fraud_edges_per_episode"]

    # Filter to available columns
    available_params = [col for col in key_params if col in top_df.columns]
    available_metrics = [col for col in key_metrics if col in top_df.columns]

    comparison_df = top_df[available_params + available_metrics].copy()
    comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

    # Round numeric columns
    for col in comparison_df.columns:
        if comparison_df[col].dtype in [np.float64, np.float32]:
            comparison_df[col] = comparison_df[col].round(4)

    print("\n" + "="*70)
    print(f"TOP {top_n} TRIALS COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print()

    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"Saved comparison table to: {save_path}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Analyze Ray Tune hyperparameter search results")

    parser.add_argument("--results-dir", type=str, default="./ray_results",
                        help="Ray results directory")
    parser.add_argument("--experiment-name", type=str, default="fraud_detection_tune",
                        help="Experiment name")
    parser.add_argument("--metric", type=str, default="fraud_f1_score",
                        help="Primary metric to analyze")
    parser.add_argument("--output-dir", type=str, default="./tune_analysis",
                        help="Output directory for plots and results")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top trials to analyze")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("RAY TUNE RESULTS ANALYSIS")
    print("="*70)
    print()

    # Load results
    analysis = load_results(args.results_dir, args.experiment_name)
    df = get_results_dataframe(analysis)

    # 1. Export best configuration
    print("\n[1/6] Exporting best configuration...")
    best_config_path = output_dir / "best_config.json"
    export_best_config(analysis, metric=args.metric, output_path=str(best_config_path))

    # 2. Create comparison table
    print("\n[2/6] Creating comparison table...")
    comparison_path = output_dir / "top_trials_comparison.csv"
    create_comparison_table(df, metric=args.metric, top_n=args.top_n, save_path=str(comparison_path))

    # 3. Plot parameter importance
    print("\n[3/6] Plotting parameter importance...")
    importance_path = output_dir / "parameter_importance.png"
    plot_parameter_importance(df, metric=args.metric, top_n=10, save_path=str(importance_path))

    # 4. Plot parallel coordinates
    print("\n[4/6] Plotting parallel coordinates...")
    parallel_path = output_dir / "parallel_coordinates.png"
    plot_parallel_coordinates(df, metric=args.metric, top_n=args.top_n, save_path=str(parallel_path))

    # 5. Plot learning curves
    print("\n[5/6] Plotting learning curves...")
    curves_path = output_dir / "learning_curves.png"
    plot_learning_curves(analysis, metric=args.metric, top_n=5, save_path=str(curves_path))

    # 6. Plot metric distribution
    print("\n[6/6] Plotting metric distribution...")
    dist_path = output_dir / "metric_distribution.png"
    plot_metric_distribution(df, metric=args.metric, save_path=str(dist_path))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {best_config_path.name}: Best hyperparameters (JSON)")
    print(f"  - {comparison_path.name}: Top trials comparison (CSV)")
    print(f"  - {importance_path.name}: Parameter importance plot")
    print(f"  - {parallel_path.name}: Parallel coordinates plot")
    print(f"  - {curves_path.name}: Learning curves")
    print(f"  - {dist_path.name}: Metric distribution")
    print()


if __name__ == "__main__":
    main()
