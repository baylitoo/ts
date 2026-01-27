"""
Experiment tracking and metrics logging.

Provides utilities for tracking training progress, logging metrics,
and generating experiment reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ExperimentTracker:
    """
    Tracks experiment metrics and training progress.

    Handles:
    - Episode-level metrics (rewards, losses, detection rates)
    - Evaluation metrics (precision, recall, F1)
    - Training statistics
    - Experiment metadata
    """

    def __init__(self, experiment_dir: Path | str, experiment_name: str):
        """
        Initialize experiment tracker.

        Args:
            experiment_dir: Directory to save experiment data
            experiment_name: Name of the experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric storage
        self.episode_metrics: dict[str, list] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilon": [],
            "detection_rates": [],
            "false_positive_rates": [],
        }

        self.eval_metrics: dict[str, list] = {
            "eval_episodes": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc": [],
        }

        self.metadata: dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": None,
            "end_time": None,
            "total_episodes": 0,
            "config": {},
        }

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        loss: float | None = None,
        epsilon: float | None = None,
        detection_rate: float | None = None,
        false_positive_rate: float | None = None,
    ) -> None:
        """
        Log metrics for a single episode.

        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length (steps)
            loss: Training loss
            epsilon: Exploration rate
            detection_rate: Fraud detection rate
            false_positive_rate: False positive rate
        """
        self.episode_metrics["episode_rewards"].append(reward)
        self.episode_metrics["episode_lengths"].append(length)

        if loss is not None:
            self.episode_metrics["losses"].append(loss)

        if epsilon is not None:
            self.episode_metrics["epsilon"].append(epsilon)

        if detection_rate is not None:
            self.episode_metrics["detection_rates"].append(detection_rate)

        if false_positive_rate is not None:
            self.episode_metrics["false_positive_rates"].append(false_positive_rate)

    def log_evaluation(
        self,
        episode: int,
        precision: float,
        recall: float,
        f1_score: float,
        auc: float | None = None,
    ) -> None:
        """
        Log evaluation metrics.

        Args:
            episode: Episode number
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            auc: AUC-ROC score
        """
        self.eval_metrics["eval_episodes"].append(episode)
        self.eval_metrics["precision"].append(precision)
        self.eval_metrics["recall"].append(recall)
        self.eval_metrics["f1_score"].append(f1_score)

        if auc is not None:
            if "auc" not in self.eval_metrics:
                self.eval_metrics["auc"] = []
            self.eval_metrics["auc"].append(auc)

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for the experiment.

        Returns:
            Dictionary of summary statistics
        """
        summary = {}

        # Episode metrics
        if self.episode_metrics["episode_rewards"]:
            rewards = np.array(self.episode_metrics["episode_rewards"])
            summary["avg_reward"] = float(np.mean(rewards))
            summary["max_reward"] = float(np.max(rewards))
            summary["final_avg_reward"] = float(np.mean(rewards[-100:]))

        if self.episode_metrics["episode_lengths"]:
            lengths = np.array(self.episode_metrics["episode_lengths"])
            summary["avg_episode_length"] = float(np.mean(lengths))

        if self.episode_metrics["detection_rates"]:
            detection_rates = np.array(self.episode_metrics["detection_rates"])
            summary["final_detection_rate"] = float(detection_rates[-1])

        # Evaluation metrics
        if self.eval_metrics["f1_score"]:
            f1_scores = np.array(self.eval_metrics["f1_score"])
            summary["best_f1_score"] = float(np.max(f1_scores))
            summary["final_f1_score"] = float(f1_scores[-1])

        if self.eval_metrics["precision"]:
            precision = np.array(self.eval_metrics["precision"])
            summary["final_precision"] = float(precision[-1])

        if self.eval_metrics["recall"]:
            recall = np.array(self.eval_metrics["recall"])
            summary["final_recall"] = float(recall[-1])

        return summary

    def save_metrics(self, filename: str = "training_stats.npz") -> Path:
        """
        Save all metrics to file.

        Args:
            filename: Filename for metrics

        Returns:
            Path to saved file
        """
        metrics_path = self.experiment_dir / filename

        # Combine all metrics
        all_metrics = {**self.episode_metrics, **self.eval_metrics}

        # Convert lists to numpy arrays
        np_metrics = {k: np.array(v) for k, v in all_metrics.items()}

        np.savez(metrics_path, **np_metrics)

        return metrics_path

    def load_metrics(self, filename: str = "training_stats.npz") -> None:
        """
        Load metrics from file.

        Args:
            filename: Filename to load
        """
        metrics_path = self.experiment_dir / filename

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        data = np.load(metrics_path)

        # Restore episode metrics
        for key in self.episode_metrics.keys():
            if key in data:
                self.episode_metrics[key] = data[key].tolist()

        # Restore eval metrics
        for key in self.eval_metrics.keys():
            if key in data:
                self.eval_metrics[key] = data[key].tolist()

    def save_metadata(self, config: Any | None = None) -> Path:
        """
        Save experiment metadata.

        Args:
            config: Optional experiment configuration

        Returns:
            Path to metadata file
        """
        metadata_path = self.experiment_dir / "metadata.json"

        if config is not None:
            self.metadata["config"] = config.__dict__ if hasattr(config, "__dict__") else config

        # Add summary statistics
        self.metadata["summary"] = self.get_summary_statistics()

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        return metadata_path

    def load_metadata(self) -> dict[str, Any]:
        """
        Load experiment metadata.

        Returns:
            Metadata dictionary
        """
        metadata_path = self.experiment_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            return json.load(f)

    def print_progress(self, episode: int, window: int = 100) -> None:
        """
        Print training progress.

        Args:
            episode: Current episode
            window: Window for averaging metrics
        """
        if episode < window:
            return

        # Calculate moving averages
        recent_rewards = self.episode_metrics["episode_rewards"][-window:]
        avg_reward = np.mean(recent_rewards)

        print(f"\nEpisode {episode}")
        print(f"  Avg Reward (last {window}): {avg_reward:.2f}")

        if self.episode_metrics["losses"]:
            recent_losses = self.episode_metrics["losses"][-window:]
            avg_loss = np.mean(recent_losses)
            print(f"  Avg Loss: {avg_loss:.4f}")

        if self.episode_metrics["epsilon"]:
            current_epsilon = self.episode_metrics["epsilon"][-1]
            print(f"  Epsilon: {current_epsilon:.4f}")

        if self.episode_metrics["detection_rates"]:
            current_detection = self.episode_metrics["detection_rates"][-1]
            print(f"  Detection Rate: {current_detection:.2%}")

    def create_report(self) -> str:
        """
        Create a text report of experiment results.

        Returns:
            Formatted text report
        """
        summary = self.get_summary_statistics()

        report_lines = [
            "=" * 80,
            f"Experiment Report: {self.experiment_name}",
            "=" * 80,
            "",
            "Training Summary:",
            f"  Total Episodes: {len(self.episode_metrics['episode_rewards'])}",
            f"  Average Reward: {summary.get('avg_reward', 0.0):.2f}",
            f"  Max Reward: {summary.get('max_reward', 0.0):.2f}",
            f"  Final Avg Reward (last 100): {summary.get('final_avg_reward', 0.0):.2f}",
            "",
            "Detection Performance:",
            f"  Final Detection Rate: {summary.get('final_detection_rate', 0.0):.2%}",
            f"  Final Precision: {summary.get('final_precision', 0.0):.2%}",
            f"  Final Recall: {summary.get('final_recall', 0.0):.2%}",
            f"  Best F1 Score: {summary.get('best_f1_score', 0.0):.2%}",
            f"  Final F1 Score: {summary.get('final_f1_score', 0.0):.2%}",
            "",
            "=" * 80,
        ]

        return "\n".join(report_lines)
