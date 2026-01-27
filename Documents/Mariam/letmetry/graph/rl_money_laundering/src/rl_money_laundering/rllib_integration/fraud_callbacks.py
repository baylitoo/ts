"""
Fraud Detection Callbacks for RLlib

Custom callbacks to track fraud-specific metrics during training.
"""

from typing import Dict, Optional
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode


class FraudDetectionCallbacks(RLlibCallback):
    """
    Callbacks for tracking fraud detection metrics.

    Monitors:
    - Fraud detection rate (precision, recall, F1)
    - Path coverage and exploration
    - Fraud edge discovery rate
    - Episode-level fraud statistics
    """

    def __init__(self):
        super().__init__()
        # Use instance-level storage for episode data keyed by episode ID
        self._episode_data = {}
        # Store completed episode metrics for aggregation
        self._completed_episode_metrics = []

    def on_episode_start(
        self,
        *,
        episode: SingleAgentEpisode,
        **kwargs
    ) -> None:
        """Initialize episode-level tracking."""
        # Initialize tracking data for this episode using episode ID
        episode_id = episode.id_
        self._episode_data[episode_id] = {
            "fraud_edges_found": 0,
            "total_edges_traversed": 0,
            "unique_nodes_visited": set(),
            "fraud_nodes_visited": set(),
        }

    def on_episode_step(
        self,
        *,
        episode: SingleAgentEpisode,
        **kwargs
    ) -> None:
        """Track step-level fraud metrics."""
        episode_id = episode.id_
        if episode_id not in self._episode_data:
            return

        # Get info dict from most recent step
        if len(episode) > 0:
            info = episode.get_infos(-1)  # Get last info
        else:
            info = None

        if info is not None:
            # Track fraud edges discovered
            if info.get("is_fraud_edge", False):
                self._episode_data[episode_id]["fraud_edges_found"] += 1

            # Track exploration
            self._episode_data[episode_id]["total_edges_traversed"] += 1

            current_node = info.get("current_node")
            if current_node is not None:
                self._episode_data[episode_id]["unique_nodes_visited"].add(current_node)

                if info.get("is_fraud_node", False):
                    self._episode_data[episode_id]["fraud_nodes_visited"].add(current_node)

    def on_episode_end(
        self,
        *,
        episode: SingleAgentEpisode,
        **kwargs
    ) -> None:
        """Compute episode-level fraud metrics."""
        episode_id = episode.id_
        if episode_id not in self._episode_data:
            return

        # Extract episode data
        data = self._episode_data[episode_id]
        fraud_edges = data.get("fraud_edges_found", 0)
        total_edges = data.get("total_edges_traversed", 1)
        unique_nodes = len(data.get("unique_nodes_visited", set()))
        fraud_nodes = len(data.get("fraud_nodes_visited", set()))

        # Compute metrics
        fraud_edge_rate = fraud_edges / max(total_edges, 1)
        fraud_node_rate = fraud_nodes / max(unique_nodes, 1)

        # Store metrics internally for aggregation in on_train_result
        episode_metrics = {
            "fraud_edges_found": fraud_edges,
            "fraud_edge_discovery_rate": fraud_edge_rate,
            "unique_nodes_explored": unique_nodes,
            "fraud_nodes_visited": fraud_nodes,
            "fraud_node_rate": fraud_node_rate,
            "exploration_efficiency": unique_nodes / max(total_edges, 1),
        }

        # Get final info for precision/recall if available
        if len(episode) > 0:
            final_info = episode.get_infos(-1)
        else:
            final_info = None
        if final_info is not None:
            # Precision: fraction of predicted frauds that are correct
            if "true_positives" in final_info and "false_positives" in final_info:
                tp = final_info["true_positives"]
                fp = final_info["false_positives"]
                precision = tp / max(tp + fp, 1)
                episode_metrics["fraud_precision"] = precision

            # Recall: fraction of actual frauds detected
            if "true_positives" in final_info and "false_negatives" in final_info:
                tp = final_info["true_positives"]
                fn = final_info["false_negatives"]
                recall = tp / max(tp + fn, 1)
                episode_metrics["fraud_recall"] = recall

                # Calculate F1 if we have precision
                if "true_positives" in final_info and "false_positives" in final_info:
                    prec = tp / max(tp + fp, 1)
                    rec = recall
                    f1 = 2 * (prec * rec) / max(prec + rec, 1e-8)
                    episode_metrics["fraud_f1_score"] = f1

        # Store completed episode metrics
        self._completed_episode_metrics.append(episode_metrics)

        # Clean up episode data to prevent memory leak
        if episode_id in self._episode_data:
            del self._episode_data[episode_id]

    def on_train_result(
        self,
        *,
        algorithm,
        result: Dict,
        **kwargs
    ) -> None:
        """Aggregate training metrics."""
        # Aggregate metrics from completed episodes
        if self._completed_episode_metrics:
            # Calculate means across all completed episodes
            num_episodes = len(self._completed_episode_metrics)

            fraud_edges_mean = sum(m.get("fraud_edges_found", 0) for m in self._completed_episode_metrics) / num_episodes
            fraud_rate_mean = sum(m.get("fraud_edge_discovery_rate", 0) for m in self._completed_episode_metrics) / num_episodes
            exploration_mean = sum(m.get("unique_nodes_explored", 0) for m in self._completed_episode_metrics) / num_episodes

            # Log to console
            print(
                f"  [Fraud Metrics] "
                f"Edges found: {fraud_edges_mean:.2f}, "
                f"Discovery rate: {fraud_rate_mean:.3f}, "
                f"Nodes explored: {exploration_mean:.1f}"
            )

            # Add fraud-specific result keys for tracking
            result["fraud_edges_per_episode"] = fraud_edges_mean
            result["fraud_discovery_rate"] = fraud_rate_mean
            result["exploration_coverage"] = exploration_mean

            # Track F1 score if available
            f1_scores = [m.get("fraud_f1_score") for m in self._completed_episode_metrics if "fraud_f1_score" in m]
            if f1_scores:
                f1_mean = sum(f1_scores) / len(f1_scores)
                result["fraud_f1_score"] = f1_mean
                print(f"  [Fraud F1]: {f1_mean:.3f}")

            # Clear metrics after aggregation
            self._completed_episode_metrics.clear()


class DetailedFraudCallbacks(FraudDetectionCallbacks):
    """
    Extended callbacks with more detailed fraud analysis.

    Additional tracking:
    - Fraud pattern recognition (layering, structuring, etc.)
    - Path diversity metrics
    - Temporal fraud patterns
    """

    def on_episode_start(self, **kwargs) -> None:
        """Initialize extended tracking."""
        super().on_episode_start(**kwargs)
        episode = kwargs["episode"]
        episode_id = episode.id_

        # Extended metrics (stored in parent's _episode_data)
        if episode_id in self._episode_data:
            self._episode_data[episode_id]["fraud_patterns"] = {
                "layering": 0,
                "structuring": 0,
                "round_tripping": 0,
            }
            self._episode_data[episode_id]["edge_timestamps"] = []
            self._episode_data[episode_id]["path_diversity_score"] = 0.0

    def on_episode_step(self, **kwargs) -> None:
        """Track detailed step information."""
        super().on_episode_step(**kwargs)
        episode = kwargs["episode"]
        episode_id = episode.id_

        if episode_id not in self._episode_data:
            return

        if len(episode) > 0:
            info = episode.get_infos(-1)
        else:
            info = None

        if info is not None:
            # Track fraud patterns
            pattern_type = info.get("fraud_pattern_type")
            if pattern_type and pattern_type in self._episode_data[episode_id].get("fraud_patterns", {}):
                self._episode_data[episode_id]["fraud_patterns"][pattern_type] += 1

            # Track temporal information
            timestamp = info.get("timestamp")
            if timestamp is not None:
                self._episode_data[episode_id]["edge_timestamps"].append(timestamp)

    def on_episode_end(self, **kwargs) -> None:
        """Compute detailed fraud metrics."""
        # Call parent to handle base metrics
        super().on_episode_end(**kwargs)

        episode = kwargs["episode"]
        episode_id = episode.id_

        if episode_id not in self._episode_data:
            return

        # Get the last episode metrics dict that parent just added
        if self._completed_episode_metrics:
            last_metrics = self._completed_episode_metrics[-1]

            # Add fraud pattern distribution
            patterns = self._episode_data[episode_id].get("fraud_patterns", {})
            for pattern_type, count in patterns.items():
                last_metrics[f"fraud_pattern_{pattern_type}"] = count

            # Add temporal analysis
            timestamps = self._episode_data[episode_id].get("edge_timestamps", [])
            if len(timestamps) > 1:
                # Compute velocity (edges per time unit)
                time_span = max(timestamps) - min(timestamps)
                if time_span > 0:
                    velocity = len(timestamps) / time_span
                    last_metrics["traversal_velocity"] = velocity

                # Temporal diversity (std of timestamp gaps)
                gaps = np.diff(sorted(timestamps))
                if len(gaps) > 0:
                    temporal_diversity = np.std(gaps)
                    last_metrics["temporal_diversity"] = temporal_diversity


def create_fraud_callbacks(detailed: bool = False) -> type:
    """
    Factory function for creating fraud callbacks.

    Args:
        detailed: If True, use DetailedFraudCallbacks with extended metrics

    Returns:
        Callback class to use in config

    Example:
        >>> callbacks_cls = create_fraud_callbacks(detailed=True)
        >>> config.callbacks(callbacks_cls)
    """
    return DetailedFraudCallbacks if detailed else FraudDetectionCallbacks
