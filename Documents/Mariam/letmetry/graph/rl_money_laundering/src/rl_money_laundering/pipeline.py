"""
Pipeline assembly utilities.

Provides a single entrypoint for constructing the full training stack
(dataset loader, graph, environment, encoder, agent, trainer) from an
``ExperimentConfig``. The goal is to keep high-level scripts lightweight
and make it easy to swap architectures or datasets purely via config files.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, SupportsFloat, SupportsInt, cast

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from .agent import AgentProtocol, DQNAgent, QRDQNAgent
from .config import AgentConfig, ExperimentConfig
from .datasets.amlnet import AMLNetConfig, AMLNetLoader
from .datasets.base import BaseGraphDataset
from .datasets.elliptic import EllipticConfig, EllipticLoader
from .environment import AMLDetectionEnv
from .features import BaseNodeFeatureExtractor, EllipticNodeFeatureExtractor, NodeFeatureExtractor
from .features.network_pyg import add_network_features_pyg
from .features.temporal import add_temporal_features_to_graph
from .gnn_encoder import StateEncoder
from .trainer import AMLTrainer
from .utils import CheckpointManager


@dataclass
class DatasetArtifacts:
    """Artifacts specific to the dataset stage of the pipeline."""

    loader: BaseGraphDataset
    transactions: pd.DataFrame
    graph: nx.DiGraph
    target_column: str


@dataclass
class PipelineArtifacts:
    """Full set of components required to run training/evaluation."""

    config: ExperimentConfig
    dataset: DatasetArtifacts
    feature_extractor: BaseNodeFeatureExtractor
    env: AMLDetectionEnv
    state_encoder: StateEncoder
    agent: AgentProtocol
    trainer: AMLTrainer
    checkpoint_manager: CheckpointManager


def _ensure_hour_column(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with an ``hour`` column (derive it if missing)."""
    if "hour" in transactions.columns:
        return transactions

    hourly = transactions.copy()
    hourly["hour"] = hourly["step"] % 24
    return hourly


def _annotate_amlnet_nodes(graph: nx.DiGraph, transactions: pd.DataFrame, target_col: str) -> None:
    """Aggregate AMLNet transaction stats onto node attributes."""
    stats: dict[str, dict[str, Any]] = {}

    def _get_stats(node: str) -> dict[str, Any]:
        if node not in stats:
            stats[node] = {
                "total_sent": 0.0,
                "total_received": 0.0,
                "num_sent": 0,
                "num_received": 0,
                "amounts": [],
                "steps": [],
                "business_hours": 0,
                "fraud_hits": 0,
                "min_step": float("inf"),
                "max_step": 0,
            }
        return stats[node]

    for row in transactions.itertuples(index=False):
        src = str(getattr(row, "nameOrig"))
        dst = str(getattr(row, "nameDest"))
        amount_value = cast(SupportsFloat, getattr(row, "amount"))
        amount = float(amount_value)
        step_value = cast(SupportsInt, getattr(row, "step"))
        step = int(step_value)
        hour_raw = getattr(row, "hour", None)
        hour = step % 24 if hour_raw is None else int(cast(SupportsInt, hour_raw))
        label_value = cast(SupportsInt, getattr(row, target_col))
        label = int(label_value)

        src_stats = _get_stats(src)
        dst_stats = _get_stats(dst)

        src_stats["total_sent"] += amount
        src_stats["num_sent"] += 1
        src_stats["amounts"].append(amount)
        src_stats["steps"].append(step)

        dst_stats["total_received"] += amount
        dst_stats["num_received"] += 1
        dst_stats["amounts"].append(amount)
        dst_stats["steps"].append(step)

        for stats_entry in (src_stats, dst_stats):
            stats_entry["min_step"] = min(stats_entry["min_step"], step)
            stats_entry["max_step"] = max(stats_entry["max_step"], step)
            if 8 <= hour < 18:
                stats_entry["business_hours"] += 1

        if label == 1:
            src_stats["fraud_hits"] += 1
            dst_stats["fraud_hits"] += 1

    senders = set(transactions["nameOrig"])
    recipients = set(transactions["nameDest"])

    for node in graph.nodes():
        node_stats: Optional[Dict[str, Any]] = stats.get(node)
        node_data = graph.nodes[node]

        if node_stats is None:
            node_data.setdefault("total_sent", 0.0)
            node_data.setdefault("total_received", 0.0)
            node_data.setdefault("num_transactions_sent", 0)
            node_data.setdefault("num_transactions_received", 0)
            node_data.setdefault("avg_transaction_amount", 0.0)
            node_data.setdefault("max_transaction_amount", 0.0)
            node_data.setdefault("min_transaction_amount", 0.0)
            node_data.setdefault("transaction_velocity", 0.0)
            node_data.setdefault("tx_velocity", 0.0)
            node_data.setdefault("business_hour_ratio", 0.0)
            node_data.setdefault("periodicity_score", 0.0)
            node_data.setdefault("risk_score", 0.0)
            node_data.setdefault("is_suspicious", False)
            node_data.setdefault("label", 0)
            node_data.setdefault("balance", 0.0)
            node_data.setdefault("first_step", 0)
            node_data.setdefault("last_step", 0)
        else:
            counts = node_stats["num_sent"] + node_stats["num_received"]
            amounts = node_stats["amounts"]
            steps = node_stats["steps"]

            velocity_denominator = max(node_stats["max_step"] - node_stats["min_step"] + 1, 1)
            velocity = counts / velocity_denominator if counts else 0.0

            periodicity = 0.0
            if len(steps) > 1:
                diffs = np.diff(sorted(steps))
                periodicity = float(1.0 / (1.0 + np.std(diffs)))

            risk_score = node_stats["fraud_hits"] / counts if counts else 0.0

            node_data["total_sent"] = float(node_stats["total_sent"])
            node_data["total_received"] = float(node_stats["total_received"])
            node_data["num_transactions_sent"] = int(node_stats["num_sent"])
            node_data["num_transactions_received"] = int(node_stats["num_received"])
            node_data["avg_transaction_amount"] = float(np.mean(amounts)) if amounts else 0.0
            node_data["max_transaction_amount"] = float(np.max(amounts)) if amounts else 0.0
            node_data["min_transaction_amount"] = float(np.min(amounts)) if amounts else 0.0
            node_data["transaction_velocity"] = float(velocity)
            node_data["tx_velocity"] = float(velocity)
            node_data["business_hour_ratio"] = float(node_stats["business_hours"] / counts) if counts else 0.0
            node_data["periodicity_score"] = periodicity
            node_data["risk_score"] = float(risk_score)
            node_data["is_suspicious"] = bool(risk_score > 0)
            node_data["label"] = int(risk_score > 0)
            node_data["balance"] = float(node_stats["total_received"] - node_stats["total_sent"])
            node_data["first_step"] = int(node_stats["min_step"]) if node_stats["min_step"] != float("inf") else 0
            node_data["last_step"] = int(node_stats["max_step"])

        if node in senders and node in recipients:
            node_type = "intermediary"
        elif node in senders:
            node_type = "customer"
        else:
            node_type = "merchant"
        node_data.setdefault("node_type", node_type)




def _build_agent_from_config(cfg: AgentConfig) -> AgentProtocol:
    """Instantiate the appropriate agent based on the configuration."""
    if cfg.agent_type == "qrdqn":
        return QRDQNAgent(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            num_quantiles=cfg.num_quantiles,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            n_step=cfg.n_step,
            epsilon_start=cfg.epsilon_start,
            epsilon_end=cfg.epsilon_end,
            epsilon_decay=cfg.epsilon_decay,
            buffer_capacity=cfg.buffer_capacity,
            hidden_dims=cfg.hidden_dims,
            risk_measure=cfg.risk_measure,
            device=cfg.device,
            prioritized_alpha=cfg.prioritized_alpha,
            prioritized_beta=cfg.prioritized_beta,
            prioritized_beta_annealing=cfg.prioritized_beta_annealing,
            positive_fraction=cfg.positive_fraction,
            use_double_dqn=cfg.use_double_dqn,
            use_guided_exploration=cfg.use_guided_exploration,
            exploration_temperature=cfg.exploration_temperature,
        )

    return DQNAgent(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        buffer_capacity=cfg.buffer_capacity,
        use_prioritized_replay=cfg.use_prioritized_replay,
        hidden_dims=cfg.hidden_dims,
        device=cfg.device,
        positive_fraction=cfg.positive_fraction,
        use_double_dqn=cfg.use_double_dqn,
        use_guided_exploration=cfg.use_guided_exploration,
        exploration_temperature=cfg.exploration_temperature,
    )


def _sample_amlnet_transactions(
    transactions: pd.DataFrame,
    target_col: str,
    max_edges: Optional[int],
    fraud_ratio: Optional[float],
    random_state: int,
) -> pd.DataFrame:
    """Return a subset of AMLNet transactions respecting optional constraints."""
    if max_edges is None:
        return transactions

    subset = transactions
    if fraud_ratio is None:
        subset = transactions.head(max_edges)
    else:
        fraud_rows = transactions[transactions[target_col] == 1]
        normal_rows = transactions[transactions[target_col] == 0]

        desired_fraud = max(int(max_edges * fraud_ratio), 1)
        desired_fraud = min(desired_fraud, len(fraud_rows))

        fraud_sample = fraud_rows.sample(n=desired_fraud, random_state=random_state) if len(fraud_rows) > desired_fraud else fraud_rows
        remaining = max_edges - len(fraud_sample)

        if remaining > 0 and len(normal_rows) > 0:
            normal_sample = normal_rows.sample(n=min(remaining, len(normal_rows)), random_state=random_state)
            subset = pd.concat([fraud_sample, normal_sample], ignore_index=True)
        else:
            subset = fraud_sample

        subset = subset.drop_duplicates(subset=["nameOrig", "nameDest", "step"], keep="first")
        subset = subset.sort_values("step").reset_index(drop=True)

    return subset


def _build_amlnet_dataset(
    config: ExperimentConfig,
    *,
    nrows: Optional[int],
    max_edges: Optional[int],
    fraud_ratio: Optional[float],
) -> DatasetArtifacts:
    loader = AMLNetLoader(AMLNetConfig(csv_path=config.dataset_path))
    transactions = loader.load_raw(nrows=nrows)
    target_col = loader.config.target_column

    subset = _sample_amlnet_transactions(
        transactions,
        target_col=target_col,
        max_edges=max_edges,
        fraud_ratio=fraud_ratio,
        random_state=config.seed,
    )

    graph = loader.build_graph(subset)
    annotate_view = _ensure_hour_column(subset)

    _annotate_amlnet_nodes(graph, annotate_view, target_col=target_col)
    add_network_features_pyg(graph, device="cpu")
    add_temporal_features_to_graph(
        graph,
        annotate_view[["nameOrig", "nameDest", "amount", "step", "hour"]].copy(),
    )

    return DatasetArtifacts(
        loader=loader,
        transactions=subset,
        graph=graph,
        target_column=target_col,
    )


def _build_elliptic_dataset(
    config: ExperimentConfig,
    *,
    nrows: Optional[int],
) -> DatasetArtifacts:
    loader = EllipticLoader(
        EllipticConfig(
            data_dir=config.dataset_path,
            include_unlabeled=bool(config.include_unlabeled),
        )
    )
    loader.load_raw()

    processed_df, target = loader.feature_pipeline()

    if nrows is not None and nrows > 0:
        processed_df = processed_df.iloc[:nrows].copy()
        target = target.iloc[: len(processed_df)].reset_index(drop=True)

    processed_df = processed_df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    graph = loader.build_graph(
        include_unknown=loader.config.include_unlabeled,
        processed_features=processed_df,
    )

    transactions = processed_df.copy()
    transactions[loader.config.target_column] = target

    return DatasetArtifacts(
        loader=loader,
        transactions=transactions,
        graph=graph,
        target_column=loader.config.target_column,
    )


def _extract_fraud_subgraphs(graph: nx.DiGraph, k_hop: int = 1) -> list[nx.DiGraph]:
    """
    Extract subgraphs around fraudulent edges for curriculum learning.

    Args:
        graph: Full transaction graph
        k_hop: Number of hops to include around each fraud edge

    Returns:
        List of subgraphs centered on fraud edges
    """
    # Find all fraud edges
    fraud_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get('is_fraud', 0) == 1 or d.get('isFraud', 0) == 1 or d.get('is_money_laundering', 0) == 1
    ]

    if not fraud_edges:
        return []

    subgraphs = []

    # For each fraud edge, extract k-hop neighborhood
    for u, v in fraud_edges:
        # Get k-hop neighborhood around both nodes
        nodes = set([u, v])

        # Expand outward k hops
        for _ in range(k_hop):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(graph.successors(node))
                new_nodes.update(graph.predecessors(node))
            nodes.update(new_nodes)

        # Extract subgraph
        subgraph = graph.subgraph(nodes).copy()

        # Keep subgraphs with reasonable size (increased limit to 500)
        if 3 <= subgraph.number_of_nodes() <= 500:
            subgraphs.append(subgraph)

    return subgraphs


def build_pipeline(
    config: ExperimentConfig,
    *,
    output_dir: Optional[Path] = None,
    nrows: Optional[int] = None,
    max_edges: Optional[int] = None,
    fraud_ratio: Optional[float] = None,
    device: Optional[str] = None,
) -> PipelineArtifacts:
    """
    Assemble the full training pipeline from a configuration object.

    Args:
        config: Experiment configuration
        output_dir: Optional output directory override
        nrows: Optional override for number of rows to load
        max_edges: Optional limit for number of edges/transactions when building graph
        fraud_ratio: Desired fraud ratio when sampling AMLNet subsets (0-1)
        device: Optional device override ("cuda", "cpu", "auto")

    Returns:
        PipelineArtifacts with ready-to-use components
    """
    cfg = copy.deepcopy(config)

    if output_dir is not None:
        cfg.output_dir = Path(output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if nrows is not None:
        cfg.nrows = nrows

    if device is not None:
        cfg.agent.device = device
        cfg.gnn.device = device

    if cfg.dataset_type == "amlnet":
        dataset = _build_amlnet_dataset(
            cfg,
            nrows=cfg.nrows,
            max_edges=max_edges,
            fraud_ratio=fraud_ratio,
        )
        feature_extractor: BaseNodeFeatureExtractor = NodeFeatureExtractor(
            include_temporal=True,
            include_network=True,
            node_feature_dim=cfg.gnn.node_feature_dim,
        )
    elif cfg.dataset_type == "elliptic":
        dataset = _build_elliptic_dataset(
            cfg,
            nrows=cfg.nrows,
        )
        feature_extractor = EllipticNodeFeatureExtractor(
            node_feature_dim=cfg.gnn.node_feature_dim
        )
    else:
        raise ValueError(f"Unsupported dataset type: {cfg.dataset_type}")

    env = AMLDetectionEnv(
        graph=dataset.graph,
        max_steps=cfg.environment.max_steps,
        max_neighbors=cfg.environment.max_neighbors,
        intrinsic_reward_weight=cfg.environment.intrinsic_reward_weight,
        fraud_reward_base=cfg.environment.fraud_reward,
        false_alarm_penalty_base=cfg.environment.false_alarm_penalty,
    )

    action_space = getattr(env.action_space, "n", cfg.environment.max_neighbors + 1)

    state_encoder = StateEncoder(
        node_feature_dim=cfg.gnn.node_feature_dim,
        gnn_type=cfg.gnn.gnn_type,
        embedding_dim=cfg.gnn.embedding_dim,
        history_dim=cfg.gnn.history_dim,
        device=cfg.gnn.device,
        multi_branch=getattr(cfg.gnn, "multi_branch", False),
        use_dqn_enhancement=getattr(cfg.gnn, "use_dqn_enhancement", False),
        num_classes=getattr(cfg.gnn, "num_classes", 2),
        time_attributes=getattr(cfg.gnn, "time_attributes", None),
    )

    state_dim = state_encoder.get_state_dim()
    cfg.agent.state_dim = state_dim
    cfg.agent.action_dim = action_space

    agent = _build_agent_from_config(cfg.agent)

    # Extract fraud subgraphs for curriculum learning
    fraud_subgraphs = _extract_fraud_subgraphs(dataset.graph)
    print(f"Extracted {len(fraud_subgraphs)} fraud subgraphs for curriculum learning")

    trainer = AMLTrainer(
        graph=dataset.graph,
        agent=agent,
        state_encoder=state_encoder,
        env=env,
        fraud_subgraphs=fraud_subgraphs,
        output_dir=str(cfg.output_dir),
        device=cfg.agent.device,
        multi_branch_config=cfg.trainer.multi_branch,
        feature_extractor=feature_extractor,
    )

    checkpoint_manager = CheckpointManager(Path(cfg.output_dir) / "checkpoints")

    return PipelineArtifacts(
        config=cfg,
        dataset=dataset,
        feature_extractor=feature_extractor,
        env=env,
        state_encoder=state_encoder,
        agent=agent,
        trainer=trainer,
        checkpoint_manager=checkpoint_manager,
    )


__all__ = [
    "DatasetArtifacts",
    "PipelineArtifacts",
    "build_pipeline",
]
