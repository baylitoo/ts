import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

from rl_money_laundering.agent import DQNAgent
from rl_money_laundering.config import (
    AgentConfig,
    GNNConfig,
    MultiBranchLossConfig,
    TrainerConfig,
)
from rl_money_laundering.environment import AMLDetectionEnv
from rl_money_laundering.features.temporal import add_temporal_features_to_graph
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.trainer import AMLTrainer


def build_toy_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    node_attrs = {
        "node_type": "customer",
        "total_sent": 1000.0,
        "total_received": 950.0,
        "balance": 50.0,
        "num_transactions_sent": 5,
        "num_transactions_received": 4,
        "risk_score": 0.2,
        "is_suspicious": False,
        "tx_velocity": 0.5,
        "business_hour_ratio": 0.7,
        "periodicity_score": 0.1,
        "degree_centrality": 0.1,
        "clustering_coefficient": 0.0,
        "in_degree": 1,
        "out_degree": 1,
    }

    for i in range(6):
        attrs = node_attrs.copy()
        attrs["risk_score"] = 0.9 if i == 5 else 0.2
        attrs["is_suspicious"] = i == 5
        graph.add_node(f"n{i}", **attrs)

    edges = [
        ("n0", "n1", False),
        ("n1", "n2", False),
        ("n2", "n3", False),
        ("n3", "n4", False),
        ("n4", "n5", True),
        ("n1", "n5", True),
        ("n2", "n0", False),
    ]
    for u, v, is_fraud in edges:
        graph.add_edge(
            u,
            v,
            amount=100.0,
            hour=10,
            is_money_laundering=is_fraud,
            scheme_id="schemeA" if is_fraud else None,
        )

    # Add minimal temporal dataframe-like info
    transactions = []
    for idx, (u, v, is_fraud) in enumerate(edges):
        transactions.append(
            {
                "nameOrig": u,
                "nameDest": v,
                "amount": 100.0,
                "step": idx,
                "hour": 10,
                "is_fraud": int(is_fraud),
            }
        )
    graph = add_temporal_features_to_graph(graph, transactions)
    return graph


def test_rmganets_multibranch_end_to_end():
    graph = build_toy_graph()
    env = AMLDetectionEnv(
        graph=graph,
        max_steps=6,
        max_neighbors=4,
        intrinsic_reward_weight=0.1,
    )

    gnn_cfg = GNNConfig(
        node_feature_dim=20,
        gnn_type="rmganets",
        embedding_dim=32,
        history_dim=16,
        multi_branch=True,
        use_dqn_enhancement=True,
    )
    state_encoder = StateEncoder(
        node_feature_dim=gnn_cfg.node_feature_dim,
        gnn_type=gnn_cfg.gnn_type,
        embedding_dim=gnn_cfg.embedding_dim,
        history_dim=gnn_cfg.history_dim,
        device="cpu",
        multi_branch=gnn_cfg.multi_branch,
        use_dqn_enhancement=gnn_cfg.use_dqn_enhancement,
        num_classes=gnn_cfg.num_classes,
    )

    state_dim = state_encoder.get_state_dim()
    action_dim = int(env.action_space.n)

    agent_cfg = AgentConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=5e-4,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        buffer_capacity=500,
        hidden_dims=[64, 32],
        device="cpu",
    )
    agent = DQNAgent(
        state_dim=agent_cfg.state_dim,
        action_dim=agent_cfg.action_dim,
        learning_rate=agent_cfg.learning_rate,
        gamma=agent_cfg.gamma,
        epsilon_start=agent_cfg.epsilon_start,
        epsilon_end=agent_cfg.epsilon_end,
        epsilon_decay=agent_cfg.epsilon_decay,
        buffer_capacity=agent_cfg.buffer_capacity,
        use_prioritized_replay=agent_cfg.use_prioritized_replay,
        hidden_dims=agent_cfg.hidden_dims,
        device=agent_cfg.device,
    )

    multi_branch_cfg = MultiBranchLossConfig(
        enabled=True,
        variant="improved",
        lambda_branch=0.25,
        epsilon_dqn=0.2,
        beta_reg=0.05,
        temporal_decay=0.1,
        adaptive_weighting=True,
        learning_rate=5e-4,
        log_frequency=1,
    )

    trainer_cfg = TrainerConfig(
        num_episodes=8,
        batch_size=16,
        eval_frequency=4,
        checkpoint_frequency=8,
        target_update_frequency=2,
        curriculum_schedule=[1.0, 0.5],
        multi_branch=multi_branch_cfg,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AMLTrainer(
            graph=graph,
            agent=agent,
            state_encoder=state_encoder,
            env=env,
            fraud_subgraphs=[],
            output_dir=Path(tmpdir),
            device="cpu",
            multi_branch_config=trainer_cfg.multi_branch,
        )

        history = trainer.train(
            num_episodes=trainer_cfg.num_episodes,
            curriculum_schedule=trainer_cfg.curriculum_schedule,
            eval_frequency=trainer_cfg.eval_frequency,
            checkpoint_frequency=trainer_cfg.checkpoint_frequency,
            target_update_frequency=trainer_cfg.target_update_frequency,
            batch_size=trainer_cfg.batch_size,
        )

        assert len(history["episode_rewards"]) == trainer_cfg.num_episodes
        assert trainer.multi_branch_enabled
        assert trainer.multi_branch_metrics, "Expected multi-branch metrics to be collected"
        # Ensure metrics contain expected keys
        last_metrics = trainer.multi_branch_metrics[-1]
        assert "total" in last_metrics and np.isfinite(last_metrics["total"])
