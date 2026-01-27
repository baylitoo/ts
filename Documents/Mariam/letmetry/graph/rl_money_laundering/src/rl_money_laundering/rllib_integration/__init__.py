"""
RLlib Integration Module

Provides Ray RLlib integration for distributed training of GNN-based RL agents
on graph-structured anti-money laundering detection tasks.
"""

from .graph_space import GraphSpace
from .env_wrapper import create_rllib_env, register_aml_env
from .gnn_rl_module import GNNDQNModule
from .batch_utils import GraphBatchCollator, batch_graph_observations
from .fraud_replay_buffer import FraudAwareReplayBuffer, create_fraud_aware_replay_buffer
from .fraud_callbacks import FraudDetectionCallbacks, DetailedFraudCallbacks, create_fraud_callbacks

__all__ = [
    # Core components
    "GraphSpace",
    "create_rllib_env",
    "register_aml_env",
    "GNNDQNModule",
    # Batch processing
    "GraphBatchCollator",
    "batch_graph_observations",
    # Fraud-specific enhancements
    "FraudAwareReplayBuffer",
    "create_fraud_aware_replay_buffer",
    "FraudDetectionCallbacks",
    "DetailedFraudCallbacks",
    "create_fraud_callbacks",
]
