"""
Centralized configuration system for temporal graph guardrail experiments.

All experiments use configuration objects for reproducibility and type safety.
AMLNet/Elliptic presets serve as case studies, but the schema accommodates
any dataset plugged into the pipeline factories.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Literal, Optional

import json
import torch


@dataclass
class AgentConfig:
    """Configuration for RL agents (DQN / QR-DQN).

    Args:
        state_dim: Dimension of state representation
        action_dim: Number of possible actions
        learning_rate: Optimizer learning rate
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate (0.1 recommended for large graphs)
        epsilon_decay: Exploration decay rate per episode (0.998 for gradual decay)
        buffer_capacity: Size of experience replay buffer
        use_prioritized_replay: Use prioritized experience replay
        hidden_dims: Hidden layer dimensions for the Q-network
        device: Training device ("auto", "cuda", or "cpu")
        agent_type: Agent variant ("dqn" or "qrdqn")
        num_quantiles: Number of quantiles (QR-DQN only)
        n_step: N-step horizon (QR-DQN only)
        risk_measure: Risk measure for QR-DQN value selection
        prioritized_alpha: PER prioritization exponent
        prioritized_beta: PER importance-sampling exponent
        prioritized_beta_annealing: Annealing rate for beta
        positive_fraction: Target fraction of fraud-positive samples per batch
        use_double_dqn: Whether to use Double DQN target selection
        use_guided_exploration: Enable hint-guided exploration during training
        exploration_temperature: Temperature for softmax hint-based sampling (1.0=neutral)
    """

    state_dim: int
    action_dim: int
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1  # INCREASED from 0.01 for better long-term exploration
    epsilon_decay: float = 0.998  # SLOWER from 0.995 for gradual decay
    buffer_capacity: int = 10000
    use_prioritized_replay: bool = True
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    device: str = "auto"
    agent_type: Literal["dqn", "qrdqn"] = "dqn"
    num_quantiles: int = 64
    n_step: int = 3
    risk_measure: str = "mean"
    prioritized_alpha: float = 0.6
    prioritized_beta: float = 0.4
    prioritized_beta_annealing: float = 0.001
    positive_fraction: float = 0.25
    use_double_dqn: bool = True
    use_guided_exploration: bool = True  # Enable hint-guided exploration (training only)
    exploration_temperature: float = 1.0  # Softmax temperature for hint-based sampling
    gradient_clip: float = 1.0  # Gradient clipping value (prevents explosions)

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.agent_type not in ("dqn", "qrdqn"):
            raise ValueError(f"Unknown agent_type: {self.agent_type}")

@dataclass
class GNNConfig:
    """Configuration for GNN State Encoder.

    Args:
        node_feature_dim: Dimension of raw node features
        gnn_type: Type of GNN ("sage", "gat", or "rmganets")
        embedding_dim: Dimension of output node embeddings
        hidden_channels: Hidden layer dimension
        num_layers: Number of GNN layers
        history_dim: Dimension of history encoding
        dropout: Dropout rate for regularization
        heads: Number of attention heads (GAT only)
        device: Computation device
    """

    node_feature_dim: int
    gnn_type: Literal["sage", "gat", "rmganets", "tgat", "tgn"] = "sage"
    embedding_dim: int = 32
    hidden_channels: int = 64
    num_layers: int = 2
    history_dim: int = 16
    dropout: float = 0.2
    heads: int = 4  # For GAT
    device: str = "auto"
    multi_branch: bool = False
    use_dqn_enhancement: bool = False
    num_classes: int = 2
    time_attributes: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.gnn_type not in ["sage", "gat", "rmganets", "tgat", "tgn"]:
            raise ValueError(
                f"gnn_type must be one of 'sage', 'gat', 'rmganets', 'tgat', or 'tgn', got {self.gnn_type}"
            )
        if self.multi_branch and self.gnn_type != "rmganets":
            raise ValueError("multi_branch can only be enabled when gnn_type='rmganets'")
        if self.time_attributes is not None:
            cleaned = [str(attr).strip() for attr in self.time_attributes if str(attr).strip()]
            self.time_attributes = cleaned or None


@dataclass
class EnvironmentConfig:
    """Configuration for temporal graph governance environment.

    Args:
        max_steps: Maximum steps per episode
        max_neighbors: Maximum neighbors to consider per state
        intrinsic_reward_weight: Weight for intrinsic curiosity reward
        step_penalty: Penalty per step (negative to encourage efficiency)
        fraud_reward: Reward for flagging fraud
        false_alarm_penalty: Penalty for false alarm
    """

    max_steps: int = 50  # INCREASED from 20 for better graph coverage
    max_neighbors: int = 5
    intrinsic_reward_weight: float = 0.1
    step_penalty: float = -0.01
    fraud_reward: float = 10.0
    false_alarm_penalty: float = -10.0


@dataclass
class MultiBranchLossConfig:
    """Configuration for RMGANets multi-branch loss."""

    enabled: bool = False
    variant: Literal["improved", "paper"] = "improved"
    lambda_branch: float = 0.25
    epsilon_dqn: float = 0.0
    beta_reg: float = 0.1
    temporal_decay: float = 0.1
    adaptive_weighting: bool = True
    learning_rate: float = 5e-4
    log_frequency: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainerConfig:
    """Configuration for Training Pipeline.

    Args:
        num_episodes: Total training episodes
        batch_size: Mini-batch size for DQN updates
        eval_frequency: Evaluate every N episodes
        checkpoint_frequency: Save checkpoint every N episodes
        target_update_frequency: Update target network every N episodes
        curriculum_schedule: List of fraud rates for curriculum learning
        max_steps_per_episode: Maximum steps per episode
        num_eval_episodes: Number of episodes for evaluation
    """

    num_episodes: int = 1000
    batch_size: int = 64
    eval_frequency: int = 50
    checkpoint_frequency: int = 100
    target_update_frequency: int = 10
    curriculum_schedule: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.02]
    )
    max_steps_per_episode: int = 20
    num_eval_episodes: int = 20
    multi_branch: MultiBranchLossConfig = field(default_factory=MultiBranchLossConfig)


@dataclass
class ExperimentConfig:
    """Master configuration for a guardrail-driven temporal graph experiment.

    Args:
        name: Experiment name
        dataset_type: Type of dataset ("amlnet" or "elliptic")
        dataset_path: Path to dataset file(s)
        agent: Agent configuration
        gnn: GNN configuration
        environment: Environment configuration
        trainer: Training configuration
        output_dir: Directory for outputs (checkpoints, logs)
        seed: Random seed for reproducibility
        nrows: Number of rows to load (None = all, for testing)
        include_unlabeled: Whether to include unlabeled nodes when supported by the dataset
    """

    name: str
    dataset_type: Literal["amlnet", "elliptic"]
    dataset_path: Path | str
    agent: AgentConfig
    gnn: GNNConfig
    environment: EnvironmentConfig
    trainer: TrainerConfig
    output_dir: Path | str = "outputs"
    seed: int = 42
    nrows: int | None = None
    include_unlabeled: bool | None = None

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filepath: Path | str | None = None) -> None:
        """Save configuration to JSON file.

        Args:
            filepath: Path to save config (default: {output_dir}/config.json)
        """
        if filepath is None:
            filepath = Path(self.output_dir) / "config.json"
        else:
            filepath = Path(filepath)

        # Convert to dict (handle Path objects and nested dataclasses)
        trainer_dict = asdict(self.trainer)
        trainer_dict["multi_branch"] = self.trainer.multi_branch.to_dict()

        config_dict = {
            "name": self.name,
            "dataset_type": self.dataset_type,
            "dataset_path": str(self.dataset_path),
            "include_unlabeled": self.include_unlabeled,
            "agent": asdict(self.agent),
            "gnn": asdict(self.gnn),
            "environment": asdict(self.environment),
            "trainer": trainer_dict,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "nrows": self.nrows,
        }

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Config saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path | str) -> ExperimentConfig:
        """Load configuration from JSON file.

        Args:
            filepath: Path to config file

        Returns:
            ExperimentConfig instance
        """
        with open(filepath) as f:
            config_dict = json.load(f)

        trainer_raw = config_dict["trainer"]
        multi_branch_cfg = trainer_raw.get("multi_branch", {})
        trainer_raw["multi_branch"] = MultiBranchLossConfig(**multi_branch_cfg)

        return cls(
            name=config_dict["name"],
            dataset_type=config_dict["dataset_type"],
            dataset_path=config_dict["dataset_path"],
            include_unlabeled=config_dict.get("include_unlabeled"),
            agent=AgentConfig(**config_dict["agent"]),
            gnn=GNNConfig(**config_dict["gnn"]),
            environment=EnvironmentConfig(**config_dict["environment"]),
            trainer=TrainerConfig(**trainer_raw),
            output_dir=config_dict["output_dir"],
            seed=config_dict["seed"],
            nrows=config_dict.get("nrows"),
        )


# Preset configurations for common experiments

def get_quick_test_config() -> ExperimentConfig:
    """Quick test configuration (small dataset, few episodes)."""
    return ExperimentConfig(
        name="quick_test",
        dataset_type="amlnet",
        dataset_path="data/amlnet/AMLNet_August_2025.csv",
        agent=AgentConfig(
            state_dim=48,
            action_dim=6,
            hidden_dims=[64, 64],
            epsilon_end=0.2,
            epsilon_decay=0.9995,
        ),
        gnn=GNNConfig(node_feature_dim=20, embedding_dim=16),
        environment=EnvironmentConfig(max_steps=10, intrinsic_reward_weight=0.5),
        trainer=TrainerConfig(
            num_episodes=100,
            curriculum_schedule=[1.0, 0.5, 0.1],
            eval_frequency=20
        ),
        output_dir="outputs/quick_test",
        nrows=10000,  # Small subset
    )


def get_full_amlnet_config() -> ExperimentConfig:
    """Full AMLNet training configuration."""
    return ExperimentConfig(
        name="amlnet_full",
        dataset_type="amlnet",
        dataset_path="data/amlnet/AMLNet_August_2025.csv",
        agent=AgentConfig(
            state_dim=48,  # 32 (GNN) + 16 (history)
            action_dim=6,  # 5 neighbors + FLAG
            learning_rate=1e-3,
            use_prioritized_replay=True,
            epsilon_end=0.2,
            epsilon_decay=0.999,
        ),
        gnn=GNNConfig(
            node_feature_dim=20,
            gnn_type="sage",
            embedding_dim=32,
        ),
        environment=EnvironmentConfig(
            max_steps=20,
            intrinsic_reward_weight=0.5,
        ),
        trainer=TrainerConfig(
            num_episodes=1000,
            batch_size=64,
            curriculum_schedule=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.02],
        ),
        output_dir="outputs/amlnet_full",
    )


def get_elliptic_config() -> ExperimentConfig:
    """Elliptic dataset configuration."""
    return ExperimentConfig(
        name="elliptic_validation",
        dataset_type="elliptic",
        dataset_path="data/elliptic",
        agent=AgentConfig(
            state_dim=48,
            action_dim=6,
            learning_rate=1e-3,
        ),
        gnn=GNNConfig(
            node_feature_dim=172,  # 166 raw features + 6 engineered features
            gnn_type="gat",  # Use attention for Elliptic
            embedding_dim=32,
        ),
        environment=EnvironmentConfig(max_steps=15),
        trainer=TrainerConfig(
            num_episodes=500,
            curriculum_schedule=[0.5, 0.2, 0.02],  # Elliptic already has ~2% fraud
        ),
        output_dir="outputs/elliptic",
    )


def get_ablation_configs() -> dict[str, ExperimentConfig]:
    """Generate configurations for ablation study."""
    base = get_full_amlnet_config()

    configs = {
        "full": base,
        "no_prioritized_replay": ExperimentConfig(
            **{**base.__dict__, "agent": AgentConfig(
                **{**base.agent.__dict__, "use_prioritized_replay": False}
            )}
        ),
        "no_intrinsic_reward": ExperimentConfig(
            **{**base.__dict__, "environment": EnvironmentConfig(
                **{**base.environment.__dict__, "intrinsic_reward_weight": 0.0}
            )}
        ),
        "gat_instead_of_sage": ExperimentConfig(
            **{**base.__dict__, "gnn": GNNConfig(
                **{**base.gnn.__dict__, "gnn_type": "gat"}
            )}
        ),
    }

    return configs
