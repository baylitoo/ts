"""
Checkpoint management for model training.

Provides utilities for saving/loading model checkpoints, training state,
and experiment configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """
    Manages model checkpoints and training state.

    Handles saving/loading of:
    - Model weights (Q-network, target network)
    - Optimizer state
    - Training statistics (episode, epsilon, losses)
    - Configuration
    """

    def __init__(self, checkpoint_dir: Path | str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        agent: Any,
        episode: int,
        metrics: dict[str, Any] | None = None,
        filename: str | None = None
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            agent: DQN agent to save
            episode: Current episode number
            metrics: Optional training metrics
            filename: Optional custom filename

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_ep{episode}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "episode": episode,
            "q_network_state_dict": agent.q_network.state_dict(),
            "target_network_state_dict": agent.target_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
        }

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(
        self,
        agent: Any,
        checkpoint_path: Path | str
    ) -> dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            agent: DQN agent to load into
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with episode and metrics
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=agent.device)

        # Load model states
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]

        return {
            "episode": checkpoint["episode"],
            "metrics": checkpoint.get("metrics", {}),
        }

    def save_final_model(
        self,
        agent: Any,
        filename: str = "final_model.pt"
    ) -> Path:
        """
        Save final trained model (Q-network only).

        Args:
            agent: DQN agent
            filename: Filename for final model

        Returns:
            Path to saved model
        """
        model_path = self.checkpoint_dir / filename

        torch.save({
            "q_network_state_dict": agent.q_network.state_dict(),
        }, model_path)

        return model_path

    def load_final_model(
        self,
        agent: Any,
        model_path: Path | str | None = None
    ) -> None:
        """
        Load final trained model.

        Args:
            agent: DQN agent to load into
            model_path: Path to model file (default: final_model.pt)
        """
        if model_path is None:
            model_path = self.checkpoint_dir / "final_model.pt"

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=agent.device)
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])

    def list_checkpoints(self) -> list[Path]:
        """
        List all checkpoint files.

        Returns:
            List of checkpoint paths sorted by episode number
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        checkpoints.sort(key=lambda p: int(p.stem.split("ep")[1]))
        return checkpoints

    def get_latest_checkpoint(self) -> Path | None:
        """
        Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def save_config(
        self,
        config: Any,
        filename: str = "config.json"
    ) -> Path:
        """
        Save experiment configuration.

        Args:
            config: ExperimentConfig object
            filename: Filename for config

        Returns:
            Path to saved config
        """
        config_path = self.checkpoint_dir / filename
        config.save(config_path)
        return config_path

    def load_config(
        self,
        filename: str = "config.json"
    ) -> dict[str, Any]:
        """
        Load experiment configuration.

        Args:
            filename: Filename of config

        Returns:
            Configuration dictionary
        """
        config_path = self.checkpoint_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            return json.load(f)

    def cleanup_old_checkpoints(
        self,
        keep_last_n: int = 5,
        keep_every_n: int = 100
    ) -> None:
        """
        Remove old checkpoints to save disk space.

        Keeps:
        - Last N checkpoints
        - Every Nth checkpoint (for long-term tracking)

        Args:
            keep_last_n: Number of recent checkpoints to keep
            keep_every_n: Keep every Nth checkpoint
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last_n:
            return

        # Determine which to keep
        to_keep = set()

        # Keep last N
        to_keep.update(checkpoints[-keep_last_n:])

        # Keep every Nth
        for cp in checkpoints:
            episode = int(cp.stem.split("ep")[1])
            if episode % keep_every_n == 0:
                to_keep.add(cp)

        # Remove others
        for cp in checkpoints:
            if cp not in to_keep:
                cp.unlink()
