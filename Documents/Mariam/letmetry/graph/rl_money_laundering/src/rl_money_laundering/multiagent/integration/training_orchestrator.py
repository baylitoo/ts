"""
Training orchestrator for multiagent integration.

Coordinates training using both the existing RL framework and the
new multiagent components without modifying original code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .config import IntegrationConfig
from .environment_adapter import EnvironmentAdapter, EpisodeCollector
from .encoder_bridge import EncoderBridge, EpisodeTextBuilder
from .reward_integrator import RewardIntegrator

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for RL agent interface."""

    def select_action(
        self,
        state: Any,
        epsilon: Optional[float] = None,
        hints: Optional[Any] = None,
    ) -> tuple[int, float]: ...

    def store_transition(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def train_step(self) -> Optional[Dict[str, float]]: ...

    def decay_epsilon(self) -> None: ...

    def update_target_network(self) -> None: ...


class TrainerProtocol(Protocol):
    """Protocol for existing trainer interface."""

    def train(self, num_episodes: int) -> Dict[str, Any]: ...


class JudgeCallbackProtocol(Protocol):
    """Protocol for judge training callbacks."""

    def on_episode_end(
        self,
        episode: int,
        episode_data: Any,
        labels: Any,
    ) -> Dict[str, Any]: ...

    def should_update(self, episode: int) -> bool: ...


@dataclass
class TrainingState:
    """Current state of training for checkpointing and monitoring."""
    episode: int = 0
    total_steps: int = 0
    best_f1: float = 0.0
    best_episode: int = 0

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_f1s: List[float] = field(default_factory=list)
    judge_rewards: List[float] = field(default_factory=list)

    training_start_time: float = field(default_factory=time.time)
    last_checkpoint_episode: int = 0

    safety_stops: int = 0
    drift_alarms: int = 0

    def update(
        self,
        reward: float,
        length: int,
        f1: float,
        judge_reward: float = 0.0,
    ) -> None:
        """Update state with episode results."""
        self.episode += 1
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_f1s.append(f1)
        self.judge_rewards.append(judge_reward)
        self.total_steps += length

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_episode = self.episode

    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """Get statistics over recent episodes."""
        if not self.episode_rewards:
            return {}

        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_f1s = self.episode_f1s[-window:]
        recent_judge = self.judge_rewards[-window:]

        return {
            "mean_reward": np.mean(recent_rewards),
            "mean_length": np.mean(recent_lengths),
            "mean_f1": np.mean(recent_f1s),
            "mean_judge_reward": np.mean(recent_judge) if recent_judge else 0.0,
            "std_reward": np.std(recent_rewards),
            "best_f1": self.best_f1,
            "best_episode": self.best_episode,
            "total_episodes": self.episode,
            "total_steps": self.total_steps,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for checkpointing."""
        return {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "best_f1": self.best_f1,
            "best_episode": self.best_episode,
            "safety_stops": self.safety_stops,
            "drift_alarms": self.drift_alarms,
            "recent_stats": self.get_recent_stats(),
        }


class TrainingOrchestrator:
    """Orchestrates training with multiagent integration.

    This orchestrator wraps the existing training loop, adding:
    1. Episode collection for judge evaluation
    2. Reward integration with learned judge
    3. Two-timescale judge updates
    4. Safety monitoring and early stopping
    5. Detailed logging and checkpointing

    The original training behavior is preserved when judge is disabled.
    """

    def __init__(
        self,
        env: Any,
        agent: AgentProtocol,
        state_encoder: Any,
        config: IntegrationConfig,
        judge_model: Optional[Any] = None,
        judge_callback: Optional[JudgeCallbackProtocol] = None,
        graph: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        log_frequency: int = 100,
    ) -> None:
        """Initialize training orchestrator.

        Args:
            env: Environment (will be wrapped with adapter).
            agent: RL agent.
            state_encoder: GNN state encoder.
            config: Integration configuration.
            judge_model: Optional judge model.
            judge_callback: Optional callback for judge updates.
            graph: NetworkX graph for state encoding.
            checkpoint_dir: Directory for saving checkpoints.
            log_frequency: Episodes between logging.
        """
        self.config = config
        self._agent = agent
        self._state_encoder = state_encoder
        self._graph = graph
        self._checkpoint_dir = checkpoint_dir
        self._log_frequency = log_frequency

        self._collector = EpisodeCollector(
            buffer_size=config.episode_buffer_size,
            anonymize_ids=True,
        )
        self._env_adapter = EnvironmentAdapter(
            env=env,
            collector=self._collector,
            enable_collection=config.judge_config.enable_judge,
        )

        self._encoder_bridge = EncoderBridge(
            state_encoder=state_encoder,
            judge_model=judge_model,
            cache_embeddings=True,
        )

        self._text_builder = EpisodeTextBuilder()

        self._reward_integrator = RewardIntegrator(
            config=config,
            judge_model=judge_model,
            episode_text_builder=self._text_builder.build,
        )

        self._judge_callback = judge_callback

        self._state = TrainingState()
        self._curriculum_stage = 0
        self._fraud_rate = 1.0

    @property
    def state(self) -> TrainingState:
        """Access current training state."""
        return self._state

    def train(
        self,
        num_episodes: int,
        curriculum_schedule: Optional[List[float]] = None,
        eval_frequency: int = 100,
        checkpoint_frequency: int = 500,
        early_stop_patience: int = 1000,
    ) -> Dict[str, Any]:
        """Run training loop with multiagent integration.

        Args:
            num_episodes: Total episodes to train.
            curriculum_schedule: Fraud rate schedule (e.g., [1.0, 0.5, 0.1]).
            eval_frequency: Episodes between evaluations.
            checkpoint_frequency: Episodes between checkpoints.
            early_stop_patience: Episodes without improvement before stopping.

        Returns:
            Training results dictionary.
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Judge enabled: {self.config.judge_config.enable_judge}")

        if curriculum_schedule is None:
            curriculum_schedule = [1.0, 0.5, 0.2, 0.1]

        episodes_per_stage = num_episodes // len(curriculum_schedule)
        episodes_without_improvement = 0

        try:
            for episode in range(num_episodes):
                stage_idx = min(
                    episode // episodes_per_stage,
                    len(curriculum_schedule) - 1,
                )
                self._fraud_rate = curriculum_schedule[stage_idx]

                episode_result = self._run_episode(episode)

                self._state.update(
                    reward=episode_result["combined_reward"],
                    length=episode_result["length"],
                    f1=episode_result["f1"],
                    judge_reward=episode_result.get("judge_reward", 0.0),
                )

                if episode_result["f1"] > self._state.best_f1 - 0.01:
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1

                if self._judge_callback and self.config.judge_config.enable_judge:
                    if self._judge_callback.should_update(episode):
                        episode_data = self._collector.get_recent_episodes(100)
                        if episode_data:
                            self._judge_callback.on_episode_end(
                                episode=episode,
                                episode_data=episode_data,
                                labels=None,
                            )

                if episode > 0 and episode % self._log_frequency == 0:
                    self._log_progress(episode)

                if episode > 0 and episode % eval_frequency == 0:
                    self._evaluate(episode)

                if (
                    self._checkpoint_dir
                    and episode > 0
                    and episode % checkpoint_frequency == 0
                ):
                    self._save_checkpoint(episode)

                should_stop, reason = self._reward_integrator.should_stop_training()
                if should_stop:
                    logger.warning(f"Safety stop triggered: {reason}")
                    self._state.safety_stops += 1
                    break

                if episodes_without_improvement >= early_stop_patience:
                    logger.info(
                        f"Early stopping: no improvement for {early_stop_patience} episodes"
                    )
                    break

                self._agent.decay_epsilon()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        return self._compile_results()

    def _run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single training episode.

        Returns:
            Episode results dictionary.
        """
        start_node = self._sample_start_node()

        obs, info = self._env_adapter.reset(start_node=start_node)

        state = self._encode_state(obs)
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            hints = self._get_neighbor_hints()
            action, confidence = self._agent.select_action(state, hints=hints)

            next_obs, reward, terminated, truncated, info = self._env_adapter.step(
                action, confidence=confidence
            )
            done = terminated or truncated

            step_reward = self._reward_integrator.compute_step_reward(reward, info)

            next_state = self._encode_state(next_obs) if not done else state

            self._agent.store_transition(
                state=state,
                action=action,
                reward=step_reward,
                next_state=next_state,
                done=done,
                info=info,
            )

            self._agent.train_step()

            state = next_state
            episode_reward += reward
            episode_length += 1

        episode_data = self._env_adapter.get_episode_data()

        episode_text = None
        if episode_data and self.config.judge_config.enable_judge:
            episode_text = self._text_builder.build(episode_data)

        combined_reward, metrics = self._reward_integrator.compute_episode_reward(
            original_episode_reward=episode_reward,
            episode_data=episode_data,
            episode_text=episode_text,
        )

        if episode_num % 10 == 0:
            self._agent.update_target_network()

        return {
            "original_reward": episode_reward,
            "combined_reward": combined_reward,
            "judge_reward": metrics.judge_reward,
            "length": episode_length,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "metrics": metrics,
        }

    def _encode_state(self, obs: Any) -> np.ndarray:
        """Encode observation to state vector."""
        if self._state_encoder is None:
            if isinstance(obs, np.ndarray):
                return obs
            return np.array(obs)

        try:
            current_node = getattr(self._env_adapter.env, "current_node", None)
            visited = getattr(self._env_adapter.env, "visited_nodes", set())
            visited_edges = getattr(self._env_adapter.env, "visited_edges", set())

            output = self._encoder_bridge.encode_state(
                graph=self._graph,
                node_id=current_node,
                visited_nodes=visited,
                visited_edges=visited_edges,
            )
            return output.full_state
        except Exception as e:
            logger.warning(f"State encoding failed: {e}, using raw observation")
            if isinstance(obs, np.ndarray):
                return obs
            return np.array(obs)

    def _sample_start_node(self) -> Optional[Any]:
        """Sample starting node based on curriculum."""
        if self._graph is None:
            return None

        if np.random.random() < self._fraud_rate:
            fraud_nodes = [
                n for n, d in self._graph.nodes(data=True)
                if d.get("is_fraud", False) or d.get("isFraud", False)
            ]
            if fraud_nodes:
                return np.random.choice(fraud_nodes)

        nodes = list(self._graph.nodes())
        return np.random.choice(nodes) if nodes else None

    def _get_neighbor_hints(self) -> Optional[np.ndarray]:
        """Get risk hints for neighbors (guided exploration)."""
        return None

    def _log_progress(self, episode: int) -> None:
        """Log training progress."""
        stats = self._state.get_recent_stats()
        integrator_stats = self._reward_integrator.get_stats()

        logger.info(
            f"Episode {episode}: "
            f"reward={stats.get('mean_reward', 0):.3f}, "
            f"F1={stats.get('mean_f1', 0):.3f}, "
            f"judge_weight={integrator_stats.get('current_judge_weight', 0):.3f}, "
            f"correlation={integrator_stats.get('correlation', 1):.3f}"
        )

    def _evaluate(self, episode: int) -> Dict[str, float]:
        """Run evaluation episodes."""
        logger.info(f"Evaluation at episode {episode}")

        eval_rewards = []

        self._env_adapter.disable_collection()

        for _ in range(10):
            obs, _ = self._env_adapter.reset()
            state = self._encode_state(obs)
            episode_reward = 0.0
            done = False

            while not done:
                action, _ = self._agent.select_action(state, epsilon=0.0)
                obs, reward, terminated, truncated, info = self._env_adapter.step(action)
                done = terminated or truncated
                state = self._encode_state(obs) if not done else state
                episode_reward += reward

            eval_rewards.append(episode_reward)

        self._env_adapter.enable_collection()

        results = {
            "mean_eval_reward": np.mean(eval_rewards),
            "std_eval_reward": np.std(eval_rewards),
        }

        logger.info(f"Eval results: reward={results['mean_eval_reward']:.3f}")
        return results

    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        if self._checkpoint_dir is None:
            return

        checkpoint_path = self._checkpoint_dir / f"checkpoint_ep{episode}.pt"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "training_state": self._state.to_dict(),
            "integrator_stats": self._reward_integrator.get_stats(),
            "config": self.config.to_dict(),
        }

        import json
        with open(checkpoint_path.with_suffix(".json"), "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        self._state.last_checkpoint_episode = episode

    def _compile_results(self) -> Dict[str, Any]:
        """Compile final training results."""
        training_time = time.time() - self._state.training_start_time

        return {
            "training_state": self._state.to_dict(),
            "final_stats": self._state.get_recent_stats(),
            "integrator_stats": self._reward_integrator.get_stats(),
            "encoder_stats": self._encoder_bridge.get_cache_stats(),
            "training_time_seconds": training_time,
            "episodes_completed": self._state.episode,
            "best_f1": self._state.best_f1,
            "best_episode": self._state.best_episode,
        }


def create_integrated_trainer(
    env: Any,
    agent: Any,
    state_encoder: Any,
    graph: Any,
    judge_model: Optional[Any] = None,
    config: Optional[IntegrationConfig] = None,
    checkpoint_dir: Optional[str] = None,
) -> TrainingOrchestrator:
    """Factory function to create integrated trainer.

    Convenience function that sets up all integration components.

    Args:
        env: AMLDetectionEnv instance.
        agent: DQN or QR-DQN agent.
        state_encoder: GNN state encoder.
        graph: NetworkX transaction graph.
        judge_model: Optional JudgeModel instance.
        config: Integration config (uses defaults if None).
        checkpoint_dir: Optional checkpoint directory.

    Returns:
        Configured TrainingOrchestrator.
    """
    if config is None:
        config = IntegrationConfig.default_conservative()

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None

    orchestrator = TrainingOrchestrator(
        env=env,
        agent=agent,
        state_encoder=state_encoder,
        config=config,
        judge_model=judge_model,
        graph=graph,
        checkpoint_dir=checkpoint_path,
    )

    return orchestrator
