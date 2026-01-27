"""
Hyperparameter Tuning with Ray Tune

Automated hyperparameter search for GNN-based fraud detection agent.
Supports both standard DQN training and multiagent mode with judge integration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rl_money_laundering.rllib_integration import (
    register_aml_env,
    GNNDQNModule,
    FraudAwareReplayBuffer,
    create_fraud_callbacks,
)
from rl_money_laundering.features.network_pyg import NetworkFeatureExtractor
from rl_money_laundering.gnn_encoder import StateEncoder
from rl_money_laundering.data.elliptic import load_elliptic_data


# =============================================================================
# SEARCH SPACE DEFINITIONS
# =============================================================================

def create_standard_search_space(search_type: str = "quick"):
    """
    Define hyperparameter search space for standard DQN training.

    Args:
        search_type: "quick" (grid search), "medium" (random), or "large" (Bayesian)

    Returns:
        Dict with tune.choice/tune.uniform/etc. for each hyperparameter
    """
    if search_type == "quick":
        # Grid search over key parameters (fastest)
        return {
            # Learning parameters
            "lr": tune.grid_search([1e-4, 5e-4, 1e-3]),
            "gamma": tune.grid_search([0.95, 0.99]),

            # Network architecture
            "embedding_dim": tune.grid_search([32, 64]),
            "gnn_type": tune.grid_search(["rmganets", "gat"]),

            # Training
            "batch_size": tune.grid_search([64, 128]),
            "n_step": tune.grid_search([1, 3]),

            # Phase 2
            "fraud_boost_factor": tune.grid_search([2.0, 3.0]),
        }

    elif search_type == "medium":
        # Random search with continuous ranges
        return {
            # Learning parameters
            "lr": tune.loguniform(1e-5, 1e-2),
            "gamma": tune.uniform(0.95, 0.999),

            # Network architecture
            "embedding_dim": tune.choice([32, 64, 96, 128]),
            "history_dim": tune.choice([8, 16, 24, 32]),
            "hidden_dims": tune.choice([
                [128, 128, 64],
                [256, 128, 64],
                [128, 64],
                [256, 256, 128],
            ]),
            "gnn_type": tune.choice(["rmganets", "gat", "sage"]),

            # Training
            "batch_size": tune.choice([32, 64, 128, 256]),
            "buffer_capacity": tune.choice([50000, 100000, 200000]),
            "per_alpha": tune.uniform(0.4, 0.8),
            "per_beta": tune.uniform(0.3, 0.6),
            "n_step": tune.choice([1, 2, 3, 5]),
            "target_update_freq": tune.choice([250, 500, 1000]),

            # Phase 2
            "fraud_boost_factor": tune.uniform(1.5, 5.0),
        }

    else:  # large
        # Full search space for Bayesian optimization
        return {
            # Learning parameters
            "lr": tune.loguniform(1e-5, 1e-2),
            "gamma": tune.uniform(0.90, 0.999),
            "lr_schedule": tune.choice([None, "linear", "exponential"]),

            # Network architecture
            "embedding_dim": tune.choice([16, 32, 48, 64, 96, 128]),
            "history_dim": tune.choice([8, 12, 16, 24, 32]),
            "hidden_dims": tune.choice([
                [64, 64],
                [128, 64],
                [128, 128, 64],
                [256, 128, 64],
                [256, 256, 128],
                [512, 256, 128],
            ]),
            "dropout": tune.uniform(0.1, 0.5),
            "gnn_type": tune.choice(["rmganets", "gat", "sage", "tgat"]),
            "multi_branch": tune.choice([True, False]),

            # Training
            "batch_size": tune.choice([16, 32, 64, 128, 256]),
            "buffer_capacity": tune.lograndint(10000, 500000),
            "per_alpha": tune.uniform(0.3, 0.9),
            "per_beta": tune.uniform(0.2, 0.7),
            "n_step": tune.choice([1, 2, 3, 5, 10]),
            "target_update_freq": tune.lograndint(100, 2000),

            # Exploration
            "epsilon_timesteps": tune.lograndint(1000, 50000),
            "final_epsilon": tune.loguniform(0.01, 0.2),

            # Phase 2
            "fraud_boost_factor": tune.uniform(1.0, 10.0),
            "use_fraud_buffer": tune.choice([True, False]),
        }


def create_multiagent_search_space(search_type: str = "quick"):
    """
    Define hyperparameter search space for multiagent training with judge.

    Includes all standard parameters plus judge integration parameters.

    Args:
        search_type: "quick", "medium", or "large"

    Returns:
        Dict with tune parameters
    """
    # Start with standard search space
    space = create_standard_search_space(search_type)

    if search_type == "quick":
        # Judge integration parameters (grid search)
        space.update({
            # Judge weight annealing
            "judge_weight_start": tune.grid_search([0.05, 0.1]),
            "judge_weight_end": tune.grid_search([0.2, 0.3]),
            "judge_warmup_episodes": tune.grid_search([2000, 5000]),

            # Safety parameters
            "uncertainty_penalty": tune.grid_search([0.1, 0.15]),
            "judge_clip_max": tune.grid_search([0.3, 0.5]),

            # Hard metric anchoring
            "hard_metric_weight": tune.grid_search([0.6, 0.7]),
        })

    elif search_type == "medium":
        space.update({
            # Judge weight annealing
            "judge_weight_start": tune.uniform(0.02, 0.15),
            "judge_weight_end": tune.uniform(0.15, 0.4),
            "judge_warmup_episodes": tune.choice([1000, 2000, 5000, 10000]),

            # Safety parameters
            "uncertainty_penalty": tune.uniform(0.05, 0.2),
            "judge_clip_max": tune.uniform(0.2, 0.7),

            # Hard metric anchoring
            "hard_metric_weight": tune.uniform(0.5, 0.8),
            "precision_weight": tune.uniform(0.15, 0.35),
            "recall_weight": tune.uniform(0.2, 0.4),
            "f1_weight": tune.uniform(0.35, 0.6),

            # Safety monitoring
            "drift_correlation_threshold": tune.uniform(0.2, 0.5),
            "hard_metric_floor": tune.uniform(0.05, 0.2),
        })

    else:  # large
        space.update({
            # Judge weight annealing (full range)
            "judge_weight_start": tune.uniform(0.0, 0.2),
            "judge_weight_end": tune.uniform(0.1, 0.5),
            "judge_warmup_episodes": tune.lograndint(500, 20000),

            # Safety parameters
            "uncertainty_penalty": tune.uniform(0.0, 0.3),
            "judge_clip_max": tune.uniform(0.1, 1.0),

            # Hard metric anchoring (will be normalized)
            "hard_metric_weight": tune.uniform(0.4, 0.9),
            "precision_weight": tune.uniform(0.1, 0.4),
            "recall_weight": tune.uniform(0.2, 0.5),
            "f1_weight": tune.uniform(0.3, 0.7),

            # Safety monitoring
            "drift_correlation_threshold": tune.uniform(0.1, 0.6),
            "hard_metric_floor": tune.uniform(0.0, 0.25),

            # Judge update settings
            "judge_update_frequency": tune.choice([50, 100, 200, 500]),
            "judge_freeze_after": tune.choice([2000, 5000, 10000, -1]),  # -1 = never
        })

    return space


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_base_config(args, env_config):
    """
    Create base DQN config (parameters NOT being tuned).

    Args:
        args: Command line arguments
        env_config: Environment configuration

    Returns:
        DQNConfig with fixed parameters
    """
    register_aml_env("aml_detection")

    config = (
        DQNConfig()
        .environment("aml_detection", env_config=env_config)
        .framework("torch")
        .callbacks(create_fraud_callbacks(detailed=False))
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length=50,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1 if args.num_gpus > 0 else 0,
        )
        .resources(
            num_cpus_per_env_runner=1,
        )
        .reporting(
            min_sample_timesteps_per_iteration=1000,
        )
        .debugging(seed=args.seed)
    )

    return config


def standard_training_function(config_dict, base_config, max_iterations=50):
    """
    Training function for standard Ray Tune trials (RLlib-based).

    Args:
        config_dict: Hyperparameters from Tune search
        base_config: Base DQN configuration
        max_iterations: Maximum training iterations

    Returns:
        Metric results for Tune
    """
    # Extract hyperparameters
    lr = config_dict["lr"]
    gamma = config_dict["gamma"]
    batch_size = config_dict["batch_size"]
    embedding_dim = config_dict.get("embedding_dim", 32)
    gnn_type = config_dict.get("gnn_type", "rmganets")
    fraud_boost = config_dict.get("fraud_boost_factor", 2.0)
    use_fraud_buffer = config_dict.get("use_fraud_buffer", True)

    # Update base config with tuned parameters
    config = (
        base_config
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=GNNDQNModule,
                model_config_dict={
                    "node_feature_dim": config_dict.get("node_feature_dim", 48),
                    "gnn_type": gnn_type,
                    "embedding_dim": embedding_dim,
                    "history_dim": config_dict.get("history_dim", 16),
                    "multi_branch": config_dict.get("multi_branch", False),
                    "hidden_dims": config_dict.get("hidden_dims", [128, 128, 64]),
                }
            )
        )
        .training(
            lr=lr,
            gamma=gamma,
            train_batch_size_per_learner=batch_size,
            replay_buffer_config={
                "type": FraudAwareReplayBuffer if use_fraud_buffer else "PrioritizedEpisodeReplayBuffer",
                "capacity": config_dict.get("buffer_capacity", 100000),
                "alpha": config_dict.get("per_alpha", 0.6),
                "beta": config_dict.get("per_beta", 0.4),
                "fraud_boost_factor": fraud_boost,
            },
            double_q=True,
            dueling=True,
            n_step=config_dict.get("n_step", 3),
            target_network_update_freq=config_dict.get("target_update_freq", 500),
        )
    )

    # Build and train
    algo = config.build()

    try:
        for iteration in range(max_iterations):
            result = algo.train()

            # Report metrics to Tune
            tune.report(
                iteration=iteration,
                episode_reward_mean=result.get("env_runners", {}).get("episode_reward_mean", 0),
                episode_len_mean=result.get("env_runners", {}).get("episode_len_mean", 0),
                fraud_f1_score=result.get("fraud_f1_score", 0),
                fraud_edges_per_episode=result.get("fraud_edges_per_episode", 0),
                timesteps_total=result.get("num_env_steps_sampled_lifetime", 0),
            )
    finally:
        algo.stop()


def multiagent_training_function(config_dict, env_config, graph, illicit_nodes, max_episodes=5000):
    """
    Training function for multiagent trials with judge integration.

    Uses TrainingOrchestrator for two-timescale training with judge rewards.

    Args:
        config_dict: Hyperparameters from Tune search
        env_config: Environment configuration
        graph: NetworkX graph
        illicit_nodes: Set of illicit node IDs
        max_episodes: Maximum training episodes
    """
    from rl_money_laundering.multiagent.integration import (
        IntegrationConfig,
        JudgeIntegrationConfig,
        TrainingOrchestrator,
    )
    from rl_money_laundering.envs import AMLDetectionEnv
    from rl_money_laundering.agents.dqn_agent import DQNAgent
    import numpy as np

    # Build integration config from hyperparameters
    judge_config = JudgeIntegrationConfig(
        enable_judge=True,
        judge_weight_start=config_dict.get("judge_weight_start", 0.1),
        judge_weight_end=config_dict.get("judge_weight_end", 0.3),
        judge_warmup_episodes=config_dict.get("judge_warmup_episodes", 5000),
        judge_clip_max=config_dict.get("judge_clip_max", 0.5),
        uncertainty_penalty=config_dict.get("uncertainty_penalty", 0.1),
        update_frequency=config_dict.get("judge_update_frequency", 100),
        freeze_after_episodes=config_dict.get("judge_freeze_after", 5000),
    )

    # Normalize metric weights if provided
    precision_w = config_dict.get("precision_weight", 0.2)
    recall_w = config_dict.get("recall_weight", 0.3)
    f1_w = config_dict.get("f1_weight", 0.5)
    total_w = precision_w + recall_w + f1_w
    precision_w, recall_w, f1_w = precision_w/total_w, recall_w/total_w, f1_w/total_w

    integration_config = IntegrationConfig(
        judge_config=judge_config,
        hard_metric_weight=config_dict.get("hard_metric_weight", 0.7),
        precision_weight=precision_w,
        recall_weight=recall_w,
        f1_weight=f1_w,
        drift_correlation_threshold=config_dict.get("drift_correlation_threshold", 0.3),
        hard_metric_floor=config_dict.get("hard_metric_floor", 0.1),
        safety_monitoring=True,
    )

    # Create environment
    env = AMLDetectionEnv(
        graph=graph,
        start_node=env_config.get("start_node"),
        node_feature_extractor=env_config.get("node_feature_extractor"),
        gnn_encoder=env_config.get("gnn_encoder"),
        max_steps=env_config.get("max_steps", 50),
    )

    # Create state encoder
    gnn_encoder = StateEncoder(
        node_feature_dim=config_dict.get("node_feature_dim", 166),
        gnn_type=config_dict.get("gnn_type", "rmganets"),
        embedding_dim=config_dict.get("embedding_dim", 32),
        history_dim=config_dict.get("history_dim", 16),
        device="cpu",
    )

    # Create agent
    state_dim = config_dict.get("embedding_dim", 32) + config_dict.get("history_dim", 16)
    action_dim = env_config.get("max_neighbors", 5) + 1  # +1 for flag action

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config_dict["lr"],
        gamma=config_dict["gamma"],
        batch_size=config_dict["batch_size"],
        buffer_size=config_dict.get("buffer_capacity", 100000),
        hidden_dims=config_dict.get("hidden_dims", [128, 128, 64]),
    )

    # Create orchestrator (no judge model for now - using hard metrics only)
    orchestrator = TrainingOrchestrator(
        env=env,
        agent=agent,
        state_encoder=gnn_encoder,
        config=integration_config,
        judge_model=None,  # Would load from checkpoint if available
        graph=graph,
        log_frequency=100,
    )

    # Training loop with reporting
    eval_frequency = max(100, max_episodes // 50)
    best_f1 = 0.0

    try:
        for episode in range(max_episodes):
            # Run single episode via orchestrator's internal method
            result = orchestrator._run_episode(episode)

            # Update orchestrator state
            orchestrator._state.update(
                reward=result["combined_reward"],
                length=result["length"],
                f1=result["f1"],
                judge_reward=result.get("judge_reward", 0.0),
            )

            # Decay exploration
            agent.decay_epsilon()

            # Report periodically
            if episode > 0 and episode % eval_frequency == 0:
                stats = orchestrator._state.get_recent_stats()
                integrator_stats = orchestrator._reward_integrator.get_stats()

                current_f1 = stats.get("mean_f1", 0)
                if current_f1 > best_f1:
                    best_f1 = current_f1

                tune.report(
                    episode=episode,
                    episode_reward_mean=stats.get("mean_reward", 0),
                    episode_len_mean=stats.get("mean_length", 0),
                    fraud_f1_score=current_f1,
                    best_f1=best_f1,
                    precision=result.get("precision", 0),
                    recall=result.get("recall", 0),
                    judge_weight=integrator_stats.get("current_judge_weight", 0),
                    judge_correlation=integrator_stats.get("correlation", 1),
                    floor_violations=integrator_stats.get("floor_violations", 0),
                )

            # Check safety stops
            should_stop, reason = orchestrator._reward_integrator.should_stop_training()
            if should_stop:
                print(f"Safety stop: {reason}")
                break

    except Exception as e:
        print(f"Training error: {e}")
        raise


# =============================================================================
# MAIN TUNING RUNNER
# =============================================================================

def run_tune(args):
    """
    Run hyperparameter tuning with Ray Tune.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True
    )

    try:
        # Load data
        print("Loading dataset...")
        graph, illicit_nodes = load_elliptic_data(
            node_features_path=args.node_features,
            edges_path=args.edges,
            classes_path=args.classes,
            include_unknown=args.include_unknown
        )

        # Create environment config
        node_feature_extractor = NetworkFeatureExtractor(
            graph=graph,
            node_feature_dim=args.node_feature_dim,
            categorical_features=[]
        )

        gnn_encoder = StateEncoder(
            node_feature_dim=args.node_feature_dim,
            gnn_type="rmganets",  # Will be overridden by Tune
            embedding_dim=32,
            history_dim=16,
            device="cpu"
        )

        import random
        start_node = random.choice(list(illicit_nodes))

        env_config = {
            "graph": graph,
            "start_node": start_node,
            "node_feature_extractor": node_feature_extractor,
            "gnn_encoder": gnn_encoder,
            "max_steps": 50,
            "max_nodes": 30,
            "max_edges": 200,
            "max_neighbors": 5,
        }

        # Define search space based on mode
        if args.mode == "multiagent":
            search_space = create_multiagent_search_space(args.search_type)
            print(f"Mode: MULTIAGENT (with judge integration)")
        else:
            search_space = create_standard_search_space(args.search_type)
            print(f"Mode: STANDARD (RLlib DQN)")

        # Add fixed parameters
        search_space["node_feature_dim"] = args.node_feature_dim

        # Select search algorithm
        if args.search_algo == "optuna":
            search_alg = OptunaSearch(
                metric="fraud_f1_score",
                mode="max"
            )
            print("Using Optuna (Tree-structured Parzen Estimator)")
        elif args.search_algo == "hyperopt":
            search_alg = HyperOptSearch(
                metric="fraud_f1_score",
                mode="max"
            )
            print("Using HyperOpt (TPE)")
        else:
            search_alg = None  # Random/Grid search
            print("Using random/grid search")

        # Select scheduler
        if args.scheduler == "asha":
            scheduler = ASHAScheduler(
                metric="fraud_f1_score",
                mode="max",
                max_t=args.max_iterations if args.mode == "standard" else args.max_episodes,
                grace_period=10 if args.mode == "standard" else 500,
                reduction_factor=3
            )
            print("Using ASHA scheduler (early stopping)")
        elif args.scheduler == "pbt":
            # PBT hyperparam mutations depend on mode
            if args.mode == "multiagent":
                mutations = {
                    "lr": tune.loguniform(1e-5, 1e-2),
                    "gamma": tune.uniform(0.95, 0.999),
                    "judge_weight_end": tune.uniform(0.15, 0.4),
                    "uncertainty_penalty": tune.uniform(0.05, 0.2),
                }
            else:
                mutations = {
                    "lr": tune.loguniform(1e-5, 1e-2),
                    "gamma": tune.uniform(0.95, 0.999),
                }

            scheduler = PopulationBasedTraining(
                time_attr="training_iteration" if args.mode == "standard" else "episode",
                metric="fraud_f1_score",
                mode="max",
                perturbation_interval=10 if args.mode == "standard" else 500,
                hyperparam_mutations=mutations
            )
            print("Using Population Based Training (PBT)")
        else:
            scheduler = None
            print("Using FIFO scheduler")

        # Configure training function based on mode
        if args.mode == "multiagent":
            training_fn = tune.with_parameters(
                multiagent_training_function,
                env_config=env_config,
                graph=graph,
                illicit_nodes=illicit_nodes,
                max_episodes=args.max_episodes,
            )
            stop_criterion = {"episode": args.max_episodes}
        else:
            base_config = create_base_config(args, env_config)
            training_fn = tune.with_parameters(
                standard_training_function,
                base_config=base_config,
                max_iterations=args.max_iterations
            )
            stop_criterion = {"training_iteration": args.max_iterations}

        # Configure tuner
        tuner = tune.Tuner(
            training_fn,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=args.num_samples,
                max_concurrent_trials=args.max_concurrent,
            ),
            run_config=tune.RunConfig(
                name=args.experiment_name,
                storage_path=args.results_dir,
                stop=stop_criterion,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=10 if args.mode == "standard" else 500,
                    num_to_keep=3,
                ),
                verbose=1,
            ),
        )

        print(f"\nStarting hyperparameter search:")
        print(f"  Mode: {args.mode}")
        print(f"  Search space: {args.search_type}")
        print(f"  Search algorithm: {args.search_algo}")
        print(f"  Scheduler: {args.scheduler}")
        print(f"  Number of trials: {args.num_samples}")
        print(f"  Max concurrent trials: {args.max_concurrent}")
        if args.mode == "multiagent":
            print(f"  Max episodes per trial: {args.max_episodes}")
        else:
            print(f"  Max iterations per trial: {args.max_iterations}")
        print()

        # Run tuning
        results = tuner.fit()

        # Print best result
        best_result = results.get_best_result(metric="fraud_f1_score", mode="max")

        print("\n" + "="*70)
        print("BEST HYPERPARAMETERS FOUND")
        print("="*70)
        print(f"Fraud F1 Score: {best_result.metrics['fraud_f1_score']:.4f}")
        print(f"Episode Reward: {best_result.metrics.get('episode_reward_mean', 0):.2f}")
        print()
        print("Best hyperparameters:")
        for key, value in sorted(best_result.config.items()):
            if not key.startswith("_"):  # Skip internal params
                print(f"  {key}: {value}")
        print()
        print(f"Best checkpoint: {best_result.checkpoint}")
        print("="*70)

        # Export results
        results_df = results.get_dataframe()
        results_csv = Path(args.results_dir) / args.experiment_name / "results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults exported to: {results_csv}")

        # Export best config as JSON for easy reuse
        best_config_path = Path(args.results_dir) / args.experiment_name / "best_config.json"
        with open(best_config_path, "w") as f:
            # Filter to serializable values
            serializable_config = {
                k: v for k, v in best_result.config.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))
            }
            json.dump(serializable_config, f, indent=2)
        print(f"Best config exported to: {best_config_path}")

    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Ray Tune")

    # Data arguments
    parser.add_argument("--node-features", type=str, required=True)
    parser.add_argument("--edges", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument("--node-feature-dim", type=int, default=166)

    # Mode selection
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "multiagent"],
                        help="Training mode: 'standard' (RLlib DQN) or 'multiagent' (with judge)")

    # Tuning arguments
    parser.add_argument("--search-type", type=str, default="medium",
                        choices=["quick", "medium", "large"],
                        help="Size of search space")
    parser.add_argument("--search-algo", type=str, default="optuna",
                        choices=["random", "optuna", "hyperopt"],
                        help="Search algorithm")
    parser.add_argument("--scheduler", type=str, default="asha",
                        choices=["fifo", "asha", "pbt"],
                        help="Trial scheduler")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of trials to run")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Maximum concurrent trials")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Max iterations per trial (standard mode)")
    parser.add_argument("--max-episodes", type=int, default=5000,
                        help="Max episodes per trial (multiagent mode)")

    # Resource arguments
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)

    # Output arguments
    parser.add_argument("--experiment-name", type=str, default="fraud_detection_tune")
    parser.add_argument("--results-dir", type=str, default="./ray_results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_tune(args)


if __name__ == "__main__":
    main()
