"""
End-to-End Integration Test for RMGANets + Multi-Branch Loss + Pipeline

Tests the complete training pipeline with:
- RMGANets encoder in multi-branch mode
- Multi-branch loss (both paper and improved variants)
- Full RL training loop
- build_pipeline() orchestration
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from rl_money_laundering.config import ExperimentConfig, MultiBranchLossConfig, GNNConfig
from rl_money_laundering.pipeline import build_pipeline


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for test artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config():
    """Create minimal config for testing"""
    config = ExperimentConfig(
        name="test_integration",
        dataset_type="simple",  # Use simple synthetic dataset
        dataset_path=None
    )
    return config


def test_build_pipeline_baseline(minimal_config, temp_output_dir):
    """
    Test pipeline with baseline GraphSAGE (no multi-branch)
    """
    # Configure for baseline
    minimal_config.gnn.gnn_type = "sage"
    minimal_config.gnn.embedding_dim = 32
    minimal_config.trainer.multi_branch.enabled = False

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Verify components created
    assert artifacts.dataset is not None
    assert artifacts.env is not None
    assert artifacts.state_encoder is not None
    assert artifacts.agent is not None
    assert artifacts.trainer is not None
    assert artifacts.checkpoint_manager is not None

    # Verify encoder is NOT in multi-branch mode
    assert not artifacts.trainer.multi_branch_enabled
    assert artifacts.state_encoder.gnn_type == "sage"


def test_build_pipeline_rmganets_paper_variant(minimal_config, temp_output_dir):
    """
    Test pipeline with RMGANets + multi-branch loss (paper variant)
    """
    # Configure for RMGANets with paper loss
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2,
        use_dqn_enhancement=False
    )

    minimal_config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="paper",
        lambda_branch=0.25,
        epsilon_dqn=0.4
    )

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Verify RMGANets multi-branch enabled
    assert artifacts.trainer.multi_branch_enabled
    assert artifacts.state_encoder.multi_branch_enabled
    assert artifacts.trainer.multi_branch_config.variant == "paper"

    # Verify loss function is SimplifiedMultiBranchLoss
    from rl_money_laundering.gnn_modules.multi_branch_loss import SimplifiedMultiBranchLoss
    assert isinstance(artifacts.trainer.multi_branch_loss_fn, SimplifiedMultiBranchLoss)


def test_build_pipeline_rmganets_improved_variant(minimal_config, temp_output_dir):
    """
    Test pipeline with RMGANets + multi-branch loss (improved variant)
    """
    # Configure for RMGANets with improved loss
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2,
        use_dqn_enhancement=False
    )

    minimal_config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="improved",
        lambda_branch=0.25,
        epsilon_dqn=0.4,
        beta_reg=0.1,
        temporal_decay=0.1,
        adaptive_weighting=True
    )

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Verify RMGANets multi-branch enabled with improvements
    assert artifacts.trainer.multi_branch_enabled
    assert artifacts.trainer.multi_branch_config.variant == "improved"
    assert artifacts.trainer.multi_branch_config.beta_reg == 0.1
    assert artifacts.trainer.multi_branch_config.adaptive_weighting is True

    # Verify loss function is MultiBranchLoss (improved)
    from rl_money_laundering.gnn_modules.multi_branch_loss import MultiBranchLoss
    assert isinstance(artifacts.trainer.multi_branch_loss_fn, MultiBranchLoss)


def test_training_loop_with_multibranch(minimal_config, temp_output_dir):
    """
    Test actual training loop with multi-branch loss
    """
    # Configure for RMGANets
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2
    )

    minimal_config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="improved",
        log_frequency=1  # Log every episode for testing
    )

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Run short training (5 episodes)
    trainer = artifacts.trainer
    history = trainer.train(
        num_episodes=5,
        eval_frequency=10,  # No eval during test
        checkpoint_frequency=10,  # No checkpoints during test
        batch_size=32
    )

    # Verify training ran
    assert len(history['episode_rewards']) == 5
    assert len(history['episode_lengths']) == 5

    # Verify multi-branch metrics were collected
    assert len(trainer.multi_branch_metrics) > 0

    # Verify multi-branch metrics have expected keys
    sample_metrics = trainer.multi_branch_metrics[0]
    assert 'loss' in sample_metrics
    assert 'loss_branch' in sample_metrics
    assert 'loss_dqn' in sample_metrics


def test_state_encoder_auxiliary_outputs(minimal_config, temp_output_dir):
    """
    Test that state encoder returns auxiliary outputs in multi-branch mode
    """
    # Configure for RMGANets
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2
    )

    minimal_config.trainer.multi_branch.enabled = True

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Get a sample state encoding
    graph = artifacts.dataset.graph
    current_node = list(graph.nodes())[0]

    # Encode with auxiliary outputs
    embedding, history, aux_outputs = artifacts.state_encoder.forward_state(
        graph=graph,
        current_node=current_node,
        visited_nodes=set(),
        visited_edges=set(),
        node_feature_extractor=artifacts.trainer.node_feature_extractor,
        return_auxiliary=True
    )

    # Verify auxiliary outputs exist
    assert aux_outputs is not None
    assert 'to_branch_logits' in aux_outputs
    assert 'hy_branch_logits' in aux_outputs
    assert 'subgraph_stats' in aux_outputs

    # Verify shapes
    assert embedding.shape[-1] == 32  # embedding_dim
    assert aux_outputs['to_branch_logits'].shape[-1] == 2  # num_classes


def test_multi_branch_loss_computation(minimal_config, temp_output_dir):
    """
    Test that multi-branch loss is actually computed during training
    """
    # Configure for RMGANets
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2
    )

    minimal_config.trainer.multi_branch = MultiBranchLossConfig(
        enabled=True,
        variant="improved",
        log_frequency=1
    )

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Run one episode
    trainer = artifacts.trainer
    metrics_before = len(trainer.multi_branch_metrics)

    trainer.run_episode(max_steps=10, epsilon=1.0)

    metrics_after = len(trainer.multi_branch_metrics)

    # Verify multi-branch loss was computed
    assert metrics_after > metrics_before, "Multi-branch metrics should have increased"

    # Verify loss components are tracked
    if trainer.multi_branch_metrics:
        latest_metrics = trainer.multi_branch_metrics[-1]
        assert 'loss' in latest_metrics
        assert isinstance(latest_metrics['loss'], float)
        assert not np.isnan(latest_metrics['loss'])


def test_gradient_flow_multibranch(minimal_config, temp_output_dir):
    """
    Test that gradients flow through multi-branch components
    """
    # Configure for RMGANets
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2
    )

    minimal_config.trainer.multi_branch.enabled = True

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Get initial parameter values
    initial_params = {}
    for name, param in artifacts.state_encoder.gnn.named_parameters():
        initial_params[name] = param.clone().detach()

    # Run one episode (should trigger multi-branch updates)
    artifacts.trainer.run_episode(max_steps=5, epsilon=1.0)

    # Verify parameters changed (gradients flowed)
    params_changed = False
    for name, param in artifacts.state_encoder.gnn.named_parameters():
        if not torch.equal(param, initial_params[name]):
            params_changed = True
            break

    assert params_changed, "Multi-branch training should update GNN parameters"


def test_checkpoint_save_load_multibranch(minimal_config, temp_output_dir):
    """
    Test that checkpoints work with multi-branch components
    """
    # Configure for RMGANets
    minimal_config.gnn = GNNConfig(
        node_feature_dim=12,
        gnn_type="rmganets",
        embedding_dim=32,
        multi_branch=True,
        num_classes=2
    )

    minimal_config.trainer.multi_branch.enabled = True

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Train for a few episodes
    artifacts.trainer.train(num_episodes=3, eval_frequency=10, checkpoint_frequency=10)

    # Save checkpoint manually
    checkpoint_path = temp_output_dir / "test_checkpoint.pt"
    artifacts.agent.save(str(checkpoint_path))

    # Verify checkpoint exists
    assert checkpoint_path.exists()

    # Load checkpoint
    artifacts.agent.load(str(checkpoint_path))

    # Verify agent state restored
    assert artifacts.agent.training_steps >= 0


@pytest.mark.parametrize("gnn_type,multi_branch", [
    ("sage", False),
    ("gat", False),
    ("rmganets", True),
])
def test_different_encoder_types(gnn_type, multi_branch, minimal_config, temp_output_dir):
    """
    Test pipeline with different GNN encoder types
    """
    minimal_config.gnn.gnn_type = gnn_type
    minimal_config.gnn.multi_branch = multi_branch
    minimal_config.trainer.multi_branch.enabled = multi_branch

    # Build pipeline
    artifacts = build_pipeline(
        minimal_config,
        nrows=100,
        max_edges=500,
        output_dir=temp_output_dir,
        device="cpu"
    )

    # Verify correct encoder type
    assert artifacts.state_encoder.gnn_type == gnn_type

    # Verify multi-branch status
    assert artifacts.trainer.multi_branch_enabled == multi_branch

    # Run one episode to verify it works
    stats = artifacts.trainer.run_episode(max_steps=5, epsilon=1.0)
    assert 'episode_reward' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
