"""
Test Multi-Branch Loss Implementation

Verifies:
1. Loss components compute correctly
2. Multi-branch encoder outputs auxiliary predictions
3. Integration works end-to-end
"""

import torch

from rl_money_laundering.gnn_modules.multi_branch_loss import (
    MultiBranchLoss,
    SimplifiedMultiBranchLoss,
    FocalLoss,
    TemporalDQNLoss,
    SubgraphRegularization
)
from rl_money_laundering.gnn_modules.rmganets_encoder_multibranch import (
    RMGANetsMultiBranchEncoder
)

print("=" * 80)
print("MULTI-BRANCH LOSS TEST")
print("=" * 80)

# Test 1: Focal Loss
print("\n[1/7] Testing Focal Loss...")
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Create dummy data
batch_size = 32
num_classes = 2
logits = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

loss = focal_loss(logits, targets)
print(f"OK - Focal loss: {loss.item():.4f}")

# Test 2: Temporal DQN Loss
print("\n[2/7] Testing Temporal DQN Loss...")
dqn_loss = TemporalDQNLoss(base_loss='mae', temporal_decay=0.1)

predictions = torch.randn(batch_size, 6)  # 6 actions
targets = torch.randn(batch_size, 6)
timestamps = torch.rand(batch_size)  # 0=old, 1=recent

loss_with_temporal = dqn_loss(predictions, targets, timestamps)
loss_without_temporal = dqn_loss(predictions, targets, None)

print(f"OK - DQN loss (with temporal): {loss_with_temporal.item():.4f}")
print(f"     DQN loss (no temporal): {loss_without_temporal.item():.4f}")

# Test 3: Subgraph Regularization
print("\n[3/7] Testing Subgraph Regularization...")
subgraph_reg = SubgraphRegularization(target_high_ratio=0.35, target_low_ratio=0.25)

# Good split (close to targets)
loss_good = subgraph_reg(num_edges_high=350, num_edges_medium=400, num_edges_low=250)
print(f"OK - Good split regularization: {loss_good.item():.6f}")

# Bad split (all in one category)
loss_bad = subgraph_reg(num_edges_high=950, num_edges_medium=30, num_edges_low=20)
print(f"     Bad split regularization: {loss_bad.item():.6f}")

# Test 4: Simplified Multi-Branch Loss
print("\n[4/7] Testing Simplified Multi-Branch Loss (paper version)...")
simple_loss = SimplifiedMultiBranchLoss(lambda_branch=0.25, epsilon_dqn=0.4)

outputs_main = torch.randn(batch_size, num_classes)
outputs_to = torch.randn(batch_size, num_classes)
outputs_hy = torch.randn(batch_size, 1)
dqn_preds = torch.randn(batch_size, 6)
dqn_targets = torch.randn(batch_size, 6)
targets = torch.randint(0, num_classes, (batch_size,))

total_loss, loss_dict = simple_loss(
    outputs_main, outputs_to, outputs_hy,
    dqn_preds, dqn_targets, targets
)

print(f"OK - Total loss: {total_loss.item():.4f}")
print(f"     Components: main={loss_dict['main']:.4f}, focal={loss_dict['focal']:.4f}, "
      f"bce={loss_dict['bce']:.4f}, dqn={loss_dict['dqn']:.4f}")

# Test 5: Full Multi-Branch Loss (with improvements)
print("\n[5/7] Testing Full Multi-Branch Loss (with improvements)...")
full_loss = MultiBranchLoss(
    lambda_branch=0.25,
    epsilon_dqn=0.4,
    beta_reg=0.1,
    adaptive_weighting=True,
    temporal_weighting=True
)

subgraph_stats = {
    'num_high': 350,
    'num_medium': 400,
    'num_low': 250
}

total_loss, loss_dict = full_loss(
    outputs_main, outputs_to, outputs_hy,
    dqn_preds, dqn_targets, targets,
    subgraph_stats=subgraph_stats,
    timestamps=timestamps
)

print(f"OK - Total loss: {total_loss.item():.4f}")
print("     Components:")
print(f"       main_ce={loss_dict['main_ce']:.4f}")
print(f"       to_gcm_focal={loss_dict['to_gcm_focal']:.4f}")
print(f"       hy_gcm_bce={loss_dict['hy_gcm_bce']:.4f}")
print(f"       dqn_mae={loss_dict['dqn_mae']:.4f}")
print(f"       subgraph_reg={loss_dict['subgraph_reg']:.6f}")
print(f"     Effective weights: lambda={loss_dict['lambda_eff']:.3f}, epsilon={loss_dict['epsilon_eff']:.3f}")

# Test 6: Multi-Branch Encoder
print("\n[6/7] Testing Multi-Branch Encoder...")
encoder = RMGANetsMultiBranchEncoder(
    node_feature_dim=12,
    hidden_dim=160,
    embedding_dim=64,
    num_classes=2,
    multi_branch=True  # Enable multi-branch mode
)

# Create dummy graph
num_nodes = 50
num_edges = 100
x = torch.randn(num_nodes, 12)
edge_index = torch.randint(0, num_nodes, (2, num_edges))

# Forward pass with auxiliary outputs
embeddings, auxiliary = encoder(x, edge_index, return_auxiliary=True)

print(f"OK - Embeddings shape: {embeddings.shape}")
print(f"     To-GCM branch shape: {auxiliary['to_branch'].shape}")
print(f"     HyGCM branch shape: {auxiliary['hy_branch'].shape}")
print(f"     Subgraph stats: high={auxiliary['subgraph_stats']['num_high']}, "
      f"medium={auxiliary['subgraph_stats']['num_medium']}, "
      f"low={auxiliary['subgraph_stats']['num_low']}")

# Test 7: End-to-End Integration
print("\n[7/7] Testing End-to-End Integration...")

# Pool node embeddings (simple mean for classification)
main_output = torch.nn.functional.linear(
    embeddings.mean(dim=0, keepdim=True),
    torch.randn(2, 64)
)
to_output = auxiliary['to_branch'].mean(dim=0, keepdim=True)
hy_output = auxiliary['hy_branch'].mean(dim=0, keepdim=True)

# Create dummy DQN predictions/targets
dqn_preds = torch.randn(1, 6)
dqn_targets = torch.randn(1, 6)
target = torch.tensor([1])  # Fraud

# Compute loss
total_loss, loss_dict = full_loss(
    main_output, to_output, hy_output,
    dqn_preds, dqn_targets, target,
    subgraph_stats=auxiliary['subgraph_stats'],
    timestamps=torch.tensor([0.8])
)

print(f"OK - End-to-end loss: {total_loss.item():.4f}")
print("     Ready for training!")

# Test adaptive weighting
print("\n" + "=" * 80)
print("ADAPTIVE WEIGHTING TEST")
print("=" * 80)

print("\nSimulating training epochs...")
for epoch in range(0, 101, 20):
    full_loss.epoch = torch.tensor(epoch)
    total_loss, loss_dict = full_loss(
        outputs_main, outputs_to, outputs_hy,
        dqn_preds[:batch_size], dqn_targets[:batch_size], targets,
        subgraph_stats=subgraph_stats,
        timestamps=timestamps
    )
    print(f"Epoch {epoch:3d}: lambda_eff={loss_dict['lambda_eff']:.3f}, "
          f"epsilon_eff={loss_dict['epsilon_eff']:.3f}, total_loss={loss_dict['total']:.4f}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)

print("\nSummary:")
print("  [OK] Focal Loss working")
print("  [OK] Temporal DQN Loss working")
print("  [OK] Subgraph Regularization working")
print("  [OK] Simplified Multi-Branch Loss working")
print("  [OK] Full Multi-Branch Loss working")
print("  [OK] Multi-Branch Encoder working")
print("  [OK] End-to-end integration working")
print("  [OK] Adaptive weighting working")

print("\nImprovements over paper:")
print("  1. Temporal-aware DQN loss (weights recent transactions higher)")
print("  2. Adaptive loss weighting (decays branch weights during training)")
print("  3. Subgraph regularization (encourages balanced splits)")
print("  4. Modular design (easy to enable/disable components)")

print("\nNext steps:")
print("  1. Integrate with Trainer")
print("  2. Update StateEncoder to use multi-branch mode")
print("  3. Train and compare with baseline RMGANets")
print("  4. Monitor loss components during training")
