# GNN Modules

## Overview
Graph neural network building blocks and RMGANets components used by the AML
state encoder. These modules implement attention-driven subgraph splitting,
adaptive topology convolutions, hybrid graph convolutions, and feature fusion.

## Key modules
- `rmganets_encoder.py`: Full RMGANets encoder pipeline integrating Att-GCM,
  To-GCM, HyGCM, and feature fusion, with optional DQN enhancement.
- `rmganets_encoder_multibranch.py`: Multi-branch variant of RMGANets for
  auxiliary outputs and loss decomposition.
- `multi_branch_loss.py`: Loss utilities for handling multi-branch outputs.
- `att_gcm.py`: Attention-based graph convolution module with subgraph
  splitting logic.
- `to_gcm.py`: Adaptive topology convolution module.
- `hy_gcm.py`: Hybrid enhanced graph convolution module.
- `fusion.py`: Feature fusion module that combines multi-branch representations.

## Usage notes
- These modules are orchestrated by the higher-level `StateEncoder` in
  `gnn_encoder.py` and exposed to RLlib via `GNNDQNModule`.
- The multi-branch encoder enables auxiliary supervision or diagnostic outputs
  during training.
