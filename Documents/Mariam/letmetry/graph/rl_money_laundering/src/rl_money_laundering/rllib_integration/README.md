# RLlib Integration

## Overview
This folder bridges the project’s AML environment and GNN encoders with Ray RLlib’s
new API stack. It focuses on packaging graph observations into RLlib-friendly
spaces, batching with PyG, and surfacing fraud-aware metrics during training.

## Key modules
- `env_wrapper.py`: Wraps `AMLDetectionEnv` into a RLlib-compatible Gymnasium
  environment, converting observations into `GraphSpace` and handling config-based
  initialization/registration.
- `graph_space.py`: Defines `GraphSpace`, a custom Gymnasium space for
  graph-structured observations with padded node/edge tensors and context features.
- `batch_utils.py`: Utilities and collators that batch `GraphSpace` observations
  into PyTorch Geometric `Batch` objects and extract per-graph node embeddings.
- `gnn_rl_module.py`: `GNNDQNModule` implementation that plugs the project’s
  `StateEncoder` into RLlib’s TorchRLModule API for DQN.
- `fraud_callbacks.py`: RLlib callbacks for fraud-specific episode metrics
  (discovery rate, precision/recall, exploration coverage).
- `fraud_replay_buffer.py`: Fraud-aware replay buffer that tracks fraud-heavy
  episodes for prioritized replay.

## Usage notes
- Use `register_aml_env()` from `env_wrapper.py` to register the environment name
  and reference it in RLlib configs.
- For graph batching, pass `GraphSpace` observations through `GraphBatchCollator`
  to obtain a PyG `Batch` before invoking GNN layers.
