# Utilities

## Overview
Shared utilities for graph manipulation, feature extraction, training
bookkeeping, and experiment tracking.

## Key modules
- `graph.py`: Graph operations (k-hop subgraphs, PyG conversion, neighbor
  sampling, graph statistics) plus leakage-aware splitting helpers.
- `features.py`: Node and edge feature extraction utilities and dataset-specific
  feature extractor factories.
- `graph_replay.py`: Graph-aware replay buffer for storing GNN-friendly
  transitions.
- `checkpoint.py`: `CheckpointManager` for saving and restoring agent state,
  optimizers, and experiment configs.
- `runtime_metrics.py`: GPU/latency profiling utilities with optional NVML
  support for hardware metrics.
- `experiment.py`: `ExperimentTracker` for logging training/evaluation metrics
  and generating experiment reports.

## Usage notes
- Use `entity_disjoint_split` or `temporal_split` from `graph.py` to reduce
  leakage between train/test sets.
- `CheckpointManager` expects the RL agent to expose `q_network`,
  `target_network`, `optimizer`, and `epsilon` attributes.
