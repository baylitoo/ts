# RL Money Laundering

## Overview
This repository explores AML detection with reinforcement learning and
GNN-based state encoders. It combines transaction graph datasets, feature
engineering, RL environments, RLlib integration, and a multi-agent judge
framework to study fraud discovery under operational alert budgets.

## Key areas
- `src/rl_money_laundering/datasets`: Dataset loaders and graph construction for
  AMLNet and Elliptic data.
- `src/rl_money_laundering/features`: Feature extractors for node, temporal, and
  network signals.
- `src/rl_money_laundering/gnn_modules`: RMGANets and related GNN components used
  by the state encoder.
- `src/rl_money_laundering/rllib_integration`: RLlib wrappers, graph spaces, and
  GNN RLModule integration.
- `src/rl_money_laundering/multiagent`: Multi-agent environment, judge model,
  reward shaping, and coordination utilities.
- `src/rl_money_laundering/multiagent/integration`: Compatibility layer that
  integrates judge/reward logic with existing training loops.
- `src/rl_money_laundering/pretraining`: Continuous pretraining utilities for
  the learned judge and AML text corpora.
- `src/rl_money_laundering/evaluation`: AML-focused classification, ranking,
  calibration, and RL metrics.
- `src/rl_money_laundering/baselines`: Classical ML baselines (e.g., XGBoost)
  for comparison.
- `src/rl_money_laundering/dataprep`: Text data preparation pipeline for
  AML corpora.
- `src/rl_money_laundering/utils`: Shared graph utilities, replay buffers,
  checkpointing, and experiment tracking helpers.

## Getting started
- Review `QUICKSTART.md` for environment setup and runnable examples.
- Use `configs/` and `scripts/` for training workflows and reproducible runs.
- See `docs/` for additional architecture notes and diagrams.
