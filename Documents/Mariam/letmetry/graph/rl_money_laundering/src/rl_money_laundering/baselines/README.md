# Baselines

## Overview
This directory contains classical ML baselines used to compare against the RL/GNN stack.
The intent is to provide a strong “flat features” reference point that does not use
graph traversal or agent behavior.

## Key modules
- `xgboost_baseline.py`: Implements an XGBoost classifier over AMLNet Algorithm-2
  features (node + edge feature concatenation). Includes feature extraction, model
  training, evaluation, and model persistence helpers.

## Usage notes
- The `train_xgboost_baseline` helper builds features from a NetworkX graph and
  reports budgeted metrics such as AUPR and Recall@K.
- This baseline is deliberately “graph-unaware” to isolate the value of RL/graph
  reasoning in experiments.
