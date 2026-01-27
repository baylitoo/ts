# Evaluation Metrics

## Overview
Evaluation utilities for AML detection, covering classification, ranking,
calibration, budgeted alerting, and RL-specific metrics.

## Key modules
- `metrics.py`: `MetricsComputer` with classification and ranking metrics
  (AUC, F1, NDCG, MRR, Precision@K, Recall@K).
- `budgeted_metrics.py`: Budget-aware metrics for compliance alerting
  (Recall@K, Precision@K, AUPR, PR curves).
- `calibration_metrics.py`: Calibration diagnostics such as ECE, MCE, Brier
  score, and reliability diagram data.
- `rl_metrics.py`: Episode-level RL metrics (returns, coverage, budget
  utilization) and aggregation helpers.

## Usage notes
- `BudgetedMetrics` is tailored to the “K alerts per day” constraint common in
  AML operations.
- `CalibrationComputer` helps validate probability outputs used for downstream
  prioritization.
