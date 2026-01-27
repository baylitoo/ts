# Multi-Agent Integration Layer

## Overview
Compatibility layer that integrates multi-agent judge/reward components with
existing single-agent RL training loops without modifying core environment
logic.

## Key modules
- `environment_adapter.py`: `EnvironmentAdapter` and `EpisodeCollector` for
  capturing episode traces, fraud metrics, and anonymized features.
- `encoder_bridge.py`: Bridges the GNN `StateEncoder` to the judge model with
  embedding caching and optional judge-derived features.
- `reward_integrator.py`: Conservative reward combination that blends original
  rewards with judge scores and uncertainty penalties.
- `training_orchestrator.py`: Orchestrates training with episode collection,
  reward integration, judge updates, and safety monitoring.
- `config.py`: Integration configuration classes for judge weighting,
  adversary settings, and safety thresholds.

## Usage notes
- The orchestrator preserves original behavior when the judge is disabled
  (`compatibility_mode=True`).
- Episode data is anonymized before being sent to the judge, supporting audit
  and privacy requirements.
