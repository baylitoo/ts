# Multi-Agent AML

## Overview
Multi-agent components that extend the AML environment with detector/adversary
roles, learned judge rewards, and coordination protocols for multi-agent
training.

## Key modules
- `multi_agent_env.py`: `MultiAgentAMLEnvironment` and
  `DetectorAdversaryConfig` for detector/adversary interactions and reward
  shaping in a multi-agent setting.
- `judge_model.py`: ModernBERT-based learned judge model for episode scoring,
  canonical episode serialization, and safety controls.
- `reward_computation.py`: Conservative reward shaping with hard-metric
  anchoring, judge shaping, uncertainty penalties, and drift checks.
- `coordination.py`: Agent coordination abstractions (message bus, proposal
  types, training modes).
- `training_callbacks.py`: Two-timescale judge training callbacks and safety
  monitoring.
- `integration/`: Compatibility layer that integrates these components with the
  existing single-agent pipeline.

## Usage notes
- The adversary role is optional; the default configuration keeps the detector
  as the sole agent for MVP runs.
- Reward shaping is intentionally conservative to prevent judge exploitation and
  maintain alignment with precision/recall targets.
