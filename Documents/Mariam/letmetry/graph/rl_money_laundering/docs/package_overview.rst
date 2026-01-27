Package Overview
================

This section summarizes the main Python packages in ``src/rl_money_laundering``
and highlights the primary entry points for AML training, evaluation, and
pretraining workflows.

Core runtime
------------
- ``rl_money_laundering.environment``: AML detection environment used by the
  RL trainer and RLlib wrappers.
- ``rl_money_laundering.agent`` and ``rl_money_laundering.trainer``: Agent
  implementations (DQN/QR-DQN) and the training loop orchestration.
- ``rl_money_laundering.gnn_encoder`` and ``rl_money_laundering.gnn_modules``:
  GNN state encoder plus RMGANets components.

Datasets & features
-------------------
- ``rl_money_laundering.datasets``: AMLNet and Elliptic dataset loaders with
  feature pipelines and graph construction utilities.
- ``rl_money_laundering.features``: Node, temporal, and network feature
  extractors used by the environment and baselines.
- ``rl_money_laundering.baselines``: Classical ML baselines (XGBoost) for
  graph-unaware comparisons.

RLlib integration
-----------------
- ``rl_money_laundering.rllib_integration``: Graph observation spaces, batching
  utilities, RLlib environment wrapper, and TorchRLModule integration.

Multi-agent & judge framework
-----------------------------
- ``rl_money_laundering.multiagent``: Detector/adversary environment, learned
  judge model, reward shaping, and coordination utilities.
- ``rl_money_laundering.multiagent.integration``: Compatibility layer that
  injects judge rewards into existing single-agent training loops via adapters
  and orchestrators.

Evaluation & utilities
----------------------
- ``rl_money_laundering.evaluation``: Classification, ranking, calibration, and
  RL-specific metrics.
- ``rl_money_laundering.utils``: Graph utilities, replay buffers, checkpoints,
  and experiment tracking.

Text data prep & pretraining
----------------------------
- ``rl_money_laundering.dataprep``: AML text corpus ingestion, cleaning,
  deduplication, and export pipeline.
- ``rl_money_laundering.pretraining``: Continuous pretraining utilities,
  collators, and judge fine-tuning helpers.
