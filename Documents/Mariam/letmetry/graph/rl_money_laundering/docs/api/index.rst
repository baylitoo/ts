API Reference
=============

.. autosummary::
   :toctree: generated
   :caption: Core Modules
   :recursive:

   rl_money_laundering.config
   rl_money_laundering.pipeline
   rl_money_laundering.environment
   rl_money_laundering.agent
   rl_money_laundering.trainer
   rl_money_laundering.baselines
   rl_money_laundering.evaluator
   rl_money_laundering.gnn_encoder

.. autosummary::
   :toctree: generated
   :caption: GNN Modules
   :recursive:

   rl_money_laundering.gnn_modules
   rl_money_laundering.gnn_modules.att_gcm
   rl_money_laundering.gnn_modules.to_gcm
   rl_money_laundering.gnn_modules.hy_gcm
   rl_money_laundering.gnn_modules.fusion
   rl_money_laundering.gnn_modules.rmganets_encoder
   rl_money_laundering.gnn_modules.rmganets_encoder_multibranch
   rl_money_laundering.gnn_modules.multi_branch_loss

.. autosummary::
   :toctree: generated
   :caption: RLlib Integration
   :recursive:

   rl_money_laundering.rllib_integration
   rl_money_laundering.rllib_integration.graph_space
   rl_money_laundering.rllib_integration.env_wrapper
   rl_money_laundering.rllib_integration.gnn_rl_module
   rl_money_laundering.rllib_integration.batch_utils
   rl_money_laundering.rllib_integration.fraud_replay_buffer
   rl_money_laundering.rllib_integration.fraud_callbacks

.. autosummary::
   :toctree: generated
   :caption: Evaluation
   :recursive:

   rl_money_laundering.evaluation
   rl_money_laundering.evaluation.budgeted_metrics

.. autosummary::
   :toctree: generated
   :caption: Datasets
   :recursive:

   rl_money_laundering.datasets
   rl_money_laundering.datasets.base
   rl_money_laundering.datasets.amlnet
   rl_money_laundering.datasets.elliptic
   rl_money_laundering.datasets.features

.. autosummary::
   :toctree: generated
   :caption: Features
   :recursive:

   rl_money_laundering.features
   rl_money_laundering.features.base
   rl_money_laundering.features.extractors
   rl_money_laundering.features.elliptic_extractor
   rl_money_laundering.features.temporal
   rl_money_laundering.features.network
   rl_money_laundering.features.network_pyg

.. autosummary::
   :toctree: generated
   :caption: Utilities
   :recursive:

   rl_money_laundering.utils
   rl_money_laundering.utils.graph
   rl_money_laundering.utils.features
   rl_money_laundering.utils.checkpoint
   rl_money_laundering.utils.experiment
