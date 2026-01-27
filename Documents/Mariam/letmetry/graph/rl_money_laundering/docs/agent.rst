.. _agent:

===========
RL Agents
===========

The policy layer ships with two interchangeable agents for graph-based AML detection. Both reuse the same environment, replay buffer abstraction, and configuration objects—switching between them requires only a couple of parameter changes in ``scripts/train_agent.py`` or a JSON config.

- ``DQNAgent`` – dueling architecture + Double DQN + prioritized replay. Optimised for fast experiments and production baselines.
- ``QRDQNAgent`` – distributional (Quantile Regression) variant with n-step prioritized replay and Conditional Value at Risk (CVaR) decision rules for risk-sensitive triage.

The diagram below highlights the shared training loop and the branching policy heads.

.. mermaid::

   graph TB
       subgraph Encode["State Encoder"]
           State["StateEncoder\n(GNN + history)"]
       end
       subgraph Replay["Replay Buffer"]
           PER["Prioritized Buffer"]
           NSPER["N-Step Prioritized Buffer"]
       end
       subgraph Policies
           DQN["DQNAgent\n(Dueling + Double)"]
           QR["QRDQNAgent\n(Quantile + CVaR)"]
       end
       Trainer["AMLTrainer"]
       Env["AMLDetectionEnv"]
       Target["Target Network"]

       State --> DQN
       State --> QR
       DQN --> Replay
       QR --> Replay
       Replay --> DQN
       Replay --> QR
       DQN --> Target
       QR --> Target
       Trainer -->|select_action| DQN
       Trainer -->|select_action| QR
       Trainer --> Env
       Env -->|transition| Trainer
       Target -->|bootstrap| DQN
       Target -->|bootstrap| QR

.. admonition:: Choosing an Agent
   :class: tip

   - Need quick convergence and lightweight checkpoints? Use ``DQNAgent`` (default).
   - Investigating rare, high-value laundering pathways or want CVaR-based risk control? Upgrade to ``QRDQNAgent``.
   - Both agents operate on the same ``state_dim`` emitted by ``rl_money_laundering.gnn_encoder.StateEncoder`` and share the trainer interface.


Architecture Details
====================

Shared Input
------------

The agents consume the state vector emitted by ``rl_money_laundering.gnn_encoder.StateEncoder``:

1. **GNN embedding** – GraphSAGE, GAT, or RMGANets encoders over a 2-hop ego network.
2. **History block** – visit counts, fraud-edge density, amount statistics (``log1p``), temporal sin/cos features, and aggregated risk scores.

Changing ``config.gnn`` (for example, ``history_dim`` or ``embedding_dim``) automatically propagates to both agents because the final ``state_dim`` is computed at runtime.


Dueling DQN Path
----------------

``DQNAgent`` instantiates ``rl_money_laundering.agent.DuelingDQNNetwork`` when ``use_dueling=True`` (default). The network splits into value and advantage streams and recombines with mean-normalised aggregation:

.. math::

   Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)

The trainer pairs this with **Double DQN** targets (online network selects, target network evaluates) and **prioritized replay** via ``rl_money_laundering.agent.PrioritizedReplayBuffer``. Each optimisation step clips gradients, anneals importance-sampling weights, and periodically syncs the target network.


Quantile (Distributional) Path
------------------------------

``QRDQNAgent`` wraps ``rl_money_laundering.agent.QuantileRegressionDQN`` and outputs a full return distribution ``Z(s,a)`` across ``num_quantiles`` support points. During evaluation and action selection, quantiles convert into Q-values using user-selectable risk measures:

- ``mean`` – risk-neutral.
- ``median`` – robust to outliers.
- ``cvar_XX`` – Conditional Value at Risk (for example ``cvar_90`` focuses on the worst 10% outcomes).

The agent combines quantile regression with n-step returns and prioritized replay (Rainbow-style) while still benefiting from Double DQN bootstrapping.


Training Loop
=============

``AMLTrainer`` coordinates both agents. The loop combines curriculum staging, epsilon decay, evaluation sweeps, and checkpointing. The curriculum defaults to a seven-stage schedule that starts in fraud-heavy subgraphs and gradually approaches a 2% fraud prior.

.. mermaid::

   sequenceDiagram
       participant Trainer
       participant Agent
       participant Env
       participant Replay
       participant Target

       Trainer->>Env: reset()
       Env-->>Trainer: state_0, info
       loop Steps
           Trainer->>Agent: select_action(state_t, valid_actions, epsilon_t)
           Agent-->>Trainer: action_t
           Trainer->>Env: step(action_t)
           Env-->>Trainer: state_{t+1}, reward_t, done?, info
           Trainer->>Agent: store_transition(...)
           alt buffer ready
               Agent->>Replay: sample(batch)
               Replay-->>Agent: transitions, weights
               Agent->>Target: compute bootstrap targets
               Agent->>Agent: optimise (Huber / Quantile Huber)
               Agent->>Replay: update priorities
           end
           alt target update tick
               Agent->>Target: sync weights
           end
       end
       alt evaluation tick
           Trainer->>Agent: set epsilon = 0
           Trainer->>Env: run greedy episodes
           Trainer-->>Trainer: log detection / FPR
       end
       alt checkpoint tick
           Trainer->>Agent: save()
       end

Curriculum & Metrics
--------------------

- ``curriculum_schedule`` – defaults to ``[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.02]``. Each stage runs ``num_episodes // len(schedule)`` iterations.
- ``eval_frequency`` – evaluate the current policy with greedy rollouts (20 episodes) and record detection rate, false positive rate, and mean reward.
- ``checkpoint_frequency`` – persists ``checkpoint_ep{N}.pt`` plus ``training_stats.npz`` containing reward curves and detection metrics.


Configuring Agents
==================

Preset configurations in ``rl_money_laundering.config`` expose the key toggles. Example: enabling the distributional agent in ``scripts/train_agent.py``.

.. code-block:: python
   :caption: Switching to the distributional agent

   from rl_money_laundering.agent import DQNAgent, QRDQNAgent
   from rl_money_laundering.config import get_full_amlnet_config

   config = get_full_amlnet_config()

   # Default dueling Double DQN
   agent = DQNAgent(
       state_dim=config.agent.state_dim,
       action_dim=config.agent.action_dim,
       use_prioritized_replay=True,
       hidden_dims=config.agent.hidden_dims,
       device=config.agent.device,
   )

   # Drop-in replacement: risk-sensitive QR-DQN
   dist_agent = QRDQNAgent(
       state_dim=config.agent.state_dim,
       action_dim=config.agent.action_dim,
       num_quantiles=64,
       risk_measure="cvar_90",
       n_step=3,
       buffer_capacity=100_000,
       device=config.agent.device,
   )

   trainer = AMLTrainer(
       graph=graph,
       agent=dist_agent,
       state_encoder=state_encoder,
       env=env,
       fraud_subgraphs=fraud_subgraphs,
   )

Key Hyperparameters
-------------------

.. list-table:: Core Agent Settings
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - DQNAgent Default
     - QRDQNAgent Default
   * - ``learning_rate``
     - ``1e-3``
     - ``1e-4``
   * - ``hidden_dims``
     - ``[128, 128, 64]``
     - ``[512, 256, 128]``
   * - ``buffer_capacity``
     - ``10_000``
     - ``100_000``
   * - ``use_prioritized_replay``
     - ``True``
     - ``True`` (n-step)
   * - ``num_quantiles``
     - N/A
     - ``64`` (configurable)
   * - ``risk_measure``
     - ``mean`` (epsilon-greedy)
     - ``mean`` (can be ``median`` or ``cvar_XX``)
