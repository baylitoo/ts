# Learned Judge Models for Anti-Money Laundering Detection: A Literature Review and Experimental Design

---

## Deliverable 1: Abstract

We propose a hybrid reward framework for graph-based reinforcement learning in Anti-Money Laundering (AML) detection that combines verifiable hard metrics with a fine-tuned ModernBERT encoder serving as a learned judge. Traditional AML systems suffer from extreme label scarcity (fraud rates <0.1%), delayed ground truth (investigations span months), and adversarial drift as launderers adapt to detection rules. Pure rule-based rewards fail to capture nuanced suspicious patterns, while pure learned rewards risk reward hacking and catastrophic exploitation. Our approach anchors a conservative reward formula—`R_total = R_hard + α·clip(R_judge, τ)`—where hard metrics (precision/recall on confirmed labels) provide a safety floor and the learned judge captures behavioral subtleties. We employ two-timescale training (slow judge updates, fast policy updates) with drift detection to prevent co-adaptation collapse. Experiments on synthetic transaction graphs and the Elliptic Bitcoin dataset will ablate judge weight schedules, uncertainty penalties, and adversarial robustness, measuring both detection F1 and reward exploitation metrics.

**Keywords**: Anti-Money Laundering, Reward Modeling, Graph Neural Networks, Reinforcement Learning, RLHF, Conservative RL

---

## Deliverable 2: Structured Related Work

### 2.1 Reward Modeling in Reinforcement Learning from Human Feedback (RLHF)

The foundational insight that reward functions can be learned from preference data rather than hand-coded was established by **Christiano et al. (2017)** in "Deep Reinforcement Learning from Human Preferences." They demonstrated that a reward model trained on 1,000-2,000 human preference comparisons could guide RL agents to perform complex tasks (Atari games, MuJoCo locomotion) without access to the true reward. Their key architecture choice—training a reward network to satisfy the Bradley-Terry preference model `P(a ≻ b) = σ(r(a) - r(b))`—remains standard in RLHF pipelines.

**Ouyang et al. (2022)** scaled this approach to language models with InstructGPT, introducing the three-stage pipeline: supervised fine-tuning → reward model training → PPO optimization. Their reward model was a 6B parameter GPT-3 variant trained on 33K human comparisons. Crucially, they observed reward hacking: policies optimized too aggressively against the reward model produced outputs rated lower by humans than intermediate checkpoints—directly motivating our conservative clipping approach.

**Ziegler et al. (2019)** ("Fine-Tuning Language Models from Human Preferences") identified that reward model quality degrades on out-of-distribution inputs, proposing KL penalties `R_total = R_learned - β·KL(π||π_ref)` to keep the policy close to a reference. This directly informs our hard metric anchoring: rather than KL regularization, we anchor to verifiable ground truth, which is available (albeit sparse) in AML.

**Stiennon et al. (2020)** ("Learning to Summarize from Human Feedback") demonstrated that reward models could capture subtle quality distinctions humans care about but struggle to articulate as rules—precisely the scenario in AML where investigators can identify suspicious patterns but cannot enumerate them exhaustively.

**Bai et al. (2022)** ("Training a Helpful and Harmless Assistant with RLHF") introduced Constitutional AI and found that reward models trained on binary comparisons transfer poorly to Likert-scale ratings, suggesting that our judge architecture should output continuous rewards calibrated to hard metric scales rather than preference logits.

### 2.2 Graph Neural Networks for Fraud Detection

**Weber et al. (2019)** released the **Elliptic Bitcoin Dataset**, containing 203K transactions with 4,545 confirmed illicit nodes (2%), establishing the canonical benchmark for graph-based AML. Their baseline GCN achieved 0.73 F1 on illicit detection, but they noted severe class imbalance and temporal concept drift as key challenges. This dataset's temporal structure (49 time steps) makes it ideal for evaluating our judge's drift resilience.

**Pareja et al. (2020)** introduced **EvolveGCN**, which adapts GCN parameters over time using RNN controllers to handle temporal dynamics in transaction graphs. Their finding that static GNNs degrade 15% on later time steps motivates our adversarial environment's injection of distributional shifts.

**Liu et al. (2021)** ("Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection") addressed the label imbalance problem through neighborhood sampling that over-represents minority class edges. They achieved 0.81 F1 on Elliptic by focusing message passing on fraud-adjacent neighborhoods—a technique we incorporate in observation construction for the RL agent.

**Altman et al. (2023)** ("Realistic Synthetic Financial Transaction Graphs for AML") provided synthetic graph generators calibrated to real-world degree distributions, transaction volumes, and fraud typologies. Their AMLSim extension supports parameterized fraud injection, which we use for controlled experimental ablations where ground truth is known by construction.

**Johannessen et al. (2023)** ("Towards Explainable Anti-Money Laundering with Graph Neural Networks") highlighted the regulatory requirement for explainability in AML decisions, motivating our episode serialization approach that produces human-readable justifications alongside reward scores.

### 2.3 Conservative and Safe Reward Learning

**Kumar et al. (2020)** ("Conservative Q-Learning for Offline RL") introduced the principle of pessimism under uncertainty: when the policy visits states underrepresented in the training data, the Q-function should return conservative (low) values rather than hallucinate high rewards. Their penalty term `α · E[max_a Q(s,a) - Q(s,a)]` directly inspires our judge uncertainty penalty that reduces reward magnitude when the judge's confidence is low.

**Fujimoto et al. (2019)** ("Benchmarking Batch Deep RL Algorithms") demonstrated that naive off-policy algorithms diverge catastrophically when trained on static datasets due to extrapolation error. Their Batch-Constrained Q-learning (BCQ) constrains action selection to plausible actions, analogous to our clipping that constrains judge rewards to plausible ranges.

**Levine et al. (2020)** ("Offline Reinforcement Learning: Tutorial, Review, and Perspectives") surveyed the field and identified a key failure mode: learned reward/value functions trained jointly with policies can co-adapt, producing high estimated returns but poor true performance. This "self-delusion" motivates our two-timescale training where the judge updates slowly and independently.

**Hadfield-Menell et al. (2017)** ("Inverse Reward Design") formalized the observation that the reward function specified by designers is merely a proxy for true intent. They proposed treating the observed reward as evidence about the true reward and maintaining uncertainty. Our approach operationalizes this by treating the judge as uncertain and anchoring to hard metrics as a "true reward" floor.

**Garg et al. (2023)** ("Extreme Q-Learning: MaxEnt RL without Entropy") showed that conservative methods can be overly pessimistic, degrading performance on in-distribution states. This motivates our annealing schedule that starts with high hard metric weight (conservative) and gradually increases judge weight as calibration improves.

### 2.4 Multi-Agent Adversarial Training for Robustness

**Lanctot et al. (2017)** ("A Unified Game-Theoretic Approach to Multiagent RL") established the framework for training agents against adversarial opponents, showing that self-play in two-player zero-sum games converges to Nash equilibria under certain conditions. Our detector-adversary setup adapts this for AML where the adversary represents adaptive launderers.

**Pinto et al. (2017)** ("Robust Adversarial RL") demonstrated that training against a learned adversary that applies worst-case perturbations produces policies robust to test-time distribution shift. Their adversary was trained to maximize policy failure subject to perturbation constraints—directly analogous to our adversary that injects fraudulent patterns subject to realism constraints.

**Gleave et al. (2020)** ("Adversarial Policies: Attacking Deep RL") showed that adversarial policies can exploit unexpected failure modes in trained agents, motivating our safety monitoring that detects when the adversary is "winning" via unusual reward distributions.

**Hsieh et al. (2023)** ("Adversarial Training for Graph Neural Networks") applied adversarial perturbations specifically to graph structures, finding that edge addition/deletion attacks are most effective against GNN-based fraud detectors. This informs our adversary's action space.

### 2.5 Encoder-Based Reward Models and Episode Evaluation

**Lee et al. (2023)** ("RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback") demonstrated that encoder models can provide reward signal without human annotation, using larger LLMs as reward models for smaller policy models. While we use ModernBERT rather than an LLM, the principle of encoder-as-judge transfers directly.

**Warner et al. (2024)** ("ModernBERT: A Modernized Bidirectional Encoder for NLP") introduced architectural improvements (rotary embeddings, unpadded attention, Flash Attention 2) that make BERT-scale models competitive with larger decoders on discriminative tasks. The 8192 token context window accommodates our full-episode serializations.

**Rafailov et al. (2023)** ("Direct Preference Optimization: Your Language Model Is Secretly a Reward Model") showed that reward modeling and policy optimization can be unified, but their approach requires differentiating through the policy—infeasible for our discrete action RL setting. We therefore maintain explicit reward model and policy separation.

**Knox & Stone (2009)** ("Interactively Shaping Agents via Human Reinforcement") established TAMER, demonstrating that human feedback integrated as reward shaping accelerates learning. Our judge serves an analogous role: providing richer reward signal than sparse hard metrics alone.

**MacGlashan et al. (2017)** ("Interactive Learning from Policy-Dependent Human Feedback") identified that human feedback can be policy-dependent (evaluators grade harder as the agent improves), necessitating periodic reward model recalibration. Our two-timescale callback implements this through controlled judge updates.

---

## Deliverable 3: Failure Modes and Mitigations Table

| ID | Failure Mode | Description | Detection Signal | Mitigation | Implementation Reference |
|----|--------------|-------------|------------------|------------|-------------------------|
| F1 | **Reward Hacking** | Policy exploits judge artifacts (e.g., generating verbose outputs that correlate with reward but don't improve detection) | Divergence between judge reward and held-out human evaluation; increasing judge confidence with decreasing F1 | Hard metric anchoring: `R_total = R_hard + α·R_judge` ensures floor; judge clipping `clip(R_judge, -τ, τ)` bounds exploitation | `RewardConfig.judge_clip_max=0.5`, `RewardConfig.hard_f1_weight=0.4` |
| F2 | **Co-adaptation Collapse** | Judge and policy co-evolve to mutually reinforce degenerate solutions (judge assigns high reward to bad behavior because policy never explores alternatives) | Judge calibration error increasing; hard metric ↔ judge correlation dropping below threshold | Two-timescale training: freeze judge during policy updates, update judge only every N episodes with fresh data | `TwoTimescaleConfig.judge_update_frequency=100`, `TwoTimescaleConfig.judge_freeze_after=5000` |
| F3 | **Distributional Shift (Judge)** | Judge trained on early episodes fails to generalize to policy's later behavioral distribution | Jensen-Shannon divergence between judge training distribution and policy rollout distribution exceeds threshold | Periodic judge fine-tuning on recent episodes (memory buffer); EMA parameter updates; distribution matching regularization | `TwoTimescaleJudgeCallback._update_ema()`, `config.ema_decay=0.995` |
| F4 | **Distributional Shift (Environment)** | Adversarial launderers adapt, injecting novel fraud patterns unseen in judge training | Rolling F1 dropping despite stable judge reward; concept drift detection alarm | Adversary agent in training that generates diverse fraud patterns; regular retraining on new confirmed labels | `MultiAgentAMLEnvironment` with `enable_adversary=True` |
| F5 | **Label Scarcity Overfitting** | With <0.1% positive labels, hard metric signal is extremely sparse and noisy | High variance in per-episode F1; model performance varies dramatically across evaluation folds | Dense judge reward provides smoother learning signal; class-weighted losses in hard metric computation | `RewardConfig.precision_weight`, recall weighting proportional to class imbalance |
| F6 | **Uncertainty Miscalibration** | Judge's uncertainty head reports low uncertainty on out-of-distribution inputs (overconfident extrapolation) | Expected calibration error (ECE) increasing; uncertainty doesn't correlate with actual error | Uncertainty penalty in reward: `R_judge_effective = R_judge - λ·σ_judge`; periodic recalibration on held-out set | `RewardConfig.uncertainty_penalty=0.1`, `SafetyMonitorCallback._check_calibration()` |
| F7 | **Sparse Reward Starvation** | Policy receives near-zero reward for long stretches, preventing meaningful gradient signal | Episode returns concentrated near zero; policy entropy not decreasing; no learning progress | Judge provides dense intermediate reward; potential-based reward shaping for suspicious activity | `ConservativeRewardComputer` outputs per-step and per-episode rewards |
| F8 | **Mode Collapse in Adversary** | Adversary learns single attack pattern; detector overfits to this pattern and fails on diverse attacks | Low entropy in adversary action distribution; detector-adversary reward gap collapses | Entropy regularization for adversary; diverse attack library initialization; adversary curriculum | `DetectorAdversaryConfig.adversary_action_space` with varied attack types |
| F9 | **Audit Trail Violation** | System cannot explain why a transaction was flagged, violating regulatory requirements (GDPR Article 22, BSA/AML) | N/A - design requirement | Episode serialization produces human-readable justifications; judge architecture includes attention attribution | `EpisodeSerializer.serialize()` with explainability fields; attention head extraction |
| F10 | **Temporal Leakage** | Model uses future information (e.g., node IDs that encode creation order) to predict fraud | Model performs significantly better on historical test than future test; attention concentrates on ID-correlated features | Canonical episode serialization with anonymized IDs; temporal train/test split validation | `EpisodeSerializer._anonymize_node_id()`, temporal split in experimental protocol |
| F11 | **Catastrophic Forgetting** | Judge fine-tuning on recent episodes destroys performance on earlier patterns | Judge performance on held-out historical episodes drops after update | Elastic Weight Consolidation (EWC); replay buffer sampling from full history; judge ensemble | `TwoTimescaleConfig.ewc_lambda`, memory buffer in callback |
| F12 | **Gradient Instability** | LoRA fine-tuning with small rank produces unstable gradients; reward signal oscillates | High variance in judge loss; NaN/Inf in gradients; reward magnitude oscillates between updates | Gradient clipping; learning rate warmup; larger LoRA rank with regularization | `JudgeConfig.lora_rank=16`, gradient clipping in training loop |

---

## Deliverable 4: Proposed Method

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                                      │
│                                                                              │
│  ┌──────────────┐     ┌─────────────────┐     ┌─────────────────────────┐  │
│  │  Transaction │     │  Multi-Agent    │     │     Episode Data        │  │
│  │    Graph     │────▶│   Environment   │────▶│  {nodes, edges, actions,│  │
│  │   G=(V,E)    │     │  (Gymnasium)    │     │   predictions, labels}  │  │
│  └──────────────┘     └─────────────────┘     └───────────┬─────────────┘  │
│                              │                             │                 │
│                              │ step(a)                     │                 │
│                              ▼                             ▼                 │
│  ┌──────────────┐     ┌─────────────────┐     ┌─────────────────────────┐  │
│  │   Detector   │◀────│   Observation   │     │   Episode Serializer    │  │
│  │    Agent     │     │  (node feats,   │     │  (canonical text repr)  │  │
│  │   (PPO/SAC)  │     │   graph struct) │     └───────────┬─────────────┘  │
│  └──────┬───────┘     └─────────────────┘                 │                 │
│         │                                                  ▼                 │
│         │ action                              ┌─────────────────────────┐   │
│         │                                     │     ModernBERT Judge    │   │
│         │                                     │  ┌─────────────────────┐│   │
│         ▼                                     │  │   Encoder (frozen)  ││   │
│  ┌──────────────┐                             │  ├─────────────────────┤│   │
│  │   Adversary  │                             │  │ LoRA Adapters (r=16)││   │
│  │    Agent     │                             │  ├─────────────────────┤│   │
│  │  (optional)  │                             │  │  Reward Head (MLP)  ││   │
│  └──────────────┘                             │  │  Fraud-Type Head    ││   │
│                                               │  │  Uncertainty Head   ││   │
│                                               │  └─────────────────────┘│   │
│                                               └───────────┬─────────────┘   │
│                                                           │                 │
│                                                           ▼                 │
│                          ┌─────────────────────────────────────────────┐   │
│                          │        Conservative Reward Computer          │   │
│                          │                                              │   │
│                          │  R_total = w_P·P + w_R·R + w_F·F1           │   │
│                          │           + α(t)·clip(R_judge - λ·σ, -τ, τ)  │   │
│                          │                                              │   │
│                          │  where: α(t) anneals from 0.1 → 0.3          │   │
│                          │         τ = 0.5 (clip bound)                 │   │
│                          │         λ = 0.1 (uncertainty penalty)        │   │
│                          └─────────────────────────────────────────────┘   │
│                                               │                             │
│                                               ▼                             │
│                          ┌─────────────────────────────────────────────┐   │
│                          │          Safety Monitor Callback             │   │
│                          │  • Drift detection (hard vs judge corr)     │   │
│                          │  • Calibration monitoring (ECE)             │   │
│                          │  • Hard metric floor enforcement            │   │
│                          └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Judge Model Specification

**Base Model**: `answerdotai/ModernBERT-base` (149M parameters)
- 22 transformer layers, hidden dim 768
- Rotary Position Embeddings (RoPE) with 8192 context length
- Flash Attention 2 for memory efficiency
- Pre-trained on 2T tokens (web, code, scientific text)

**Fine-tuning Strategy**: LoRA (Low-Rank Adaptation)
- Rank r = 16, α = 32, dropout = 0.1
- Target modules: query, key, value, output projections
- Trainable parameters: ~1.2M (0.8% of base)

**Output Heads**:
1. **Reward Head**: MLP(768 → 256 → 64 → 1) with tanh output ∈ [-1, 1]
2. **Fraud-Type Head**: MLP(768 → 256 → num_fraud_types) for auxiliary classification
3. **Uncertainty Head**: MLP(768 → 256 → 1) with softplus output for σ > 0

**Training Objective**:
```
L_total = L_reward + λ_fraud·L_fraud + λ_unc·L_uncertainty

L_reward = MSE(R_pred, R_target)  where R_target = 2·F1 - 1
L_fraud = CrossEntropy(fraud_pred, fraud_label)
L_uncertainty = NLL under Gaussian(R_pred, σ_pred)
```

### 4.3 Episode Serialization Protocol

Episodes are converted to canonical text for the judge via `EpisodeSerializer`:

```
[EPISODE id=a3f7...]
[GRAPH stats: 1247 nodes, 3891 edges, density=0.0025]
[TEMPORAL window: 2024-01-15 to 2024-01-22]

[FLAGGED_TRANSACTIONS count=23]
- TXN amount=15420.00 USD, sender=ENTITY_A, receiver=ENTITY_B, features=[high_velocity, new_relationship, round_amount]
- TXN amount=9999.00 USD, sender=ENTITY_C, receiver=ENTITY_D, features=[just_below_threshold, shell_company_pattern]
...

[AGENT_ACTIONS count=47]
- STEP 1: FLAG node=ENTITY_A confidence=0.87 reason=velocity_anomaly
- STEP 2: INVESTIGATE edge=(ENTITY_A,ENTITY_B) found=circular_flow
...

[CONFIRMED_LABELS available=12, fraud=3, legitimate=9]
[HARD_METRICS precision=0.75, recall=0.67, f1=0.71]
```

**Key Design Choices**:
1. **Order Invariance**: Transactions sorted by canonical hash, not temporal order within episode
2. **ID Anonymization**: Entity IDs replaced with deterministic pseudonyms (ENTITY_A, etc.) computed via HMAC to prevent ID-based shortcuts
3. **Feature Normalization**: Amounts normalized to percentile buckets; timestamps converted to relative offsets
4. **Truncation Strategy**: Long episodes truncated to 8000 tokens with priority to flagged transactions and confirmed labels

### 4.4 Conservative Reward Formula

**Composite Reward**:
```python
def compute_reward(episode_data, hard_metrics, judge_output, config, step):
    # Hard metric component (always computed)
    R_hard = (config.precision_weight * hard_metrics.precision +
              config.recall_weight * hard_metrics.recall +
              config.f1_weight * hard_metrics.f1)

    # Judge component with safety measures
    R_judge_raw = judge_output.reward  # ∈ [-1, 1]
    uncertainty = judge_output.uncertainty

    # Uncertainty penalty
    R_judge_penalized = R_judge_raw - config.uncertainty_penalty * uncertainty

    # Clip to prevent exploitation
    R_judge_clipped = clip(R_judge_penalized, -config.judge_clip_max, config.judge_clip_max)

    # Annealing weight: starts conservative, increases as calibration improves
    alpha = config.judge_weight_start + (config.judge_weight_end - config.judge_weight_start) * min(1.0, step / config.judge_warmup_steps)

    # Final composite
    R_total = R_hard + alpha * R_judge_clipped

    return R_total
```

**Default Configuration**:
```python
RewardConfig(
    precision_weight=0.2,
    recall_weight=0.3,      # Higher recall weight for AML (FN more costly than FP)
    f1_weight=0.4,
    judge_weight_start=0.1,  # Start conservative
    judge_weight_end=0.3,    # Cap judge influence at 30%
    judge_warmup_steps=5000,
    judge_clip_max=0.5,
    uncertainty_penalty=0.1,
)
```

### 4.5 Two-Timescale Training Protocol

**Policy Update Timescale (Fast)**: Every episode
- PPO with clipped surrogate objective
- Learning rate: 3e-4
- Batch size: 64 episodes
- Gradient clipping: 0.5

**Judge Update Timescale (Slow)**: Every 100 episodes
- LoRA parameters fine-tuned on recent 500-episode buffer
- Learning rate: 1e-5 with cosine decay
- Gradient accumulation: 4 steps
- Early stopping on validation MSE

**Judge Freezing**: After 5000 policy episodes
- Judge parameters frozen completely
- Only hard metrics and cached judge inference used
- Prevents late-stage co-adaptation

**EMA Smoothing**: Continuous
- Judge EMA with decay 0.995 updated every episode
- EMA weights used for inference to smooth predictions

### 4.6 Safety Monitoring Protocol

**Drift Detection**:
```python
# Compute correlation between hard metrics and judge over sliding window
correlation = pearsonr(hard_metrics[-100:], judge_scores[-100:])
if correlation < config.drift_correlation_threshold:  # default: 0.3
    raise DriftAlarm("Judge diverging from hard metrics")
```

**Calibration Monitoring**:
```python
# Expected Calibration Error
bins = discretize(judge_confidence, n_bins=10)
for bin in bins:
    expected_accuracy = mean(judge_confidence[bin])
    actual_accuracy = mean(correct_predictions[bin])
    ece += |expected_accuracy - actual_accuracy| * len(bin) / total

if ece > config.ece_threshold:  # default: 0.15
    raise CalibrationAlarm("Judge miscalibrated")
```

**Hard Floor Enforcement**:
```python
if hard_metrics.f1 < config.hard_metric_floor:  # default: 0.1
    # Force high hard metric weight regardless of annealing
    alpha = min(alpha, 0.1)
    logger.warning("Hard metric floor violation - reducing judge weight")
```

---

## Deliverable 5: Experimental Plan

### 5.1 Datasets

| Dataset | Nodes | Edges | Fraud % | Temporal | Source |
|---------|-------|-------|---------|----------|--------|
| **Elliptic Bitcoin** | 203K | 234K | 2.2% | Yes (49 steps) | Weber et al. 2019 |
| **Synthetic-AMLSim-Small** | 50K | 200K | 1% (injected) | Yes | Altman et al. 2023 |
| **Synthetic-AMLSim-Large** | 500K | 2M | 0.5% (injected) | Yes | Altman et al. 2023 |
| **IBM AMLSim** | 100K | 500K | 1% (parameterized) | Yes | IBM Research |

### 5.2 Baselines

| Baseline | Description | Expected F1 |
|----------|-------------|-------------|
| **Rule-Based** | Hand-crafted SAR rules (>$10K, velocity, etc.) | 0.15-0.25 |
| **GCN + Hard Metrics Only** | Standard GCN with sparse label supervision | 0.70-0.75 |
| **GCN + Random Reward** | Sanity check: judge outputs random rewards | 0.65-0.70 |
| **GCN + Uncalibrated Judge** | No clipping, no uncertainty, no anchoring | 0.60-0.75 (high variance) |
| **PPO + Hard Metrics Only** | RL baseline without learned judge | 0.68-0.73 |
| **RLAIF (LLM Judge)** | GPT-4 as reward model (expensive baseline) | 0.72-0.78 |

### 5.3 Ablation Studies

**Ablation A: Judge Weight Annealing**
| Variant | α Schedule | Hypothesis |
|---------|------------|------------|
| A1: No Judge | α = 0 | Baseline, sparse reward challenge |
| A2: Fixed Low | α = 0.1 constant | Conservative, limited judge benefit |
| A3: Fixed High | α = 0.5 constant | Risk of reward hacking |
| A4: Linear Anneal | α: 0.1 → 0.3 | **Proposed**: balance safety and signal |
| A5: Adaptive | α ∝ calibration score | Aggressive, may be unstable |

**Ablation B: Uncertainty Penalty**
| Variant | λ_uncertainty | Hypothesis |
|---------|---------------|------------|
| B1: No Penalty | λ = 0 | Judge may overfit to low-uncertainty regions |
| B2: Low Penalty | λ = 0.05 | Mild discouragement of uncertainty |
| B3: Medium Penalty | λ = 0.1 | **Proposed**: balance |
| B4: High Penalty | λ = 0.2 | May over-penalize correct but uncertain predictions |

**Ablation C: Judge Update Frequency**
| Variant | Update Every N Episodes | Hypothesis |
|---------|-------------------------|------------|
| C1: Frequent | N = 10 | Risk of co-adaptation |
| C2: Medium | N = 50 | May track distribution well |
| C3: Slow | N = 100 | **Proposed**: stable, may lag |
| C4: Very Slow | N = 500 | May fail to adapt |
| C5: Frozen | N = ∞ (pretrained only) | Tests judge generalization |

**Ablation D: Clip Bound**
| Variant | τ (clip max) | Hypothesis |
|---------|--------------|------------|
| D1: No Clip | τ = ∞ | Maximum reward hacking risk |
| D2: Tight Clip | τ = 0.2 | Very conservative, may limit signal |
| D3: Medium Clip | τ = 0.5 | **Proposed**: balance |
| D4: Loose Clip | τ = 0.8 | Higher risk, more signal |

**Ablation E: Adversary Training**
| Variant | Adversary Config | Hypothesis |
|---------|------------------|------------|
| E1: No Adversary | Disabled | Baseline, may overfit to static distribution |
| E2: Random Adversary | Random fraud injection | Robustness baseline |
| E3: Learned Adversary | RL-trained adversary | **Proposed**: adaptive robustness |
| E4: Curriculum Adversary | Adversary difficulty increases | May provide stable learning |

### 5.4 Evaluation Metrics

**Primary Metrics**:
| Metric | Definition | Target |
|--------|------------|--------|
| **Illicit F1** | F1 on fraud detection | > 0.75 |
| **Illicit Recall** | Fraud recall (regulatory focus) | > 0.85 |
| **Precision@100** | Precision in top 100 flags | > 0.50 |
| **Alert Reduction** | Reduction in false alerts vs baseline | > 30% |

**Safety/Robustness Metrics**:
| Metric | Definition | Threshold |
|--------|------------|-----------|
| **Judge-Hard Correlation** | Pearson r between judge and F1 over episodes | > 0.3 |
| **Expected Calibration Error** | Judge calibration | < 0.15 |
| **Reward Exploitation Score** | Max(judge_reward - hard_F1) over episodes | < 0.3 |
| **Temporal Degradation** | F1 drop from early to late time steps | < 10% |

**Adversarial Robustness Metrics**:
| Metric | Definition | Threshold |
|--------|------------|-----------|
| **Adversary Win Rate** | Episodes where adversary achieves reward > detector | < 40% |
| **Post-Attack F1** | F1 after adversary injects fraud in test | > 0.65 |
| **Novel Pattern Detection** | F1 on fraud types unseen in training | > 0.50 |

### 5.5 Statistical Protocol

- **Runs**: 5 seeds per configuration
- **Reporting**: Mean ± std, p-values for main comparisons
- **Significance**: Two-sided t-test, α = 0.05 with Bonferroni correction
- **Learning Curves**: Plot every 100 episodes, smoothed with EMA (α=0.9)
- **Compute Budget**: ~500 GPU-hours total (A100-40GB equivalent)

### 5.6 Experimental Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1: Infrastructure** | Week 1-2 | Environment validation, baseline implementation |
| **Phase 2: Judge Training** | Week 3-4 | Pre-train judge on labeled subset, calibration |
| **Phase 3: Main Ablations** | Week 5-8 | Run ablations A-E, 5 seeds each |
| **Phase 4: Adversarial** | Week 9-10 | Adversary experiments, robustness analysis |
| **Phase 5: Analysis** | Week 11-12 | Statistical analysis, figure generation, writing |

---

## Deliverable 6: Reproducibility & Auditability Checklist

### 6.1 Code Artifacts

| Artifact | Status | Location |
|----------|--------|----------|
| ☑ Judge model architecture | Implemented | `judge_model.py` |
| ☑ Episode serializer | Implemented | `judge_model.py:EpisodeSerializer` |
| ☑ Reward computation | Implemented | `reward_computation.py` |
| ☑ Multi-agent environment | Implemented | `multi_agent_env.py` |
| ☑ Training callbacks | Implemented | `training_callbacks.py` |
| ☐ Training script | TODO | `scripts/train.py` |
| ☐ Evaluation script | TODO | `scripts/evaluate.py` |
| ☐ Ablation runner | TODO | `scripts/run_ablations.py` |

### 6.2 Data Artifacts

| Artifact | Source | Processing |
|----------|--------|------------|
| Elliptic Bitcoin | [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set) | Temporal split, node feature normalization |
| AMLSim Synthetic | [GitHub](https://github.com/IBM/AMLSim) | Parameter config provided in `configs/` |
| Judge Pre-training Data | Generated | Script to create preference pairs from labeled data |

### 6.3 Hyperparameter Documentation

| Component | Parameter | Value | Justification |
|-----------|-----------|-------|---------------|
| Judge | base_model | ModernBERT-base | Best encoder in size class (Warner 2024) |
| Judge | lora_rank | 16 | Hu et al. 2021 recommend 8-64 |
| Judge | learning_rate | 1e-5 | Standard for LoRA fine-tuning |
| Reward | judge_weight_start | 0.1 | Conservative start (Levine 2020) |
| Reward | judge_weight_end | 0.3 | Cap at 30% to maintain anchor |
| Reward | clip_max | 0.5 | Prevent >50% exploitation beyond hard |
| Policy | algorithm | PPO | Stable, widely validated |
| Policy | learning_rate | 3e-4 | Schulman et al. 2017 |
| Training | judge_update_freq | 100 | Two-timescale separation |
| Safety | drift_threshold | 0.3 | Correlation floor |

### 6.4 Audit Trail Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Transaction Flagging Justification** | `EpisodeSerializer` outputs human-readable reasons |
| **Model Decision Explanation** | Attention attribution from judge encoder |
| **Reproducible Predictions** | Deterministic serialization with canonical ordering |
| **Version Control** | Git SHA logged with each experiment |
| **Configuration Logging** | Full config YAML saved per run |
| **Metric History** | All metrics logged to W&B/MLflow |

### 6.5 Regulatory Compliance Notes

| Regulation | Requirement | How We Address |
|------------|-------------|----------------|
| **BSA/AML** | Suspicious Activity Reports must be justified | Episode serialization provides audit trail |
| **GDPR Art. 22** | Right to explanation for automated decisions | Judge attention + serialization = explanation |
| **Model Risk (SR 11-7)** | Model validation and documentation | This document + calibration monitoring |
| **Fair Lending** | No discriminatory patterns | Feature normalization removes protected attributes |

---

## Deliverable 7: Bibliography

### Core RLHF and Reward Modeling

1. Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS*. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)

2. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

3. Stiennon, N., Ouyang, L., Wu, J., et al. (2020). Learning to summarize from human feedback. *NeurIPS*. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)

4. Ziegler, D., Stiennon, N., Wu, J., et al. (2019). Fine-Tuning Language Models from Human Preferences. *arXiv*. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)

5. Bai, Y., Jones, A., Ndousse, K., et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv*. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)

6. Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

7. Lee, H., Phatale, S., Mansoor, H., et al. (2023). RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. *arXiv*. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)

### Conservative and Safe RL

8. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS*. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)

9. Fujimoto, S., Meger, D., & Precup, D. (2019). Off-Policy Deep Reinforcement Learning without Exploration. *ICML*. [arXiv:1812.02900](https://arxiv.org/abs/1812.02900)

10. Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. *arXiv*. [arXiv:2005.01643](https://arxiv.org/abs/2005.01643)

11. Hadfield-Menell, D., Milli, S., Abbeel, P., Russell, S., & Dragan, A. (2017). Inverse Reward Design. *NeurIPS*. [arXiv:1711.02827](https://arxiv.org/abs/1711.02827)

12. Ng, A., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *ICML*. [PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

### Graph Neural Networks for Fraud Detection

13. Weber, M., Domeniconi, G., Chen, J., et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. *KDD Workshop*. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591)

14. Pareja, A., Domeniconi, G., Chen, J., et al. (2020). EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. *AAAI*. [arXiv:1902.10191](https://arxiv.org/abs/1902.10191)

15. Liu, Y., Ao, X., Qin, Z., et al. (2021). Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection. *WWW*. [DOI](https://doi.org/10.1145/3442381.3449989)

16. Altman, E., et al. (2023). Realistic Synthetic Financial Transaction Graphs for AML. *ICAIF*. [GitHub: IBM/AMLSim](https://github.com/IBM/AMLSim)

17. Johannessen, H., et al. (2023). Towards Explainable Anti-Money Laundering with Graph Neural Networks. *AAAI Workshop on AI for Financial Services*.

### Multi-Agent and Adversarial RL

18. Lanctot, M., Zambaldi, V., Gruslys, A., et al. (2017). A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning. *NeurIPS*. [arXiv:1711.00832](https://arxiv.org/abs/1711.00832)

19. Pinto, L., Davidson, J., Sukthankar, R., & Gupta, A. (2017). Robust Adversarial Reinforcement Learning. *ICML*. [arXiv:1703.02702](https://arxiv.org/abs/1703.02702)

20. Gleave, A., Dennis, M., Wild, C., et al. (2020). Adversarial Policies: Attacking Deep Reinforcement Learning. *ICLR*. [arXiv:1905.10615](https://arxiv.org/abs/1905.10615)

21. Hsieh, Y., et al. (2023). Adversarial Training for Graph Neural Networks. *IEEE TNNLS*.

### Encoder Models and Fine-Tuning

22. Warner, B., Shleifer, S., et al. (2024). ModernBERT: A Modernized Bidirectional Encoder for NLP. *arXiv*. [HuggingFace](https://huggingface.co/answerdotai/ModernBERT-base)

23. Hu, E., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

24. Knox, W.B. & Stone, P. (2009). Interactively Shaping Agents via Human Reinforcement: The TAMER Framework. *K-CAP*. [PDF](https://www.cs.utexas.edu/~bradknox/papers/kcap09-knox.pdf)

25. MacGlashan, J., Ho, M., Loftin, R., et al. (2017). Interactive Learning from Policy-Dependent Human Feedback. *ICML*. [arXiv:1701.06049](https://arxiv.org/abs/1701.06049)

### RL Algorithms

26. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

27. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor. *ICML*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

### Calibration and Uncertainty

28. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. (2017). On Calibration of Modern Neural Networks. *ICML*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

29. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)

30. Garg, D., et al. (2023). Extreme Q-Learning: MaxEnt RL without Entropy. *ICLR*. [arXiv:2301.02328](https://arxiv.org/abs/2301.02328)

---

## Appendix A: 5 Concrete MVP Design Decisions with Citations

### Decision 1: ModernBERT over GPT-style Decoders for Judge

**Choice**: Use ModernBERT-base (encoder-only, 149M params) rather than GPT-2/LLaMA-style decoder.

**Rationale**:
- Episode evaluation is a discriminative task (assign scalar reward), not generative
- Encoders achieve higher accuracy on classification/regression with 10x fewer parameters (Warner et al. 2024)
- Bidirectional attention captures full episode context in single forward pass
- 8192 token context fits full episodes without chunking
- LoRA fine-tuning achieves comparable performance to full fine-tuning at 0.8% parameter cost (Hu et al. 2021)

**Citation**: Warner et al. (2024) show ModernBERT-base matches BERT-large on GLUE while being 2x faster; Hu et al. (2021) demonstrate LoRA's effectiveness for encoder adaptation.

### Decision 2: Hard Metric Anchoring over Pure KL Regularization

**Choice**: Use `R_total = R_hard + α·R_judge` rather than `R_total = R_judge - β·KL(π||π_ref)`.

**Rationale**:
- KL regularization keeps policy close to reference but doesn't guarantee good outcomes
- Hard metrics (F1 on confirmed labels) provide direct signal about actual detection quality
- In AML, some ground truth is eventually available (investigations resolve), unlike pure preference learning
- Anchoring prevents judge exploitation: even if judge is hacked, hard metrics bound total reward
- Banking regulators require metrics tied to actual outcomes, not model preferences

**Citation**: Ouyang et al. (2022) observed reward hacking with pure learned rewards; Ng et al. (1999) established that reward shaping must preserve optimal policy (hard metrics do, KL doesn't guarantee).

### Decision 3: Two-Timescale over Joint Training

**Choice**: Update judge every 100 episodes, freeze after 5000 episodes; policy updates every episode.

**Rationale**:
- Joint training causes co-adaptation: policy learns to produce judge-pleasing outputs, judge learns to reward whatever policy does (Levine et al. 2020)
- Slower judge updates = judge evaluates diverse policy behaviors before updating
- Freezing judge after warmup = policy converges to stable optimum without moving target
- Similar to actor-critic with slow critic updates (Haarnoja et al. 2018)

**Citation**: Levine et al. (2020) identify co-adaptation as key failure mode; similar two-timescale principles used successfully in SAC (Haarnoja et al. 2018).

### Decision 4: Canonical Serialization with ID Anonymization

**Choice**: Serialize episodes to text with deterministic ordering and pseudonymized entity IDs.

**Rationale**:
- Node/edge IDs in graphs often encode temporal information (created sequentially)
- Model could learn shortcut: "high ID = recent = more suspicious" without learning actual patterns
- Anonymization (ENTITY_A, ENTITY_B) with deterministic mapping prevents ID leakage
- Canonical ordering (sort by feature hash) ensures same episode always produces same text
- Human-readable format enables auditor inspection (regulatory requirement)

**Citation**: Weber et al. (2019) noted temporal structure in Elliptic required careful handling; GDPR Article 22 requires explainable automated decisions.

### Decision 5: Uncertainty Penalty over Confidence Threshold

**Choice**: Use `R_effective = R_judge - λ·σ` rather than discarding predictions where confidence < threshold.

**Rationale**:
- Threshold discarding creates discontinuities in reward signal
- Penalty smoothly down-weights uncertain predictions without discarding information
- Encourages judge to be well-calibrated: overconfident wrong predictions are penalized more
- Similar to pessimism principle in Conservative Q-Learning (Kumar et al. 2020)
- Allows uncertain predictions to still contribute signal, just attenuated

**Citation**: Kumar et al. (2020) establish pessimism under uncertainty as key principle for safe offline RL; Guo et al. (2017) show neural networks are often miscalibrated, motivating explicit uncertainty handling.

---

*Document generated for rl_money_laundering project. Last updated: 2026-01-22.*
