Project Roadmap
===============

.. admonition:: Version 1.0 -- January 2026
   :class: note

   Multi-Agent Reinforcement Learning for Anti-Money Laundering Detection

.. contents:: Table of Contents
   :local:
   :depth: 3

Executive Summary
-----------------

.. admonition:: Project Vision
   :class: tip

   Build a **multi-agent reinforcement learning system** for anti-money laundering detection that combines:

   - Graph Neural Networks (GNN) for transaction network analysis
   - LLM-based "Judge" agent for semantic reasoning on textual evidence
   - RL agent for optimal investigation resource allocation
   - Domain-adapted language models via continuous pretraining

Current State
~~~~~~~~~~~~~

.. list-table:: Project Component Status Overview
   :header-rows: 1
   :widths: 40 30 30

   * - Component
     - Status
     - Completion
   * - Core RL Environment
     - ✅ Done
     - 100%
   * - GNN Encoder (RMGANets)
     - ✅ Done
     - 100%
   * - Multi-Agent Framework
     - ✅ Done
     - 100%
   * - Data Preparation Module
     - ✅ Done
     - 95%
   * - Continuous Pretraining
     - ✅ Done
     - 95%
   * - Supervised Judge Head
     - ✅ Done
     - 100%
   * - Evaluation Harness
     - ⏳ Todo
     - 0%
   * - Production Deployment
     - ⏳ Todo
     - 0%

Key Deliverables
~~~~~~~~~~~~~~~~

1. **Research Paper**: Novel multi-agent RL approach with LLM judge for AML
2. **Production System**: Deployable in regulated banking environments
3. **Open-Source Framework**: Reproducible with public-only data sources
4. **Benchmark Dataset**: Curated AML corpus with evaluation metrics

Architecture Overview
---------------------

System Architecture
~~~~~~~~~~~~~~~~~~~

.. mermaid::

   flowchart TB
       subgraph Data["Data Layer"]
           DS[Data Sources<br/>FinCEN, FATF, SEC, ...]
           CB[Corpus Builder<br/>Dedup, PII, Mix]
           PT[Pretraining<br/>DAPT + TAPT]
       end

       subgraph Models["Model Layer"]
           GNN[GNN Encoder<br/>RMGANets]
           Judge[LLM Judge<br/>Calibrated]
           RL[RL Agent<br/>PPO/DQN]
       end

       subgraph Integration["Integration Layer"]
           ENV[Multi-Agent Environment]
           RW[Reward Integrator]
       end

       EVAL[Evaluation Harness]

       DS --> CB --> PT --> Judge
       GNN --> ENV
       Judge --> ENV
       ENV --> RL --> RW --> ENV
       ENV --> EVAL

Module Dependencies
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Module
     - Location
     - Dependencies
   * - ``dataprep``
     - ``src/.../dataprep/``
     - requests, bson, lzma
   * - ``pretraining``
     - ``src/.../pretraining/``
     - transformers, peft, bitsandbytes
   * - ``multiagent``
     - ``src/.../multiagent/``
     - ray[rllib], torch
   * - ``gnn_modules``
     - ``src/.../gnn_modules/``
     - torch_geometric
   * - ``evaluation``
     - ``src/.../evaluation/``
     - sklearn, scipy

Phase 1: Data Infrastructure (Weeks 1-3)
----------------------------------------

**Objective**: Build a clean, compliant, deduplicated corpus for domain-adaptive pretraining.

1.1 Compliance Mode Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Production Blocker
   :class: danger

   **Problem**: ICIJ FinCEN Files are leaked SARs -- governance landmine in banking.

   **Solution**: Introduce ``--compliance_mode=strict`` flag.

**Scope:**

- Add ``ComplianceMode`` enum: ``RESEARCH``, ``STRICT``, ``PUBLIC_ONLY``
- ``STRICT`` mode auto-disables: ``FinCENLoader``, ``VenmoNotesLoader``
- Add replacement loaders for public sources:

  - FinCEN Advisories (public)
  - FATF/APG/Egmont typology reports
  - DOJ/SEC enforcement actions

- Audit trail: log which sources were enabled/disabled

**Files to modify:**

- ``dataprep/pipeline.py``: Add ``compliance_mode`` to ``PipelineConfig``
- ``dataprep/base.py``: Add ``ComplianceMode`` enum
- New: ``dataprep/fincen_advisory_loader.py``
- New: ``dataprep/enforcement_loader.py``

**Estimated effort**: 2 days

1.2 Deduplication Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Data Quality
   :class: danger

   **Problem**: GDELT has massive duplication (press syndication, near-identical headlines).

   **Solution**: MinHash/SimHash near-duplicate removal.

**Scope:**

- Exact hash deduplication (SHA256 on normalized text)
- Near-duplicate detection using MinHash LSH
- Configurable similarity threshold (default: 0.85)
- Document length thresholds:

  - Drop documents < 50 tokens
  - Drop documents > 10,000 tokens (likely OCR errors)

- Statistics reporting: duplicates removed per source

**Algorithm**:

.. math::

   \text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|} \approx \frac{\text{matching MinHash signatures}}{\text{total signatures}}

**Files to create:**

- New: ``dataprep/dedup.py``: ``ExactDedup``, ``MinHashDedup``, ``DedupPipeline``
- Modify: ``dataprep/corpus_builder.py``: Integrate dedup step

**Dependencies**: ``datasketch`` (MinHash LSH implementation)

**Estimated effort**: 2 days

1.3 Boilerplate Stripping
~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Noise Reduction
   :class: warning

   **Problem**: SEC EDGAR has headers, HTML junk, signatures that pollute training.

   **Solution**: Source-specific boilerplate removal.

**Scope:**

- **SEC EDGAR**:

  - Remove HTML tags and entities
  - Strip filing headers (ACCESSION NUMBER, CONFORMED SUBMISSION TYPE, etc.)
  - Remove table of contents sections
  - Strip exhibit references

- **GDELT/News**:

  - Remove "Subscribe to newsletter" CTAs
  - Strip social media share buttons text
  - Remove cookie consent boilerplate

- **General**:

  - Email signatures
  - Legal disclaimers (configurable: keep for some sources)
  - Repeated whitespace normalization

**Files to create:**

- New: ``dataprep/cleaning.py``: ``BoilerplateStripper``, ``TextNormalizer``

**Estimated effort**: 1.5 days

1.4 OpenSanctions Capping
~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Distribution Skew
   :class: warning

   **Problem**: 500K short entity descriptions will dominate token distribution.

   **Solution**: Cap contribution + enrich templates.

**Scope:**

- Cap OpenSanctions at 5-10% of total training tokens
- Template enrichment: convert entity records to richer sentences

.. code-block:: text

   Before: "Entity: John Doe | Type: Person | Country: RU"
   After:  "John Doe is a sanctioned individual from Russia,
            listed under OFAC SDN for involvement in..."

**Files to modify:**

- ``dataprep/opensanctions_loader.py``: Add ``template_mode``
- ``dataprep/corpus_builder.py``: Add per-source token caps

**Estimated effort**: 1 day

1.5 Enhanced Metadata Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scope:** Add required fields for audit trail and reproducibility.

**New JSONL schema:**

.. code-block:: json

   {
     "id": "uuid-v4",
     "text": "...",
     "source": "fincen|fatf|sec|gdelt|opensanctions",
     "category": "sar_narrative|typology_case|enforcement|...",
     "lang": "en",
     "timestamp": "2024-01-15T10:30:00Z",
     "labels": {
       "typology": ["layering", "structuring"],
       "risk_level": "high",
       "entities": ["Bank A", "Company B"]
     },
     "provenance": {
       "url": "https://...",
       "license": "CC-BY-4.0|public-domain|restricted",
       "access_date": "2024-01-15",
       "content_hash": "sha256:abc123..."
     },
     "processing": {
       "dedup_hash": "minhash:...",
       "pii_scrubbed": true,
       "boilerplate_removed": true
     }
   }

**Estimated effort**: 1 day

1.6 PII Scrubbing
~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Compliance Requirement
   :class: warning

   **Problem**: Training data may contain real names, addresses, account numbers.

   **Solution**: Entity recognition + redaction.

**Scope:**

- Named Entity Recognition for:

  - Person names → ``[PERSON]``
  - Phone numbers → ``[PHONE]``
  - Email addresses → ``[EMAIL]``
  - Physical addresses → ``[ADDRESS]``
  - Account/card numbers → ``[ACCOUNT]``
  - SSN/Tax IDs → ``[ID]``

- Configurable: full redaction vs. entity-type replacement
- Preserve entity boundaries for downstream tasks

**Implementation options:**

1. Microsoft Presidio (recommended for compliance)
2. spaCy NER + regex patterns
3. Custom fine-tuned NER model

**Files to create:**

- New: ``dataprep/pii_scrubber.py``

**Dependencies**: ``presidio-analyzer``, ``presidio-anonymizer``

**Estimated effort**: 2 days

Phase 1 Summary
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 15 20

   * - Task
     - Priority
     - Effort
     - Week
     - Status
   * - 1.1 Compliance Mode
     - P0
     - 2d
     - 1
     - ✅ Done
   * - 1.2 Deduplication
     - P0
     - 2d
     - 1
     - ✅ Done
   * - 1.3 Boilerplate Stripping
     - P1
     - 1.5d
     - 2
     - ✅ Done
   * - 1.4 OpenSanctions Capping
     - P1
     - 1d
     - 2
     - ⏳ Todo
   * - 1.5 Metadata Schema
     - P1
     - 1d
     - 2
     - ⏳ Todo
   * - 1.6 PII Scrubbing
     - P1
     - 2d
     - 3
     - ✅ Done
   * - **Total**
     -
     - **9.5d**
     -
     -

Phase 2: Training Infrastructure (Weeks 3-5)
--------------------------------------------

**Objective**: Robust pretraining pipeline with overfitting prevention and IO optimization.

2.1 TAPT Overfitting Prevention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Model Quality
   :class: danger

   **Problem**: TAPT corpus is only 1.7M tokens -- high overfitting risk.

   **Solution**: Regularization + early stopping + span masking.

**Scope:**

- Lower learning rate: :math:`1 \times 10^{-5}` (was :math:`2 \times 10^{-5}`)
- Increase weight decay: 0.05 (was 0.01)
- Early stopping on validation perplexity (patience=3)
- Implement span masking (T5-style):

.. math::

   \text{Span Masking}: \text{mask } k \text{ consecutive tokens where } k \sim \text{Geometric}(\mu=3)

**Benefits of span masking**:

- Forces model to learn phrase-level semantics
- More robust than random token masking
- Better for downstream classification tasks

**Files to modify:**

- ``pretraining/scripts.py``: Update TAPT defaults
- ``pretraining/collators.py``: Add ``SpanMLMCollator``
- ``pretraining/trainer.py``: Add early stopping logic

**Estimated effort**: 1.5 days

2.2 Streaming & Pre-tokenization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Training Efficiency
   :class: warning

   **Problem**: Large corpus + Python dataloader = IO bottleneck.

   **Solution**: Pre-tokenize to disk + streaming dataset.

**Scope:**

- Pre-tokenization script:

  - Tokenize entire corpus offline
  - Save as memory-mapped numpy arrays
  - Include attention masks and special tokens

- Streaming dataset:

  - Read from disk in chunks
  - Shuffle buffer for randomization
  - Multi-worker prefetching

- Packing optimization:

  - Concatenate short documents to fill context window
  - Reduce padding waste

**Expected speedup**: 2-3x on large corpora

**Files to create:**

- New: ``pretraining/pretokenize.py``
- Modify: ``pretraining/collators.py``: Add ``PackedDataset``

**Estimated effort**: 2 days

2.3 Supervised Judge Head
~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Judge Behavior
   :class: danger

   **Problem**: MLM makes embeddings good, but judge needs *decision behavior*.

   **Solution**: Post-TAPT supervised fine-tuning on AML tasks.

This is the **most important upgrade** for production quality.

Task 1: Typology Classification (Multi-label)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Classes: layering, structuring, shell_company, trade_based, crypto, pep, etc.
- Input: Transaction description / SAR narrative
- Output: Multi-hot vector of applicable typologies
- Loss: Binary cross-entropy with class weights

Task 2: Suspiciousness Scoring (Binary/Regression)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Input: Transaction description
- Output: Score ∈ [0, 1] indicating suspiciousness
- Can be trained as binary classification or regression
- Calibrate using temperature scaling

Task 3: Evidence Sentence Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Input: Document with multiple sentences
- Output: Which sentences are "red flags"
- Token classification (BIO tagging) or sentence classification
- Critical for explainability

Task 4: Pairwise Ranking
^^^^^^^^^^^^^^^^^^^^^^^^

- Input: Two cases (A, B)
- Output: Which is more suspicious
- Trained with margin ranking loss
- Enables prioritization of investigation queue

**Architecture**:

.. code-block:: text

   [Pretrained Encoder] -> [Task-Specific Head] -> [Output]
           |                      |
      Frozen/LoRA            Trainable

**Files to create:**

- New: ``pretraining/supervised_heads.py``
- New: ``pretraining/judge_finetuning.py``
- New: ``pretraining/ranking_loss.py``

**Training data requirements:**

.. list-table::
   :header-rows: 1

   * - Task
     - Min Samples
     - Source
   * - Typology Classification
     - 5,000
     - FATF cases + FinCEN
   * - Suspiciousness Scoring
     - 10,000
     - Labeled SAR excerpts
   * - Evidence Selection
     - 2,000
     - Annotated documents
   * - Pairwise Ranking
     - 5,000 pairs
     - Generated from above

**Estimated effort**: 4 days

Phase 2 Summary
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Task
     - Priority
     - Effort
     - Week
     - Status
   * - 2.1 TAPT Overfitting Prevention
     - P0
     - 1.5d
     - 3
     - ✅ Done
   * - 2.2 Streaming + Pre-tokenization
     - P1
     - 2d
     - 4
     - ✅ Done
   * - 2.3 Supervised Judge Head
     - P0
     - 4d
     - 4-5
     - ✅ Done
   * - **Total**
     -
     - **7.5d**
     -
     -

Phase 3: RL Integration (Weeks 5-7)
-----------------------------------

**Objective**: Fast, stable, calibrated judge integration with RL training loop.

3.1 Batched Judge Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Training Speed
   :class: danger

   **Problem**: Per-step judge inference creates Python bottleneck.

   **Solution**: GPU micro-batching with async pipeline.

**Scope:**

- Batch multiple RL environment steps before judge inference
- GPU-resident inference (avoid CPU-GPU transfers)
- Async inference pipeline:

  - RL step n uses judge scores from step n-1
  - 1-step delay acceptable for stability

- TorchScript/ONNX export for faster inference

**Target throughput**: 1000+ samples/second on V100

**Files to modify:**

- ``multiagent/judge_model.py``: Add batched inference
- ``multiagent/reward_computation.py``: Async scoring

**Estimated effort**: 2 days

3.2 Judge Calibration
~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Reward Stability
   :class: danger

   **Problem**: Uncalibrated confidence = reward spikes = unstable RL.

   **Solution**: Temperature scaling + isotonic regression.

Temperature Scaling
^^^^^^^^^^^^^^^^^^^

Find optimal temperature T that minimizes calibration error:

.. math::

   \hat{p} = \sigma\left(\frac{z}{T}\right)

where z is the logit and σ is the sigmoid function.

Isotonic Regression
^^^^^^^^^^^^^^^^^^^

Non-parametric calibration for more flexibility:

.. math::

   \hat{p} = f(p) \text{ where } f \text{ is monotonically increasing}

Calibration Metrics
^^^^^^^^^^^^^^^^^^^

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability diagrams

**Files to create:**

- New: ``multiagent/calibration.py``
- Modify: ``multiagent/judge_model.py``: Add calibration layer

**Estimated effort**: 2 days

3.3 Reward Smoothing
~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: RL Stability
   :class: warning

   **Problem**: Noisy rewards slow convergence.

   **Solution**: Clipping + EMA + baseline subtraction.

**Scope:**

- Reward clipping: :math:`r \in [-R_{max}, R_{max}]`
- Exponential moving average:

  .. math::

     \bar{r}_t = \alpha \cdot r_t + (1-\alpha) \cdot \bar{r}_{t-1}

- Baseline subtraction:

  .. math::

     r'_t = r_t - V(s_t)

- Reward normalization (running mean/std)

**Files to modify:**

- ``multiagent/reward_computation.py``: Add smoothing
- ``multiagent/integration/reward_integrator.py``: Update interface

**Estimated effort**: 1 day

3.4 Multi-Agent Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scope:**

- GNN agent: Proposes suspicious nodes/edges
- Judge agent: Scores proposals with reasoning
- RL agent: Decides investigation actions
- Communication protocol between agents
- Shared vs. independent replay buffers

**Training modes:**

1. **Centralized**: Single policy, shared observations
2. **Independent**: Each agent learns separately
3. **CTDE**: Centralized training, decentralized execution

**Files to modify:**

- ``multiagent/multi_agent_env.py``: Coordination logic
- ``multiagent/training_callbacks.py``: Multi-agent metrics

**Estimated effort**: 3 days

Phase 3 Summary
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Task
     - Priority
     - Effort
     - Week
     - Status
   * - 3.1 Batched Judge Inference
     - P0
     - 2d
     - 5
     - ✅ Done
   * - 3.2 Judge Calibration
     - P0
     - 2d
     - 5-6
     - ✅ Done
   * - 3.3 Reward Smoothing
     - P1
     - 1d
     - 6
     - ✅ Done
   * - 3.4 Multi-Agent Coordination
     - P1
     - 3d
     - 6-7
     - ✅ Done
   * - **Total**
     -
     - **8d**
     -
     -

Phase 4: Evaluation Harness (Weeks 7-9)
---------------------------------------

**Objective**: Comprehensive evaluation framework for paper and production.

4.1 Core Metrics Module
~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Paper Requirement
   :class: danger

   Metrics needed for publication and production monitoring.

Classification Metrics
^^^^^^^^^^^^^^^^^^^^^^

- Typology: Macro-F1, Micro-F1, per-class precision/recall
- Suspiciousness: AUC-ROC, AUC-PR, F1 at various thresholds
- Multi-label: Hamming loss, subset accuracy

Ranking Metrics
^^^^^^^^^^^^^^^

- NDCG@k (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Precision@k

Calibration Metrics
^^^^^^^^^^^^^^^^^^^

- ECE (Expected Calibration Error)
- Brier Score
- Reliability diagrams

RL-Specific Metrics
^^^^^^^^^^^^^^^^^^^

- Episode return (mean, std)
- Investigation efficiency (fraud caught / resources spent)
- Alert-to-SAR ratio
- False positive rate at fixed recall

**Files to create:**

- New: ``evaluation/metrics.py``
- New: ``evaluation/calibration_metrics.py``
- New: ``evaluation/rl_metrics.py``

**Estimated effort**: 2 days

4.2 Robustness Tests
~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Generalization
   :class: warning

   Show model works beyond training distribution.

**Scope:**

- **Noise injection**: Random character swaps, typos
- **Truncation**: Cut text at various points
- **Alias variants**: Name variations (John Smith → J. Smith → SMITH, JOHN)
- **Language mixing**: Code-switching, non-English terms
- **Temporal shift**: Test on newer data than training
- **Domain shift**: Different AML typologies not in training

**Files to create:**

- New: ``evaluation/robustness.py``
- New: ``evaluation/perturbations.py``

**Estimated effort**: 2 days

4.3 Ablation Framework
~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Paper Contribution
   :class: warning

   Systematic ablations to demonstrate each component's value.

**Required ablations:**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Ablation
     - Purpose
   * - Base vs +DAPT vs +DAPT+TAPT
     - Show value of domain adaptation
   * - Remove GDELT
     - Is news helpful?
   * - Remove OpenSanctions
     - Is entity data helpful?
   * - Remove SEC
     - Is fraud case text helpful?
   * - Weight sensitivity (FinCEN 2.0→1.0)
     - How important is SAR weighting?
   * - MLM vs Span Masking
     - Which objective is better?
   * - With vs without calibration
     - Calibration impact on RL
   * - GNN-only vs +Judge
     - Is LLM judge worth the cost?

**Files to create:**

- New: ``evaluation/ablations.py``
- New: ``scripts/run_ablations.py``

**Estimated effort**: 3 days

4.4 Benchmark Dataset
~~~~~~~~~~~~~~~~~~~~~

**Scope:**

- Curated test set with human annotations
- Multiple difficulty levels (easy/medium/hard)
- Balanced across typologies
- Held-out from training entirely

**Dataset splits:**

.. list-table::
   :header-rows: 1

   * - Split
     - Samples
     - Use
     - Public?
   * - Train
     - 80%
     - Model training
     - Yes
   * - Validation
     - 10%
     - Hyperparameter tuning
     - Yes
   * - Test
     - 10%
     - Final evaluation
     - Yes
   * - Challenge
     - 500
     - Hard cases for leaderboard
     - No (hidden)

**Estimated effort**: 2 days

Phase 4 Summary
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Task
     - Priority
     - Effort
     - Week
     - Status
   * - 4.1 Core Metrics Module
     - P0
     - 2d
     - 7
     - ⏳ Todo
   * - 4.2 Robustness Tests
     - P1
     - 2d
     - 8
     - ⏳ Todo
   * - 4.3 Ablation Framework
     - P1
     - 3d
     - 8-9
     - ⏳ Todo
   * - 4.4 Benchmark Dataset
     - P1
     - 2d
     - 9
     - ⏳ Todo
   * - **Total**
     -
     - **9d**
     -
     -

Phase 5: Production Readiness (Weeks 9-12)
------------------------------------------

**Objective**: Deployable system meeting banking compliance requirements.

5.1 Compliance-Safe Corpus Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P0 Critical: Bank Deployment
   :class: danger

   Reproducible results using only public data sources.

**Public-only sources:**

- FATF/APG/Egmont typology reports (public PDF)
- FinCEN advisories (public)
- DOJ/SEC enforcement actions (public)
- OpenSanctions (CC-BY licensed)
- Academic papers on AML (public)

**Validation:**

- Single-command corpus rebuild
- Checksum verification
- License audit trail
- Documented data provenance

**Estimated effort**: 2 days

5.2 Model Serving Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scope:**

- REST API for judge inference
- Batch scoring endpoint
- Model versioning and A/B testing
- Monitoring and alerting
- Horizontal scaling

**Technology stack:**

- FastAPI or Flask for API
- Triton Inference Server for GPU serving
- Kubernetes for orchestration
- Prometheus/Grafana for monitoring

**Estimated effort**: 5 days

5.3 Explainability Module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: P1 High: Regulatory Requirement
   :class: warning

   Banks must explain why a transaction was flagged.

**Scope:**

- Attention visualization
- SHAP/LIME feature importance
- Evidence sentence highlighting
- Natural language explanations
- Counterfactual explanations ("what would make this not suspicious?")

**Estimated effort**: 4 days

5.4 Audit Logging
~~~~~~~~~~~~~~~~~

**Scope:**

- Log all model predictions
- Log input data and features
- Log model version used
- Tamper-proof storage
- Retention policy compliance

**Estimated effort**: 2 days

Phase 5 Summary
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Task
     - Priority
     - Effort
     - Week
     - Status
   * - 5.1 Compliance-Safe Corpus
     - P0
     - 2d
     - 9-10
     - ⏳ Todo
   * - 5.2 Model Serving
     - P1
     - 5d
     - 10-11
     - ⏳ Todo
   * - 5.3 Explainability
     - P1
     - 4d
     - 11-12
     - ⏳ Todo
   * - 5.4 Audit Logging
     - P1
     - 2d
     - 12
     - ⏳ Todo
   * - **Total**
     -
     - **13d**
     -
     -

Resource Estimates
------------------

Compute Requirements
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Task
     - GPU
     - Time
     - Cost (Cloud)
   * - DAPT (base corpus)
     - V100 32GB
     - 5-8 hrs
     - $15-25
   * - DAPT (extended)
     - A100 40GB
     - 10-15 hrs
     - $35-55
   * - TAPT
     - V100 32GB
     - 30-60 min
     - $2-4
   * - Supervised heads
     - V100 32GB
     - 2-4 hrs
     - $6-12
   * - RL training
     - V100 32GB
     - 10-20 hrs
     - $30-60
   * - Ablations (8 runs)
     - A100 40GB
     - 40-80 hrs
     - $150-300
   * - **Total estimate**
     -
     -
     - **$250-500**

Storage Requirements
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Component
     - Size
   * - Raw data sources
     - 10 GB
   * - Processed corpus
     - 5 GB
   * - Pre-tokenized cache
     - 15 GB
   * - Model checkpoints (all)
     - 50 GB
   * - Logs and metrics
     - 5 GB
   * - **Total**
     - **85 GB**

Human Resources
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Phase
     - Person-Days
     - Skills Needed
   * - Phase 1: Data
     - 9.5
     - Data engineering, NLP
   * - Phase 2: Training
     - 7.5
     - ML engineering, PyTorch
   * - Phase 3: RL Integration
     - 8
     - RL, multi-agent systems
   * - Phase 4: Evaluation
     - 9
     - ML evaluation, statistics
   * - Phase 5: Production
     - 13
     - MLOps, DevOps
   * - **Total**
     - **47 days**
     -

Timeline
--------

Key Milestones
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Week
     - Milestone
     - Deliverable
   * - 3
     - Corpus Ready
     - Clean, deduplicated, compliant corpus
   * - 5
     - Judge Ready
     - Calibrated judge with supervised heads
   * - 7
     - RL Integration
     - Stable multi-agent training
   * - 9
     - Paper Ready
     - Ablations complete, metrics reported
   * - 12
     - Production Ready
     - Deployable with explainability

Gantt Chart
~~~~~~~~~~~

.. mermaid::

   gantt
       title Project Timeline (12 weeks)
       dateFormat  YYYY-MM-DD
       section Phase 1
       Compliance Mode      :p1a, 2026-01-27, 5d
       Deduplication        :p1b, 2026-01-27, 7d
       Boilerplate/OpenSanctions :p1c, 2026-02-03, 5d
       PII Scrubbing        :p1d, 2026-02-10, 5d
       Corpus Ready         :milestone, m1, 2026-02-17, 0d
       section Phase 2
       TAPT Fixes           :p2a, 2026-02-10, 7d
       Streaming            :p2b, 2026-02-17, 5d
       Supervised Heads     :p2c, 2026-02-17, 10d
       Judge Ready          :milestone, m2, 2026-03-03, 0d
       section Phase 3
       Batched Inference    :p3a, 2026-03-03, 7d
       Calibration          :p3b, 2026-03-03, 7d
       Multi-Agent          :p3c, 2026-03-10, 7d
       RL Integration Complete :milestone, m3, 2026-03-17, 0d
       section Phase 4
       Metrics Module       :p4a, 2026-03-17, 7d
       Ablations            :p4b, 2026-03-24, 10d
       Paper Ready          :milestone, m4, 2026-04-07, 0d
       section Phase 5
       Compliance Corpus    :p5a, 2026-04-07, 7d
       Model Serving        :p5b, 2026-04-14, 10d
       Explainability       :p5c, 2026-04-21, 10d
       Production Ready     :milestone, m5, 2026-05-04, 0d

Risk Assessment
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 30 20

   * - Risk
     - Likelihood
     - Impact
     - Mitigation
     - Owner
   * - Data access blocked
     - Medium
     - High
     - Pre-download all data; use public-only mode
     - Data team
   * - TAPT overfitting
     - High
     - Medium
     - Early stopping; span masking; regularization
     - ML team
   * - Judge too slow for RL
     - Medium
     - High
     - Batching; ONNX export; smaller model
     - ML team
   * - RL training unstable
     - Medium
     - Medium
     - Calibration; reward smoothing; curriculum
     - RL team
   * - Compliance rejection
     - Low
     - High
     - Public-only corpus; PII scrubbing; audit trail
     - Compliance
   * - Paper rejection
     - Medium
     - Medium
     - Strong ablations; novel contributions
     - Research lead

Success Criteria
----------------

Research Success
~~~~~~~~~~~~~~~~

- Publication at top venue (NeurIPS, ICML, KDD, or domain-specific)
- Demonstrate improvement over GNN-only baseline
- Show value of domain-adaptive pretraining via ablations
- Release reproducible codebase and public corpus

Production Success
~~~~~~~~~~~~~~~~~~

- Deployed in at least one pilot bank
- Reduces false positive rate by ≥20% at fixed recall
- Meets compliance requirements (explainability, audit trail)
- Inference latency <100ms per transaction

Quantitative Targets
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Metric
     - Baseline
     - Target
   * - Typology Macro-F1
     - 0.60
     - ≥0.75
   * - Suspiciousness AUC-ROC
     - 0.80
     - ≥0.90
   * - ECE (calibration)
     - 0.15
     - ≤0.05
   * - RL Episode Return
     - --
     - +15% vs GNN-only
   * - FP Rate @ 90% Recall
     - 0.40
     - ≤0.25

Appendix: File Structure
------------------------

.. code-block:: text

   rl_money_laundering/
   ├── src/rl_money_laundering/
   │   ├── dataprep/                    # Phase 1
   │   │   ├── __init__.py
   │   │   ├── base.py                  # Base classes, TextSample
   │   │   ├── fincen_loader.py         # FinCEN Files
   │   │   ├── fincen_advisory_loader.py # NEW: Public advisories
   │   │   ├── sec_fraud_loader.py
   │   │   ├── fatf_loader.py
   │   │   ├── opensanctions_loader.py
   │   │   ├── gdelt_loader.py
   │   │   ├── venmo_loader.py
   │   │   ├── enforcement_loader.py    # NEW: DOJ/SEC actions
   │   │   ├── dedup.py                 # NEW: Deduplication
   │   │   ├── cleaning.py              # NEW: Boilerplate
   │   │   ├── pii_scrubber.py          # NEW: PII removal
   │   │   ├── corpus_builder.py
   │   │   └── pipeline.py
   │   │
   │   ├── pretraining/                 # Phase 2
   │   │   ├── __init__.py
   │   │   ├── config.py
   │   │   ├── collators.py
   │   │   ├── trainer.py
   │   │   ├── scripts.py
   │   │   ├── pretokenize.py           # NEW: Pre-tokenization
   │   │   ├── supervised_heads.py      # NEW: Judge heads
   │   │   ├── judge_finetuning.py      # NEW: Head training
   │   │   └── HARDWARE_REQUIREMENTS.md
   │   │
   │   ├── multiagent/                  # Phase 3
   │   │   ├── __init__.py
   │   │   ├── judge_model.py           # Batched inference
   │   │   ├── calibration.py           # NEW: Calibration
   │   │   ├── multi_agent_env.py
   │   │   ├── reward_computation.py    # Smoothing
   │   │   └── integration/
   │   │
   │   ├── evaluation/                  # Phase 4
   │   │   ├── __init__.py
   │   │   ├── metrics.py               # NEW: Core metrics
   │   │   ├── calibration_metrics.py   # NEW: ECE, Brier
   │   │   ├── rl_metrics.py            # NEW: RL-specific
   │   │   ├── robustness.py            # NEW: Perturbations
   │   │   └── ablations.py             # NEW: Ablation runner
   │   │
   │   └── serving/                     # Phase 5
   │       ├── __init__.py
   │       ├── api.py                   # NEW: REST API
   │       ├── explainability.py        # NEW: Explanations
   │       └── audit.py                 # NEW: Logging
   │
   ├── docs/
   │   ├── roadmap.rst                  # This document
   │   └── ...
   │
   ├── scripts/
   │   ├── run_ablations.py             # NEW
   │   ├── build_corpus.py
   │   └── train_judge.py
   │
   └── tests/
       ├── test_dataprep/
       ├── test_pretraining/
       ├── test_multiagent/
       └── test_evaluation/
