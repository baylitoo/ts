# Pretraining Plan Review

## Scope
This review focuses on the pretraining roadmap for data preparation (compliance, deduplication,
cleaning, PII handling, metadata) and the training pipeline (TAPT/DAPT, streaming, overfitting
controls, supervised judge heads).

## Data Preparation (Dataprep, Cleaning, Compliance)

### Strengths
- **Compliance-first design**: Explicit compliance modes with strict/public-only options provide a
  governance-safe pathway to disable sensitive sources, alongside an audit trail of enabled data
  sources. This is crucial for handling leaked SARs and similar sources.
- **Deduplication plan**: Combining exact hash dedup with MinHash near-duplicate detection and
  document length filtering directly addresses the heavy duplication in news sources.
- **Source-specific cleaning**: Boilerplate stripping for SEC EDGAR and news sources addresses the
  highest-noise inputs, and makes the corpus more semantically aligned with AML narratives.
- **PII scrubbing**: Early redaction strategy reduces compliance exposure while preserving entity
  type signals that can still benefit model training.
- **Metadata schema**: A structured schema with provenance/processing fields supports auditability
  and reproducibility for pretraining datasets.

### Gaps & Risks
- **Validation/QA checks**: The roadmap does not specify concrete post-cleaning validation gates
  (e.g., dedup recall estimates, PII redaction accuracy, or per-source token contribution audits).
- **Language filtering**: There is no explicit language detection/normalization plan (e.g., if
  GDELT is multilingual, we need filtering or multilingual model alignment).
- **License enforcement**: Although provenance is planned, there is no explicit enforcement step
  or automated check to stop data with restricted licenses from entering strict/public-only modes.
- **Bias control**: OpenSanctions capping is good, but overall class/source imbalance controls
  should also include token-level caps per source and monitoring of domain skew.

### Recommendations
1. **Add dataset QA checkpoints**
   - PII redaction precision/recall sampling.
   - Per-source token and document counts post-dedup/cleaning.
   - Dedup quality check: estimate near-duplicate false negatives.
2. **Enforce license compliance**
   - Build an allowlist/denylist for licenses and fail the pipeline in strict mode if any
     restricted license is detected.
3. **Language and content filtering**
   - Add language detection (fastText or CLD3) with configurable thresholds.
   - Add basic profanity or non-AML content filters if GDELT or web sources dominate.
4. **PII and redaction strategy**
   - Ensure masking tokens align with the model’s tokenizer to avoid tokenization artifacts.
   - Evaluate a “partial masking” policy for legal names if downstream tasks rely on
     entity linking.

## Training (DAPT/TAPT, Streaming, Supervised Judge Heads)

### Strengths
- **Overfitting mitigation**: Lower LR, higher weight decay, early stopping, and span masking
  are appropriate for a small TAPT corpus and align with best practice for low-token regimes.
- **IO optimizations**: Pre-tokenization and packed streaming reduce padding waste and improve
  training throughput, which is crucial for long corpora.
- **Supervised judge heads**: The explicit supervised tasks and ranking loss target practical
  decision behavior rather than only MLM embeddings.
- **Structured outputs**: The multi-task design (classification, ranking, evidence extraction)
  supports explainability requirements for AML.

### Gaps & Risks
- **Evaluation plan**: No mention of a held-out evaluation set for DAPT/TAPT beyond validation
  perplexity; need downstream task evaluation or proxy metrics.
- **Cross-domain drift**: Pretraining on finance/legal sources may diverge from task text
  distributions; no plan for domain mixing or weighting during training.
- **Calibration strategy**: Temperature scaling is mentioned for suspiciousness scoring, but
  calibration metrics and reference splits are not defined.
- **Compute budget alignment**: The roadmap lacks an explicit mapping of expected GPU hours or
  per-stage token counts to confirm feasibility vs. resources.

### Recommendations
1. **Add a formal evaluation battery**
   - Track MLM metrics plus downstream proxy tasks on a fixed dev set.
   - Include calibration curves (ECE/Brier) for suspiciousness scoring.
2. **Mixing strategy across DAPT/TAPT**
   - Consider interleaving domain and task data with adjustable mixing ratios.
   - Add a small slice of general-domain data to reduce catastrophic specialization.
3. **Explicit compute and scaling plan**
   - Token count budgets for DAPT and TAPT.
   - Expected wall-clock time per stage on target hardware.
4. **Ablation tracking**
   - Compare baseline vs. span masking vs. early stopping.
   - Compare supervised heads with and without LoRA fine-tuning.

## Summary
The roadmap is strong on core compliance, deduplication, and overfitting mitigation. The largest
missing pieces are explicit QA checkpoints for the data pipeline and an evaluation/calibration
plan for the training side. Addressing these will make the pretraining pipeline more reliable and
production-ready.

## Command Log
- `rg -n "pretraining|pre-train|pretrain" /workspace/ts/Documents/Mariam/letmetry/graph/rl_money_laundering`
- `sed -n '120,220p' /workspace/ts/Documents/Mariam/letmetry/graph/rl_money_laundering/docs/roadmap.rst`
- `sed -n '220,340p' /workspace/ts/Documents/Mariam/letmetry/graph/rl_money_laundering/docs/roadmap.rst`
- `sed -n '340,460p' /workspace/ts/Documents/Mariam/letmetry/graph/rl_money_laundering/docs/roadmap.rst`
- `sed -n '460,620p' /workspace/ts/Documents/Mariam/letmetry/graph/rl_money_laundering/docs/roadmap.rst`
