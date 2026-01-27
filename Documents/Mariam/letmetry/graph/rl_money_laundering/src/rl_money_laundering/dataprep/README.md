# Data Preparation

## Overview
Text-centric AML data preparation pipeline that aggregates multiple sources,
cleans and deduplicates documents, and exports unified corpora for
pretraining or supervised fine-tuning.

## Key modules
- `base.py`: Core data governance models, compliance modes, dataset registries,
  and shared dataclasses for text samples and provenance tracking.
- `cleaning.py`: Source-aware boilerplate removal (SEC filings, news, generic
  boilerplate) plus normalization helpers.
- `pii_scrubber.py`: Redaction utilities for sensitive fields.
- `dedup.py`: Near-duplicate detection and deduplication helpers.
- `corpus_builder.py`: `UnifiedCorpusBuilder` for weighted mixing and chunking
  across sources.
- `pipeline.py`: `DataPrepPipeline` orchestrator for end-to-end data prep runs.
- Dataset loaders: `fincen_loader.py`, `fatf_loader.py`, `sec_fraud_loader.py`,
  `fraudnlp_loader.py`, `opensanctions_loader.py`, `gdelt_loader.py`,
  `venmo_loader.py`.

## Usage notes
- Use `PipelineConfig` and `DataPrepPipeline` to control compliance mode,
  enabled sources, and corpus export format.
- Cleaning, PII scrubbing, and deduplication are designed to be modular and
  can be toggled per source in custom workflows.
