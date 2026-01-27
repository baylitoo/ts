# Datasets

## Overview
Dataset loaders in this folder standardize how transaction graphs are ingested,
engineered, and split for training and evaluation. The core abstractions produce
NetworkX graphs with AML-focused node/edge attributes and provide consistent
train/validation/test splits.

## Key modules
- `base.py`: `BaseGraphDataset` and `DatasetSplits` base classes for dataset
  loaders, plus file existence checks and target summaries.
- `amlnet.py`: AMLNet synthetic transaction dataset loader with extensive feature
  engineering, split creation, and graph construction utilities.
- `elliptic.py`: Elliptic Bitcoin dataset loader with temporal splitting,
  engineered features, and graph construction that attaches node attributes.
- `features.py`: Shared feature helpers (categorical encoding, scaling,
  cyclical time features, log transforms).

## Usage notes
- Load raw data with `load_raw()`, then call `feature_pipeline()` to create
  model-ready features and targets.
- Graph construction methods (`build_graph`) create directed graphs with
  transaction metadata and fraud labels under consistent attribute names.
