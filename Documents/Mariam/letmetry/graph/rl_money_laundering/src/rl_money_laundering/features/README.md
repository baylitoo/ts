# Feature Extraction

## Overview
Feature extractors for AML detection that combine AMLNet Algorithm-2 heuristics
(amount, temporal, and network signals) with dataset-specific extractors.

## Key modules
- `base.py`: Abstract `BaseNodeFeatureExtractor` interface for node feature
  extractors.
- `extractors.py`: `NodeFeatureExtractor` that composes account type, amount,
  temporal, and network features into a fixed-size vector for GNNs.
- `temporal.py`: Temporal feature extraction (velocity, periodicity, business
  hour signals) and graph enrichment helpers.
- `network.py`: Network/centrality feature extraction with caching and
  graph enrichment helpers.
- `network_pyg.py`: PyTorch Geometric implementations for degree-based
  features on large graphs.
- `elliptic_extractor.py`: Elliptic dataset node feature extractor (166 raw +
  engineered temporal/statistical features).

## Usage notes
- The extractors are designed to work directly with NetworkX node attribute
  dictionaries and can be injected into RL environments or data pipelines.
- For large graphs, `network_pyg.py` can compute degree-based features without
  the full NetworkX centrality cost.
