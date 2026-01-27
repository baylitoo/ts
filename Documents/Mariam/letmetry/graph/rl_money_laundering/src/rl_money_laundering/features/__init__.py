"""
Feature extraction modules for AML detection.

Implements the three feature categories from AMLNet Algorithm 2:
- Amount features
- Temporal features (velocity, periodicity, deviations)
- Network features (centrality, clustering)

And provides dataset-specific extractors:
- NodeFeatureExtractor: For AMLNet dataset
- EllipticNodeFeatureExtractor: For Elliptic dataset
"""

from .base import BaseNodeFeatureExtractor
from .extractors import NodeFeatureExtractor
from .elliptic_extractor import EllipticNodeFeatureExtractor
from .temporal import TemporalFeatureExtractor
from .network import NetworkFeatureExtractor

__all__ = [
    "BaseNodeFeatureExtractor",
    "NodeFeatureExtractor",
    "EllipticNodeFeatureExtractor",
    "TemporalFeatureExtractor",
    "NetworkFeatureExtractor",
]
