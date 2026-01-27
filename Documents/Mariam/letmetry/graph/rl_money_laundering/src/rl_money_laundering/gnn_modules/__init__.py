"""
RMGANets Graph Neural Network Modules

Implements the 3-module architecture from RMGANets paper:
1. Att-GCM: Attention-Relation Graph Convolution
2. To-GCM: Adaptive Topology Graph Convolution
3. HyGCM: Hybrid Enhanced Graph Convolution
"""

from .att_gcm import AttentionGCM
from .to_gcm import AdaptiveTopoGCM
from .hy_gcm import HybridEnhancedGCM
from .fusion import FeatureFusionModule
from .rmganets_encoder import RMGANetsEncoder

__all__ = [
    'AttentionGCM',
    'AdaptiveTopoGCM',
    'HybridEnhancedGCM',
    'FeatureFusionModule',
    'RMGANetsEncoder'
]
