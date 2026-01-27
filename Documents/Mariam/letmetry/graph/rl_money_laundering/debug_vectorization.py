"""
Debug script to test vectorization issue with GraphSpace.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from gymnasium.vector import SyncVectorEnv
from rl_money_laundering.rllib_integration.graph_space import GraphSpace

def make_env():
    """Create a dummy environment for testing."""
    from gymnasium import Env

    class DummyGraphEnv(Env):
        def __init__(self):
            self.observation_space = GraphSpace(
                max_nodes=10,
                max_edges=20,
                node_feature_dim=5,
                context_feature_dim=3
            )
            self.action_space = self.observation_space  # Dummy

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            # Generate valid observation with edges pointing to valid nodes
            num_nodes = 7
            num_edges = 15

            # Return observation matching our GraphSpace format
            obs = {
                "node_features": np.random.randn(10, 5).astype(np.float32),
                # Edge indices must be < num_nodes (7), not max_nodes (10)
                "edge_index": np.random.randint(0, num_nodes, size=(2, 20)).astype(np.int64),
                "num_nodes": np.array([num_nodes], dtype=np.int64),  # shape (1,)
                "num_edges": np.array([num_edges], dtype=np.int64),  # shape (1,)
                "context_features": np.random.randn(3).astype(np.float32),
            }

            print(f"Single env observation shapes:")
            for key, val in obs.items():
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

            # Check if observation is valid
            is_valid = self.observation_space.contains(obs)
            print(f"Observation valid: {is_valid}")

            if not is_valid:
                # Debug each field individually
                print(f"  DEBUG: Checking each field:")
                for key in ['node_features', 'edge_index', 'num_nodes', 'num_edges', 'context_features']:
                    field_valid = self.observation_space.spaces[key].contains(obs[key])
                    print(f"    {key}: {field_valid}")
                    if not field_valid:
                        print(f"      Expected: {self.observation_space.spaces[key]}")
                        print(f"      Got: shape={obs[key].shape}, dtype={obs[key].dtype}")

            return obs, {}

        def step(self, action):
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

    return DummyGraphEnv

print("="*70)
print("Testing Single Environment")
print("="*70)
env = make_env()()
obs, info = env.reset()
print()

print("="*70)
print("Testing Vectorized Environment (2 envs)")
print("="*70)
try:
    vec_env = SyncVectorEnv([make_env() for _ in range(2)])
    print(f"Vector env created successfully")
    print(f"Observation space: {vec_env.observation_space}")
    print()

    obs, info = vec_env.reset()
    print(f"Vector reset successful!")
    print(f"Vectorized observation shapes:")
    for key, val in obs.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
