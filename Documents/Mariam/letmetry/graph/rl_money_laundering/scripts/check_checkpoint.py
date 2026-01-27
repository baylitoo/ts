"""Quick script to inspect checkpoint contents."""
import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/cpu_tests/05_CVAR_RISK_AVERSE/checkpoints/checkpoint_ep2300.pt"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, dict):
        print(f"  {key}: dict with {len(value)} items")
    elif hasattr(value, 'shape'):
        print(f"  {key}: tensor {value.shape}")
    else:
        print(f"  {key}: {type(value).__name__} = {value}")
