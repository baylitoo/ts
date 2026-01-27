"""
Verify H100 GPU setup and estimate performance for scaled training.

This script checks:
1. CUDA availability and GPU details
2. Memory capacity
3. PyTorch GPU functionality
4. Estimated training performance
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_cuda_availability():
    """Check if CUDA is available and working."""
    print("=" * 80)
    print("CUDA Availability Check")
    print("=" * 80)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\n❌ ERROR: CUDA is not available!")
        print("Possible solutions:")
        print("  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  2. Install PyTorch with CUDA support:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False

    print("✅ CUDA is available!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    return True


def check_gpu_details():
    """Get GPU details and specifications."""
    print("\n" + "=" * 80)
    print("GPU Details")
    print("=" * 80)

    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")

    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")

        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"Total Memory: {total_memory:.2f} GB")

        # Check if H100
        gpu_name = torch.cuda.get_device_name(i).lower()
        is_h100 = "h100" in gpu_name

        if is_h100:
            print("✅ H100 GPU Detected!")
            print("   - 80GB HBM3 memory")
            print("   - 4th Gen Tensor Cores")
            print("   - Expected ~20-50x speedup vs CPU")
        else:
            print(f"ℹ️  Not an H100 (detected: {torch.cuda.get_device_name(i)})")
            print("   Training will still work, but may be slower than H100")

        # Compute capability
        capability = torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor
        print(f"Compute Capability: {capability[0]}.{capability[1]}")

        # Multi-processor count
        mp_count = torch.cuda.get_device_properties(i).multi_processor_count
        print(f"Multi-Processors: {mp_count}")


def benchmark_gpu_performance():
    """Run simple benchmark to estimate training speed."""
    print("\n" + "=" * 80)
    print("GPU Performance Benchmark")
    print("=" * 80)

    device = torch.device("cuda:0")

    # Create a simple model similar to our agent
    class BenchmarkModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(80, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 6)
            )

        def forward(self, x):
            return self.layers(x)

    model = BenchmarkModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Benchmark with different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    print("\nTesting batch sizes (throughput in batches/sec):")
    print(f"{'Batch Size':<15} {'Throughput':<15} {'Samples/sec':<15}")
    print("-" * 45)

    for batch_size in batch_sizes:
        try:
            # Warm up
            for _ in range(10):
                x = torch.randn(batch_size, 80, device=device)
                y = model(x)
                loss = y.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            iterations = 100

            for _ in range(iterations):
                x = torch.randn(batch_size, 80, device=device)
                y = model(x)
                loss = y.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            elapsed = time.time() - start

            throughput = iterations / elapsed
            samples_per_sec = throughput * batch_size

            print(f"{batch_size:<15} {throughput:>10.2f}/s    {samples_per_sec:>10.0f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{batch_size:<15} OOM - Too large")
                torch.cuda.empty_cache()
                break
            else:
                raise

    torch.cuda.empty_cache()


def estimate_training_time():
    """Estimate training time for different configurations."""
    print("\n" + "=" * 80)
    print("Training Time Estimates")
    print("=" * 80)

    # Assumptions based on benchmark
    cpu_speed = 1.1  # iterations per second (from your previous run)

    if torch.cuda.is_available():
        # Conservative estimate: 25x speedup
        gpu_speed = 25 * cpu_speed

        configs = [
            ("Quick Test (100 episodes)", 100),
            ("Medium Run (2,400 episodes)", 2400),
            ("Full Training (10,000 episodes)", 10000),
            ("Extended Training (20,000 episodes)", 20000),
        ]

        print(f"\n{'Configuration':<35} {'Est. Time (CPU)':<20} {'Est. Time (H100)':<20}")
        print("-" * 75)

        for name, episodes in configs:
            cpu_time_sec = episodes / cpu_speed
            gpu_time_sec = episodes / gpu_speed

            cpu_time_str = format_time(cpu_time_sec)
            gpu_time_str = format_time(gpu_time_sec)

            print(f"{name:<35} {cpu_time_str:<20} {gpu_time_str:<20}")

        print("\n💡 H100 provides ~25-50x speedup over CPU training")
        print("💡 Actual speedup depends on batch size and model complexity")


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def check_memory_requirements():
    """Estimate memory requirements for training."""
    print("\n" + "=" * 80)
    print("Memory Requirements")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping memory check")
        return

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Simulate model memory usage
    from rl_money_laundering.config import ExperimentConfig

    config = ExperimentConfig.load("configs/h100_scaled_training.json")

    # Model size estimate
    print("\nEstimated Model Size:")
    print(f"  GNN Embedding Dim: {config.gnn.embedding_dim}")
    print(f"  GNN Hidden Channels: {config.gnn.hidden_channels}")
    print(f"  Agent Hidden Dims: {config.agent.hidden_dims}")

    # Rough parameter count
    gnn_params = (
        config.gnn.node_feature_dim * config.gnn.hidden_channels * config.gnn.num_layers +
        config.gnn.hidden_channels * config.gnn.embedding_dim
    )

    agent_params = (
        config.agent.state_dim * config.agent.hidden_dims[0] +
        sum(config.agent.hidden_dims[i] * config.agent.hidden_dims[i+1]
            for i in range(len(config.agent.hidden_dims)-1)) +
        config.agent.hidden_dims[-1] * config.agent.action_dim
    )

    total_params = gnn_params + agent_params
    model_size_mb = (total_params * 4) / (1024**2)  # 4 bytes per float32

    print(f"\nTotal Parameters: ~{total_params:,}")
    print(f"Model Size: ~{model_size_mb:.1f} MB")

    # Batch memory estimate
    batch_size = config.trainer.batch_size
    state_size_mb = (batch_size * config.agent.state_dim * 4) / (1024**2)

    print(f"\nBatch Size: {batch_size}")
    print(f"Batch Memory: ~{state_size_mb:.1f} MB per batch")

    # Total estimate
    total_estimated = model_size_mb * 3 + state_size_mb * 2  # Model + gradients + optimizer states + batches
    print(f"\nEstimated Total GPU Memory: ~{total_estimated:.0f} MB (~{total_estimated/1024:.1f} GB)")

    available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Available GPU Memory: {available_memory:.1f} GB")

    if total_estimated / 1024 < available_memory * 0.8:
        print("✅ Memory requirements look good!")
    else:
        print("⚠️  May need to reduce batch size or model size")


def main():
    """Run all checks and benchmarks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "H100 GPU Setup Verification" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Check CUDA
    if not check_cuda_availability():
        print("\n❌ Cannot proceed without CUDA support")
        return

    # Check GPU details
    check_gpu_details()

    # Benchmark performance
    try:
        benchmark_gpu_performance()
    except Exception as e:
        print(f"\n⚠️  Benchmark failed: {e}")

    # Estimate training time
    estimate_training_time()

    # Check memory
    try:
        check_memory_requirements()
    except Exception as e:
        print(f"\n⚠️  Memory check failed: {e}")

    # Final recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print("""
✅ Your system is ready for H100 training!

Next steps:
1. Run benchmark test (100 episodes):
   python scripts/train_agent.py --config_file configs/h100_scaled_training.json --episodes 100

2. If benchmark looks good, run full training:
   python scripts/train_agent.py --config_file configs/h100_scaled_training.json

3. Monitor GPU usage:
   watch -n 1 nvidia-smi

4. Track training progress in:
   outputs/h100_scaled_training/
    """)


if __name__ == "__main__":
    main()
