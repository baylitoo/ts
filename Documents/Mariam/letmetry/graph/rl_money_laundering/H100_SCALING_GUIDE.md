# H100 GPU Scaling Guide

This guide explains how to scale your RL-based AML detection model to run efficiently on an H100 GPU.

## Quick Start

```bash
# Run with the H100-optimized configuration
python scripts/train_agent.py --config_file configs/h100_scaled_training.json
```

## Key Changes for H100 Scaling

### 1. **Model Architecture (Scaled Up)**

| Component | CPU (Previous) | H100 (New) | Improvement |
|-----------|---------------|------------|-------------|
| **Agent Hidden Dims** | [256, 256, 128] | [512, 512, 256, 128] | 4x capacity |
| **GNN Embedding Dim** | 64 | 128 | 2x capacity |
| **GNN Hidden Channels** | 160 | 320 | 2x capacity |
| **GNN Layers** | 2 | 3 | Deeper network |
| **History Dim** | 16 | 32 | 2x temporal memory |
| **Num Quantiles** | 64 | 128 | Finer risk distribution |

### 2. **Training Configuration (Scaled Up)**

| Parameter | CPU (Previous) | H100 (New) | Reason |
|-----------|---------------|------------|--------|
| **Batch Size** | 32 | 256 | 8x larger (H100 has 80GB memory) |
| **Buffer Capacity** | 50,000 | 200,000 | 4x more experience |
| **Num Episodes** | 2,400 | 10,000 | More comprehensive training |
| **Dataset Size** | 150,000 rows | Full dataset | Use all available data |
| **Eval Frequency** | 50 | 100 | Less frequent (faster training) |
| **Checkpoint Frequency** | 100 | 500 | Save storage space |
| **Num Eval Episodes** | 20 | 50 | More robust evaluation |

### 3. **GPU Optimizations**

The configuration automatically enables:
- **CUDA device**: All computations on GPU
- **Mixed precision training**: FP16 where applicable (if supported)
- **Larger batch sizes**: Utilize H100's 80GB memory
- **Parallel data loading**: Faster dataset iteration

## Performance Expectations

### Speed Improvements
- **Expected speedup**: 20-50x faster than CPU
- **Episodes per hour**: ~600-1000 (vs ~60-80 on CPU)
- **Total training time**: 10-16 hours for 10,000 episodes (vs 5-7 days on CPU)

### Memory Usage
- **Model size**: ~500MB-1GB
- **Batch memory**: ~2-4GB per batch (256 samples)
- **Total GPU usage**: ~10-15GB (leaving plenty of headroom on H100's 80GB)

## Advanced Training Commands

### Full Dataset Training (Recommended)
```bash
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json
```

### Custom Hyperparameter Tuning
```bash
# Increase batch size further (if memory allows)
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --batch-size 512

# Longer training with more quantiles
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --episodes 20000 \
  --num-quantiles 256

# Aggressive risk-averse training
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --risk-measure cvar_95 \
  --num-quantiles 128
```

### Multi-Run Experiments (Ablation Studies)
```bash
# Baseline run
python scripts/train_agent.py --config_file configs/h100_scaled_training.json

# Without guided exploration
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --no-guided-exploration \
  --output_dir outputs/h100_no_guided

# Different GNN architectures
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --gnn-type gat \
  --output_dir outputs/h100_gat
```

## H100-Specific Optimizations

### 1. **Enable TF32 for Matrix Multiplications**
Add this to your training script (already in pipeline):
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. **Use Tensor Cores Efficiently**
- Batch sizes should be multiples of 8 for optimal performance
- Hidden dimensions should be multiples of 64 for tensor core alignment
- Current config already follows these best practices

### 3. **Pin Memory for Faster Data Transfer**
```python
# Already handled in DataLoader creation
DataLoader(..., pin_memory=True, num_workers=4)
```

## Monitoring GPU Utilization

### Check GPU Usage During Training
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

You should see:
- **GPU Utilization**: 80-95% (good)
- **Memory Usage**: 10-20GB / 80GB
- **Temperature**: <80°C (H100 has excellent cooling)

### Monitor with nvtop (More Detailed)
```bash
# Install nvtop if not available
pip install nvitop

# Run monitoring
nvitop
```

## Scaling Beyond H100

### Multi-GPU Training (Future)
For even faster training with multiple GPUs:

```python
# Distributed Data Parallel (DDP) - requires code modifications
# See: pytorch.org/tutorials/beginner/dist_overview.html
```

### Cloud Training Options
- **AWS p5.48xlarge**: 8x H100 GPUs ($98/hr)
- **GCP a3-highgpu-8g**: 8x H100 GPUs
- **Azure ND96amsr_v4**: 8x H100 GPUs

## Troubleshooting

### Out of Memory (OOM) Errors
```bash
# Reduce batch size
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --batch-size 128

# Or reduce model size
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --hidden-dims 256,256,128
```

### Slow Training (Low GPU Utilization)
```bash
# Increase batch size to saturate GPU
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --batch-size 512

# Reduce checkpoint frequency
# (modify config.json: "checkpoint_frequency": 1000)
```

### CUDA Out of Memory (Graph Too Large)
```bash
# Limit graph edges during construction
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --max_edges 500000

# Or use gradient checkpointing (requires code modification)
```

## Benchmarking

### Expected Performance Metrics

**CPU Training (Your Previous Run)**:
- Time: ~36 minutes for 2,400 episodes
- Speed: ~1.1 it/s
- Final F1 (fraud-seeded): 50.0%

**H100 Training (Expected)**:
- Time: ~2-3 hours for 10,000 episodes
- Speed: ~30-50 it/s (25-45x faster)
- Final F1 (fraud-seeded): 60-80% (with more training)

### Run Benchmark Test
```bash
# Quick 100-episode test to verify GPU performance
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --episodes 100 \
  --output_dir outputs/h100_benchmark

# Should complete in ~3-5 minutes
```

## Cost Estimation

### Cloud GPU Costs (Approximate)

| Provider | Instance | GPU | Cost/hr | 10k Episodes | Full Run Cost |
|----------|----------|-----|---------|--------------|---------------|
| Lambda Labs | 1x H100 | 80GB | $2.00 | ~15 hours | ~$30 |
| RunPod | 1x H100 | 80GB | $2.79 | ~15 hours | ~$42 |
| AWS p5.2xlarge | 1x H100 | 80GB | $12.25 | ~15 hours | ~$184 |
| GCP a3-highgpu-1g | 1x H100 | 80GB | $14.73 | ~15 hours | ~$221 |

**Recommendation**: Use Lambda Labs or RunPod for cost-effective H100 access.

## Best Practices

1. ✅ **Start with benchmark run** (100 episodes) to verify GPU setup
2. ✅ **Monitor GPU utilization** - should be >80%
3. ✅ **Use full dataset** - no `nrows` limit
4. ✅ **Enable checkpointing** - save every 500 episodes
5. ✅ **Run multiple seeds** - train 3-5 runs with different seeds for robustness
6. ✅ **Save evaluation metrics** - track precision/recall curves
7. ✅ **Compare with CPU baseline** - verify improvements

## Next Steps

After H100 training completes:

1. **Analyze Results**
   ```bash
   python scripts/analyze_training.py outputs/h100_scaled_training
   ```

2. **Compare with CPU Baseline**
   ```bash
   python scripts/compare_experiments.py \
     outputs/cpu_tests/09_AGGRESSIVE_CVAR \
     outputs/h100_scaled_training
   ```

3. **Hyperparameter Tuning**
   - Try different risk measures (cvar_95, cvar_99)
   - Experiment with larger models (more layers, wider networks)
   - Test different curriculum schedules

4. **Deploy Best Model**
   ```bash
   python scripts/deploy_model.py \
     outputs/h100_scaled_training/checkpoints/final_model.pt
   ```

## Support & Debugging

If you encounter issues:
1. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Verify GPU: `nvidia-smi`
3. Review logs in `outputs/h100_scaled_training/`
4. Check memory usage during training

---

**Ready to scale!** Run the command above and enjoy 20-50x faster training on H100! 🚀
