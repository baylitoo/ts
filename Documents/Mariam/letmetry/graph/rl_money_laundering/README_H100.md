# H100 GPU Training - Quick Start

## 🚀 Quick Start (3 Simple Steps)

### Step 1: Verify H100 Setup
```bash
python scripts/check_h100_setup.py
```
This will verify your GPU is detected and estimate performance.

### Step 2: Run Benchmark Test (5 minutes)
```bash
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --episodes 100 \
  --output_dir outputs/h100_benchmark
```

### Step 3: Full Training (~12-15 hours)
```bash
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json
```

## 📊 What's Different from CPU Training?

| Aspect | CPU (Previous) | H100 (New) | Speedup |
|--------|---------------|------------|---------|
| **Training Speed** | 1.1 it/s | 25-50 it/s | **25-50x** |
| **Batch Size** | 32 | 256 | 8x larger |
| **Model Size** | 256/256/128 | 512/512/256/128 | 2x capacity |
| **Episodes** | 2,400 | 10,000 | 4x more training |
| **Dataset** | 150k rows | Full dataset | Complete data |
| **Buffer Size** | 50k | 200k | 4x experience |
| **Total Time** | 36 min (2.4k episodes) | ~12 hours (10k episodes) | ~20x faster for 4x work |

## 📁 Files Created

- **`configs/h100_scaled_training.json`** - H100-optimized configuration
- **`H100_SCALING_GUIDE.md`** - Comprehensive guide and troubleshooting
- **`scripts/check_h100_setup.py`** - GPU verification and benchmarking
- **`scripts/compare_experiments.py`** - Compare CPU vs GPU results

## 🎯 Expected Results

Based on the scaled configuration, you should see:

**Training Metrics:**
- **Speed**: 25-50x faster than CPU
- **GPU Utilization**: 80-95%
- **Memory Usage**: ~10-15 GB / 80 GB

**Model Performance (after 10,000 episodes):**
- **Precision**: 80-95% (improved from 100% but with better recall)
- **Recall**: 60-80% (improved from 33%)
- **F1 Score**: 70-85% (improved from 50%)
- **Average Reward**: +2.0 to +4.0 (improved from +1.00)

## 🔧 Common Commands

### Monitor GPU During Training
```bash
# Simple monitoring
watch -n 1 nvidia-smi

# Detailed monitoring
pip install nvitop
nvitop
```

### Adjust Batch Size (if OOM)
```bash
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --batch-size 128
```

### Compare Results with CPU Baseline
```bash
python scripts/compare_experiments.py \
  outputs/cpu_tests/09_AGGRESSIVE_CVAR \
  outputs/h100_scaled_training
```

### Resume from Checkpoint
```bash
# Modify config to load from checkpoint
# Then run training again
python scripts/train_agent.py --config_file configs/h100_scaled_training.json
```

## 💡 Key Optimizations Enabled

1. ✅ **8x Larger Batch Size** (32 → 256) - Better GPU utilization
2. ✅ **2x Larger Model** (256 → 512 hidden dims) - More capacity
3. ✅ **128 Quantiles** (vs 64) - Finer risk distribution for QR-DQN
4. ✅ **Full Dataset** - No artificial limits
5. ✅ **4x Larger Replay Buffer** - More diverse experience
6. ✅ **Deeper GNN** (2 → 3 layers) - Better graph representation
7. ✅ **Larger Embeddings** (64 → 128) - Richer node features

## 🐛 Troubleshooting

### CUDA Out of Memory
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
# Increase batch size
python scripts/train_agent.py \
  --config_file configs/h100_scaled_training.json \
  --batch-size 512
```

### CUDA Not Available
```bash
# Check installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 📈 Monitoring Training

During training, you'll see output like:
```
Training:  42%|████████████████▌                      | 4200/10000 [2:30<3:28, 27.82it/s]

Ep 4200 | Stage 8/13 | Reward: +2.34 | ε: 0.25 | IntW: 0.70 | Found%: 2.1 | Flagged%: 38.2
```

Key metrics to watch:
- **it/s**: Should be 25-50 (vs 1.1 on CPU)
- **Reward**: Should increase over time
- **Found%**: Fraud detection rate (target: 2-5%)
- **Flagged%**: Flag usage rate

## 🎓 Next Steps After Training

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

3. **Visualize Best Episodes**
   ```bash
   python scripts/visualize_episodes.py \
     outputs/h100_scaled_training/checkpoints/final_model.pt
   ```

4. **Hyperparameter Tuning** - Try different:
   - Risk measures: `cvar_95`, `cvar_99`
   - Model architectures: More layers, wider networks
   - Learning rates: 1e-4, 5e-4, 1e-3

## 💰 Cost Estimates

If using cloud H100:

| Provider | Cost/hr | 10k Episodes (~12 hrs) | 20k Episodes (~24 hrs) |
|----------|---------|----------------------|----------------------|
| Lambda Labs | $2.00 | ~$24 | ~$48 |
| RunPod | $2.79 | ~$33 | ~$67 |
| AWS p5 | $12.25 | ~$147 | ~$294 |

**Recommendation**: Lambda Labs or RunPod for cost-effective training.

## 📚 Documentation

- Full guide: [H100_SCALING_GUIDE.md](H100_SCALING_GUIDE.md)
- Original training: `outputs/cpu_tests/09_AGGRESSIVE_CVAR/`
- H100 config: [configs/h100_scaled_training.json](configs/h100_scaled_training.json)

## 🤝 Support

For issues or questions:
1. Check [H100_SCALING_GUIDE.md](H100_SCALING_GUIDE.md) troubleshooting section
2. Run `python scripts/check_h100_setup.py` to verify setup
3. Review logs in `outputs/h100_scaled_training/`

---

**Happy Training! 🚀**
