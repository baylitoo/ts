# Deprecated Legacy Scripts

**These scripts are deprecated and kept only for reference.**

---

## ⚠️ DO NOT USE THESE FILES

The following scripts use deprecated data loaders and old patterns:

- `train_agent_legacy.py` - Old training script (use `train_agent_v2.py` instead)
- `train_baselines.py` - Old baseline training (needs updating)

---

## ✅ Use These Instead

### For RL Agent Training
**Use:** `train_agent_v2.py`

```bash
# Quick test
python scripts/train_agent_v2.py --config quick_test

# Full training
python scripts/train_agent_v2.py --config amlnet_full
```

**Benefits:**
- Config-based experiments
- Automatic checkpoint management
- Metrics tracking
- Reproducibility

**Documentation:** [QUICK_START_REFACTORED.md](../QUICK_START_REFACTORED.md)

---

## Migration Path

If you have existing code using the old loaders:

### Before (Deprecated)
```python
from rl_money_laundering.data_loader import AMLNetDataLoader

loader = AMLNetDataLoader("data/amlnet/AMLNet_August_2025.csv")
df = loader.load_data(nrows=10000)
df_features = loader.extract_features()
graph = loader.build_transaction_graph()
```

### After (Current)
```python
from rl_money_laundering.datasets.amlnet import AMLNetLoader

loader = AMLNetLoader("data/amlnet/AMLNet_August_2025.csv")
splits = loader.load_and_split(nrows=10000, test_size=0.2, val_size=0.1)

# Access components
graph = splits.graph
train_nodes = splits.train_nodes
val_nodes = splits.val_nodes
test_nodes = splits.test_nodes
```

---

## Files Removed

The following deprecated files have been **permanently deleted**:

- ✅ `src/rl_money_laundering/data_loader.py` (replaced by `datasets/amlnet.py`)
- ✅ `src/rl_money_laundering/elliptic_loader.py` (replaced by `datasets/elliptic.py`)
- ✅ `src/rl_money_laundering/hello.py` (boilerplate, not needed)

---

## Why Were They Deprecated?

1. **Code Duplication**: Old loaders duplicated functionality in `datasets/` module
2. **Inconsistent Interface**: Old loaders didn't follow the `BaseGraphDataset` pattern
3. **No Train/Val/Test Splits**: Old loaders didn't provide proper data splitting
4. **Missing Features**: Old loaders lacked feature scaling, encoding, and proper graph construction

---

## Need Help?

- **Quick Start:** [QUICK_START_REFACTORED.md](../QUICK_START_REFACTORED.md)
- **Full Documentation:** [REFACTORING_COMPLETE.md](../REFACTORING_COMPLETE.md)
- **Config System:** [config.py](../src/rl_money_laundering/config.py)

---

**Status:** These files are kept for reference only. Do not use in production.
