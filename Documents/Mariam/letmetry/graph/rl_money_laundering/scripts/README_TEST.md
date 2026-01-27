# Quick Setup Test

Run this **before** deploying to GPU to verify all components work correctly.

## Requirements

- AMLNet dataset CSV file downloaded
- CPU machine (no GPU needed for testing)
- Python environment with dependencies installed

## Usage

### Basic test (recommended):

```bash
# With uv (recommended)
uv run python scripts/test_setup.py --data-path /path/to/amlnet.csv

# Or with regular python
python scripts/test_setup.py --data-path /path/to/amlnet.csv
```

### Custom options:

```bash
uv run python scripts/test_setup.py \
    --data-path /path/to/amlnet.csv \
    --nrows 2000 \              # Load 2000 rows instead of 1000
    --max-edges 1000 \          # Build graph with 1000 edges
    --episode-steps 20          # Run 20-step episode
```

## What it tests:

1. ✅ **Imports**: All modules import correctly
2. ✅ **Data Loading**: AMLNet dataset loads from CSV
3. ✅ **Graph Construction**: Transaction graph builds correctly
4. ✅ **Environment**: RL environment initializes and runs
5. ✅ **GNN Encoder**: GraphSAGE encoder creates embeddings
6. ✅ **Agent**: QR-DQN agent initializes (tiny CPU config)
7. ✅ **Episode Rollout**: Agent can interact with environment
8. ✅ **Training Step**: Agent can perform gradient updates
9. ✅ **Checkpoint**: Save/load model checkpoints

## Expected output:

```
============================================================
AML RL AGENT - SETUP TEST (CPU MODE)
============================================================
Data path: /path/to/amlnet.csv
Loading: 1000 rows
Graph edges: 500
PyTorch: 2.x.x
Device: cpu (GPU test will happen after deployment)

============================================================
TEST 1: Importing modules...
============================================================
✅ All imports successful!

============================================================
TEST 2: Loading 1000 rows from AMLNet...
============================================================
✅ Loaded 1000 transactions
   Columns: ['nameOrig', 'nameDest', 'amount', ...]
   Shape: (1000, 15)

... [more tests] ...

============================================================
SUMMARY
============================================================
✅ PASS: imports
✅ PASS: data_loading
✅ PASS: graph_construction
✅ PASS: environment
✅ PASS: gnn_encoder
✅ PASS: agent
✅ PASS: episode_rollout
✅ PASS: training_step
✅ PASS: checkpoint

============================================================
RESULT: 9/9 tests passed
============================================================

🎉 ALL TESTS PASSED!
✅ Ready for GPU deployment on H100!

Next steps:
1. Push code to GitHub
2. Deploy to H100 machine
3. Run: docker-compose build
4. Run: ./docker-run.sh train
```

## Troubleshooting

### "Data file not found"
Make sure you provide the correct path to your AMLNet CSV file.

### "Missing columns"
Verify your AMLNet CSV has these columns:
- `nameOrig`, `nameDest`, `amount`, `step`, `isMoneyLaundering`

### Out of memory on CPU
Reduce the number of rows and edges:
```bash
uv run python scripts/test_setup.py \
    --data-path /path/to/amlnet.csv \
    --nrows 500 \
    --max-edges 250
```

## After testing

Once all tests pass, you're ready to:
1. Commit and push to GitHub
2. Deploy to your H100 machine
3. Run full training with Docker

The test creates a tiny model config for CPU testing. The actual GPU training will use the full QR-DQN configuration.
