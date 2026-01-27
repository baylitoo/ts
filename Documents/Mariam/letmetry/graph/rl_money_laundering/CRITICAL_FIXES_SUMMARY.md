# Critical Architectural Fixes - Summary

## Fixed Issues

### 1. GNN Never Trained in Non-RLlib Pipeline (CRITICAL)

**Update**: Added optional graph replay buffer for GNN TD updates when `use_graph_replay=True` in `AMLTrainer`.

**Issue**: The GNN encoder was set to `eval()` mode and never trained unless `multi_branch=True` (which is `False` by default). This made the GNN act as a frozen random feature extractor instead of learning useful representations.

**Root Cause**:
- [trainer.py:117](src/rl_money_laundering/trainer.py#L117): `self.state_encoder.eval()` when multi_branch disabled
- [trainer.py:266-274](src/rl_money_laundering/trainer.py#L266-L274): Forward pass wrapped in `torch.no_grad()` when multi_branch disabled
- Line 276: Output detached and converted to numpy, breaking gradient flow

**Fixes Applied**:
1. Moved `self.state_encoder.train()` outside the multi_branch conditional ([trainer.py:92](src/rl_money_laundering/trainer.py#L92))
2. Created dedicated GNN optimizer ([trainer.py:96-99](src/rl_money_laundering/trainer.py#L96-L99))
3. Removed `torch.no_grad()` wrapper from state encoder forward pass ([trainer.py:265-272](src/rl_money_laundering/trainer.py#L265-L272))

**Remaining Work**:
- TD-style GNN update is now applied per transition (online), but replay-buffer training still requires graph storage
- Current architecture still detaches embeddings before storing in replay buffer
- Full end-to-end training requires storing graph structures in replay buffer (major refactor)

### 2. RLlib Wrapper Incorrectly Parses Base Environment State (CRITICAL)

**Issue**: RLlib wrapper assumed base environment returns `[GNN_embedding, history]` but it actually returns `[raw_node_features (10), history_features (6+)]`.

**Root Cause**:
- [env_wrapper.py:170-180](src/rl_money_laundering/rllib_integration/env_wrapper.py#L170-L180): Incorrect slicing using `embedding_dim` instead of actual node feature count
- Base environment ([environment.py:286-345](src/rl_money_laundering/environment.py#L286-L345)) returns raw features, not GNN-encoded states

**Fix Applied**:
- Updated `_state_to_graph_obs()` to correctly extract history features from position 10 onwards ([env_wrapper.py:180-191](src/rl_money_laundering/rllib_integration/env_wrapper.py#L180-L191))
- Changed from `state[embedding_dim:]` to `state[num_node_features:num_node_features + history_dim]`
- Added proper comment explaining actual state structure

### 3. Random Node Ordering in RLlib Wrapper (CRITICAL)

**Issue**: `node_list = list(subgraph_nodes)` where `subgraph_nodes` is an unordered set. This makes the "current node" embedding random, breaking the assumption in `batch_utils.extract_first_node_embeddings()`.

**Fix Applied**:
- [env_wrapper.py:156](src/rl_money_laundering/rllib_integration/env_wrapper.py#L156): Ensure current_node is ALWAYS first
- Changed from: `node_list = list(subgraph_nodes)`
- To: `node_list = [current_node] + sorted([n for n in subgraph_nodes if n != current_node])`

### 4. Subgraph Size Overflow (CRITICAL)

**Issue**: No capping of k-hop neighborhoods. If subgraph exceeds `max_nodes`, indexing `node_features[idx]` will be out of bounds.

**Fix Applied**:
- [env_wrapper.py:146-150](src/rl_money_laundering/rllib_integration/env_wrapper.py#L146-L150): Cap subgraph size before creating node list
- Keep current_node + closest neighbors up to `max_nodes` limit

### 5. Temporal GNN Edge Times Missing in RLlib (CRITICAL)

**Issue**: RLlib observations did not include edge timestamps, so TGAT/TGN could not receive `edge_time` and would silently run as non-temporal GNNs.

**Fix Applied**:
- Added `edge_time` to GraphSpace definition and sampling (`rllib_integration/graph_space.py`)
- Extracted edge timestamps (`time` / `timestamp` / `step`) in `_state_to_graph_obs` (`rllib_integration/env_wrapper.py`)
- Batched edge_time into PyG `Batch` objects (`rllib_integration/batch_utils.py`)

**Note**: This enables temporal GNNs in RLlib; configs still need to select temporal encoders explicitly.

### 6. Missing Info Keys for RLlib Callbacks (FIXED)
### 7. NodeFeatureExtractor Reuse (FIXED)
- RLlib env wrapper now reuses a shared NodeFeatureExtractor via a module-level cache (configurable via `feature_extractor_cache_key`).

- Added `is_fraud_edge`, `is_fraud_node`, `fraud_pattern_type`, and `timestamp` to `environment.py` info payload.

## Remaining Critical Issues (NOT YET FIXED)

### 8. Wrong Fraud Edge Counting (PARTIALLY FIXED)
- Environment and StateEncoder history now use unified fraud label checks (`is_fraud` / `is_money_laundering` / `isFraud`).
- Other modules may still rely on single-key checks.

### 9. Data Leakage in Dataset Preprocessing
- Dataset split happens after feature engineering
- Features should be computed separately for train/test to avoid leakage

## Testing Status

- **RLlib Tests 1-2**: Should now pass (vectorization fixed, context_features fixed)
- **RLlib Tests 3-4**: Need Ray installed to verify
- **Non-RLlib Training**: TD-style GNN update enabled; replay-buffer GNN updates available when enabled

## Next Steps

1. Validate TD-style GNN update (loss stability, effect on rewards)
2. Test RLlib integration with Ray installed
3. Fix remaining high-priority issues (fraud counting cleanup, data leakage)
