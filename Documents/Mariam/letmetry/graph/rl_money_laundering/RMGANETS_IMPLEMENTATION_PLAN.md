# RMGANets Implementation Plan
## Strategic Analysis & Smart Integration

Based on paper analysis (pages 11-13, ablation studies) and our current architecture comparison.

---

## Executive Summary

**Strategy:** Cherry-pick RMGANets' proven components while keeping our superior RL infrastructure (QR-DQN, entity-disjoint splits, budgeted metrics).

**Key Insight from Ablation Studies (Tables 4-6):**
- **Att-GCM alone**: F1=78.77% (Elliptic) ❌ Poor
- **To-GCM alone**: F1=79.03% ❌ Marginally better
- **HyGCM alone**: F1=78.98% ❌ Poor
- **Att-To-HyGCM (fusion)**: F1=90.43% ✅ **+11.66% improvement!**
- **+ DQN**: F1=92.39% ✅ **+1.96% more**
- **+ Multi-branch loss**: F1=93.85% ✅ **+1.46% more**

**Conclusion:** The magic is in the **fusion**, not individual modules. DQN integration matters but less than fusion.

---

## Phase 1: Att-GCM with Subgraph Splitting (2 weeks)
### Priority: HIGH | Impact: HIGH | Complexity: MEDIUM

### What RMGANets Does (Equations 5-7):

```python
# Step 1: Multi-head attention edge weighting (Eq 5-6)
A_ij = ||δ(x_i) - δ(x_j)||₂  if {x_i, x_j} ∈ x

# Step 2: Split into 3 subgraphs by similarity (Eq 7)
A^∇h = {edges where A_ij ≥ T₁}           # High similarity
A^∇m = {edges where T₂ ≤ A_ij < T₁}      # Medium similarity
A^∇l = {edges where A_ij < T₂}           # Low similarity
```

### Key Ablation Finding (Table 6, RD-Elliptic(0.4)):
- **Att-GCM**: F1=77.93%
- **With subgraph splitting + To/HyGCM**: F1=81.24% → **+3.31% gain**

### Our Implementation:

```python
# File: src/rl_money_laundering/gnn_encoder/att_gcm.py

class AttentionGCM(nn.Module):
    """
    Att-GCM: Multi-head attention graph convolution with subgraph splitting.

    Based on RMGANets Equations 1-7.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 160,
        num_heads: int = 8,
        threshold_high: float = 0.7,  # T₁ from paper
        threshold_low: float = 0.3,   # T₂ from paper
        dropout: float = 0.2
    ):
        super().__init__()

        # Multi-head attention (8 heads as per paper Section 4.2)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Initial graph convolution (Eq 1-2)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)

        self.T1 = threshold_high
        self.T2 = threshold_low

    def compute_edge_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention-based edge weights (Eq 5-6).

        Returns:
            edge_weights: Tensor of shape (num_edges,)
        """
        # Multi-head attention
        attn_output, attn_weights = self.attention(x, x, x)

        # Compute L2 distance between attended features (Eq 5)
        row, col = edge_index
        edge_weights = torch.norm(
            attn_output[row] - attn_output[col],
            p=2,
            dim=1
        )

        return edge_weights

    def split_subgraphs(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split graph into high/medium/low similarity subgraphs (Eq 7).

        Returns:
            edge_index_high, edge_index_medium, edge_index_low
        """
        # High similarity: A_ij ≥ T₁
        mask_high = edge_weights >= self.T1
        edge_index_high = edge_index[:, mask_high]

        # Medium similarity: T₂ ≤ A_ij < T₁
        mask_medium = (edge_weights >= self.T2) & (edge_weights < self.T1)
        edge_index_medium = edge_index[:, mask_medium]

        # Low similarity: A_ij < T₂
        mask_low = edge_weights < self.T2
        edge_index_low = edge_index[:, mask_low]

        return edge_index_high, edge_index_medium, edge_index_low

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass.

        Returns:
            H: Node embeddings after Att-GCM
            subgraphs: (edge_index_high, edge_index_medium, edge_index_low)
        """
        # Batch normalization (Eq 1)
        x = self.batch_norm(x)

        # Initial graph convolution (Eq 2)
        H = F.leaky_relu(self.conv1(x, edge_index))

        # Compute edge weights via attention (Eq 5-6)
        edge_weights = self.compute_edge_weights(x, edge_index)

        # Split into subgraphs (Eq 7)
        subgraphs = self.split_subgraphs(edge_index, edge_weights)

        return H, subgraphs
```

### Parameters from Paper (Section 4.2):
- **Attention heads**: 8 (h̄=8 in Eq 6)
- **Hidden units**: 160
- **Threshold T₁**: 0.7 (high similarity cutoff)
- **Threshold T₂**: 0.3 (low similarity cutoff)
- **Negative slope**: 0.01 (LeakyReLU)

### Deliverable:
- `att_gcm.py` module with subgraph splitting
- Unit test verifying 3 subgraphs created
- Threshold sensitivity analysis (T₁, T₂ tuning)

---

## Phase 2: To-GCM (Adaptive Topology Convolution) (1 week)
### Priority: HIGH | Impact: MEDIUM | Complexity: MEDIUM

### What RMGANets Does (Equation 8):

```python
# Adaptive convolution on medium + high similarity subgraphs
H₁ = σ(A^∇hm ⊙ H⁽¹⁾ ⊙ W₁) ⊗ [σ(A^∇h ⊙ H⁽¹⁾ ⊙ W₂) ⊕ σ(A^∇m ⊙ H⁽¹⁾ ⊙ W₃)]

# Where A^∇hm = A^∇h + A^∇m (merged adjacency, Eq 9)
```

### Key Ablation Finding (Table 5, RD-Elliptic(0.1)):
- **To-GCM alone**: F1=81.10%
- **Att-To-GCM (fusion)**: F1=87.92% → **+6.82% gain from fusion**

### Our Implementation:

```python
# File: src/rl_money_laundering/gnn_encoder/to_gcm.py

class AdaptiveTopoGCM(nn.Module):
    """
    To-GCM: Adaptive topology graph convolution.

    Processes medium and high similarity subgraphs with multi-scale filters.
    Based on RMGANets Equation 8.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 160,
        dropout: float = 0.2
    ):
        super().__init__()

        # Three parallel convolutions (W₁, W₂, W₃ in Eq 8)
        self.conv_merged = GCNConv(in_channels, hidden_channels)  # W₁
        self.conv_high = GCNConv(in_channels, hidden_channels)    # W₂
        self.conv_medium = GCNConv(in_channels, hidden_channels)  # W₃

        self.dropout = dropout

    def merge_adjacency(
        self,
        edge_index_high: torch.Tensor,
        edge_index_medium: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Merge high and medium similarity graphs (Eq 9).

        Creates A^∇hm where shared nodes get weight 2.
        """
        # Concatenate edges
        edge_index_merged = torch.cat([edge_index_high, edge_index_medium], dim=1)

        # Convert to adjacency matrix to handle duplicates
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index_high.device)

        # High similarity edges
        row_h, col_h = edge_index_high
        adj[row_h, col_h] = 1.0

        # Medium similarity edges (shared nodes get 2x weight)
        row_m, col_m = edge_index_medium
        adj[row_m, col_m] += 1.0  # Adds to existing 1.0 if shared

        # Convert back to edge_index
        edge_index_merged = adj.nonzero().t()
        edge_weights = adj[edge_index_merged[0], edge_index_merged[1]]

        return edge_index_merged, edge_weights

    def forward(
        self,
        H: torch.Tensor,
        edge_index_high: torch.Tensor,
        edge_index_medium: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive topology convolution (Eq 8).

        Args:
            H: Node features from Att-GCM
            edge_index_high: High similarity subgraph
            edge_index_medium: Medium similarity subgraph

        Returns:
            H₁: Adapted node features
        """
        num_nodes = H.size(0)

        # Merge high + medium subgraphs (Eq 9)
        edge_index_merged, edge_weights = self.merge_adjacency(
            edge_index_high, edge_index_medium, num_nodes
        )

        # Three parallel convolutions
        h_merged = F.leaky_relu(self.conv_merged(H, edge_index_merged, edge_weights))
        h_high = F.leaky_relu(self.conv_high(H, edge_index_high))
        h_medium = F.leaky_relu(self.conv_medium(H, edge_index_medium))

        # Feature concatenation (⊕ in Eq 8)
        h_concat = torch.cat([h_high, h_medium], dim=1)

        # Matrix multiplication (⊗ in Eq 8)
        # Simplified: element-wise multiply + residual
        H1 = h_merged * h_concat.mean(dim=1, keepdim=True)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)

        return H1
```

### Deliverable:
- `to_gcm.py` module
- Test verifying adjacency merging (2x weights for shared nodes)

---

## Phase 3: HyGCM (Hybrid Enhanced Convolution) (1 week)
### Priority: HIGH | Impact: MEDIUM | Complexity: MEDIUM

### What RMGANets Does (Equations 10-11):

```python
# Hybrid convolution on high + low similarity subgraphs
H₂ = σ(A^∇hl ⊙ H⁽¹⁾ ⊙ W₄) + LAM([σ(A^∇h ⊙ H⁽¹⁾ ⊙ W₅), σ(A^∇l ⊙ H⁽¹⁾ ⊙ W₆)])

# Local Attention Module (Eq 11)
LAM = σ(A^∇l ⊙ H⁽¹⁾ ⊙ W₄) + α · σ(A^∇h ⊙ H⁽¹⁾ ⊙ W₅)
```

### Key Ablation Finding (Table 4, Elliptic):
- **HyGCM alone**: F1=78.98%
- **Att-HyGCM**: F1=81.66% → **+2.68% gain**
- **Att-To-HyGCM**: F1=90.43% → **+8.77% gain!** (fusion is key)

### Our Implementation:

```python
# File: src/rl_money_laundering/gnn_encoder/hy_gcm.py

class HybridEnhancedGCM(nn.Module):
    """
    HyGCM: Hybrid enhanced graph convolution.

    Processes high and low similarity subgraphs with local attention.
    Based on RMGANets Equations 10-11.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 160,
        dropout: float = 0.2
    ):
        super().__init__()

        # Three parallel convolutions (W₄, W₅, W₆ in Eq 10)
        self.conv_merged = GCNConv(in_channels, hidden_channels)  # W₄
        self.conv_high = GCNConv(in_channels, hidden_channels)    # W₅
        self.conv_low = GCNConv(in_channels, hidden_channels)     # W₆

        # Attention matrix (α in Eq 11)
        self.attention = nn.Parameter(torch.ones(1))

        self.dropout = dropout

    def local_attention_module(
        self,
        h_low: torch.Tensor,
        h_high: torch.Tensor
    ) -> torch.Tensor:
        """
        Local Attention Module (LAM, Eq 11).

        Embeds high-similarity features into low-similarity space
        to compensate for lost spatial information.
        """
        # LAM = h_low + α · h_high
        lam_output = h_low + self.attention * h_high
        return lam_output

    def forward(
        self,
        H: torch.Tensor,
        edge_index_high: torch.Tensor,
        edge_index_low: torch.Tensor
    ) -> torch.Tensor:
        """
        Hybrid enhanced convolution (Eq 10-11).

        Args:
            H: Node features from Att-GCM
            edge_index_high: High similarity subgraph
            edge_index_low: Low similarity subgraph

        Returns:
            H₂: Enhanced node features
        """
        # Merge high + low subgraphs
        edge_index_merged = torch.cat([edge_index_high, edge_index_low], dim=1)

        # Three parallel convolutions
        h_merged = F.leaky_relu(self.conv_merged(H, edge_index_merged))
        h_high = F.leaky_relu(self.conv_high(H, edge_index_high))
        h_low = F.leaky_relu(self.conv_low(H, edge_index_low))

        # Local Attention Module (Eq 11)
        h_lam = self.local_attention_module(h_low, h_high)

        # Fusion (Eq 10)
        H2 = h_merged + h_lam
        H2 = F.dropout(H2, p=self.dropout, training=self.training)

        return H2
```

### Deliverable:
- `hy_gcm.py` module
- Test verifying local attention embedding

---

## Phase 4: Feature Fusion Module (1 week)
### Priority: CRITICAL | Impact: VERY HIGH | Complexity: LOW

### Key Insight from Ablation (Table 4):
- **Individual modules**: F1 ~79%
- **Att-To-HyGCM fusion**: F1=90.43% → **+11% improvement!**

**Conclusion:** Fusion is MORE important than individual module sophistication.

### What RMGANets Does (Equation 12):

```python
# Concatenate all branch features
H' = Cat(H₁, H₂, H)  # To-GCM + HyGCM + original

# DQN enhancement (embedded in last Att-GCM layer)
F' = DQN(σ(BN(H')))

# Final Att-GCM for classification
H_o = σ(Ã ⊙ F' ⊙ W' + b)
```

### Our Implementation:

```python
# File: src/rl_money_laundering/gnn_encoder/rmganets_encoder.py

class RMGANetsEncoder(nn.Module):
    """
    Complete RMGANets architecture with 3-branch fusion.

    Architecture:
    1. Att-GCM → Subgraph splitting
    2. To-GCM → Medium/high similarity processing
    3. HyGCM → High/low similarity processing
    4. Feature fusion → Concatenate all branches
    5. DQN enhancement → Fill missing features
    6. Final Att-GCM → Classification
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 160,
        num_heads: int = 8,
        embedding_dim: int = 64,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        # Module 1: Att-GCM with subgraph splitting
        self.att_gcm = AttentionGCM(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Module 2: To-GCM (adaptive topology)
        self.to_gcm = AdaptiveTopoGCM(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            dropout=dropout
        )

        # Module 3: HyGCM (hybrid enhanced)
        self.hy_gcm = HybridEnhancedGCM(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            dropout=dropout
        )

        # Feature fusion
        self.fusion_bn = nn.BatchNorm1d(hidden_dim * 3)  # Cat(H₁, H₂, H)

        # DQN enhancement (simple MLP for now)
        self.dqn_enhance = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim)
        )

        # Final Att-GCM (1 attention head for classification)
        self.final_att_gcm = GATConv(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            heads=1,
            dropout=dropout
        )

        self.device = device
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Full RMGANets forward pass.

        Args:
            x: Node features (num_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)

        Returns:
            embeddings: Node embeddings (num_nodes, embedding_dim)
        """
        # 1. Att-GCM + subgraph splitting (Eq 1-7)
        H, (edge_high, edge_medium, edge_low) = self.att_gcm(x, edge_index)

        # 2. To-GCM on medium/high similarity (Eq 8)
        H1 = self.to_gcm(H, edge_high, edge_medium)

        # 3. HyGCM on high/low similarity (Eq 10-11)
        H2 = self.hy_gcm(H, edge_high, edge_low)

        # 4. Feature fusion (Eq 12)
        H_fused = torch.cat([H1, H2, H], dim=1)  # Concatenate all branches
        H_fused = self.fusion_bn(H_fused)

        # 5. DQN enhancement (Eq 12)
        F_enhanced = self.dqn_enhance(F.leaky_relu(H_fused, 0.01))

        # 6. Final Att-GCM for classification (Eq 13)
        embeddings = self.final_att_gcm(F_enhanced, edge_index)

        return embeddings
```

### Deliverable:
- Complete `rmganets_encoder.py` with 3-branch fusion
- Integration test with our existing StateEncoder

---

## Phase 5: Multi-Branch Loss Function (3 days)
### Priority: HIGH | Impact: MEDIUM | Complexity: LOW

### What RMGANets Does (Equation 14):

```python
ζ_total = ζ_M · β + λ · (ζ_a + ζ_b) + ε · ζ_d

# Where:
# ζ_M: Primary loss (cross-entropy with class weights)
# ζ_a: Focal loss for To-GCM branch
# ζ_b: BCEWithLogitsLoss for HyGCM branch
# ζ_d: MAE loss for DQN module
# λ=0.25, ε=0.4, β=class_weight
```

### Key Ablation Finding (Table 4, Elliptic):
- **Att-To-HyGCM+DQN**: F1=92.39%
- **+ ζ_a loss**: F1=92.65% → **+0.26% gain**
- **+ ζ_b loss**: F1=92.99% → **+0.34% gain**
- **Full (ζ_a + ζ_b + ζ_d)**: F1=93.85% → **+1.46% total gain**

### Our Implementation:

```python
# File: src/rl_money_laundering/loss/multi_branch_loss.py

class MultiBranchLoss(nn.Module):
    """
    Multi-branch loss function for RMGANets.

    Based on Equation 14-15:
    ζ_total = ζ_M · β + λ · (ζ_a + ζ_b) + ε · ζ_d
    """

    def __init__(
        self,
        lambda_branch: float = 0.25,  # λ from paper
        epsilon_dqn: float = 0.4,     # ε from paper
        focal_gamma: float = 0.2,     # γ from paper (Eq 15)
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.lambda_branch = lambda_branch
        self.epsilon_dqn = epsilon_dqn
        self.focal_gamma = focal_gamma

        # Primary loss (ζ_M)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Branch losses
        self.focal_loss = self._focal_loss  # ζ_a for To-GCM
        self.bce_loss = nn.BCEWithLogitsLoss()  # ζ_b for HyGCM
        self.mae_loss = nn.L1Loss()  # ζ_d for DQN

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss (Eq 15): ζ_a = -(1 - p)^γ log(p)

        Reduces weight on easily classified samples.
        """
        probs = torch.sigmoid(logits)
        focal_weight = (1 - probs) ** self.focal_gamma
        loss = -focal_weight * torch.log(probs + 1e-8)
        return loss.mean()

    def forward(
        self,
        outputs_main: torch.Tensor,     # Final predictions
        outputs_to_gcm: torch.Tensor,   # To-GCM branch outputs
        outputs_hy_gcm: torch.Tensor,   # HyGCM branch outputs
        dqn_preds: torch.Tensor,        # DQN predictions
        dqn_targets: torch.Tensor,      # DQN targets (ground truth features)
        targets: torch.Tensor           # Ground truth labels
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute multi-branch loss.

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components (for logging)
        """
        # Primary loss (ζ_M)
        loss_main = self.ce_loss(outputs_main, targets)

        # Branch losses
        loss_focal = self.focal_loss(outputs_to_gcm, targets)  # ζ_a
        loss_bce = self.bce_loss(outputs_hy_gcm, targets.float())  # ζ_b
        loss_dqn = self.mae_loss(dqn_preds, dqn_targets)  # ζ_d

        # Combined loss (Eq 14)
        total_loss = (
            loss_main +
            self.lambda_branch * (loss_focal + loss_bce) +
            self.epsilon_dqn * loss_dqn
        )

        loss_dict = {
            'loss_main': loss_main.item(),
            'loss_focal': loss_focal.item(),
            'loss_bce': loss_bce.item(),
            'loss_dqn': loss_dqn.item(),
            'loss_total': total_loss.item()
        }

        return total_loss, loss_dict
```

### Deliverable:
- `multi_branch_loss.py` module
- Loss component visualization (track each ζ separately)

---

## Phase 6: Integration & Testing (1 week)
### Priority: CRITICAL | Impact: VERY HIGH | Complexity: HIGH

### Integration Points:

1. **Replace StateEncoder with RMGANetsEncoder**:
```python
# Before (our current code)
encoder = StateEncoder(
    node_feature_dim=12,
    gnn_type="sage",
    embedding_dim=64
)

# After (RMGANets integration)
encoder = RMGANetsEncoder(
    node_feature_dim=12,
    hidden_dim=160,
    num_heads=8,
    embedding_dim=64
)
```

2. **Update Trainer to use Multi-Branch Loss**:
```python
# In trainer.py
loss_fn = MultiBranchLoss(
    lambda_branch=0.25,
    epsilon_dqn=0.4,
    class_weights=compute_class_weights(train_labels)
)
```

3. **Keep Our Superior Components**:
- ✅ **QR-DQN agent** (better than RMGANets' basic DQN)
- ✅ **Entity-disjoint splits** (prevents leakage)
- ✅ **Budgeted metrics** (Recall@K, AUPR)
- ✅ **XGBoost baseline** (for comparison)

### Test Suite:

```python
# File: test_rmganets_integration.py

def test_att_gcm_subgraph_splitting():
    """Test Att-GCM creates 3 subgraphs (high/medium/low)."""
    ...

def test_to_gcm_adjacency_merging():
    """Test To-GCM merges high+medium with 2x weights for shared nodes."""
    ...

def test_hy_gcm_local_attention():
    """Test HyGCM local attention embedding."""
    ...

def test_feature_fusion():
    """Test 3-branch concatenation: Cat(H₁, H₂, H)."""
    ...

def test_multi_branch_loss():
    """Test loss components: ζ_M + λ(ζ_a + ζ_b) + ε·ζ_d."""
    ...

def test_end_to_end_rmganets():
    """Test complete pipeline: Data → RMGANets → QR-DQN → Metrics."""
    ...
```

### Deliverable:
- Full integration with existing codebase
- Comprehensive test suite (6 tests minimum)
- Benchmark: RMGANets vs Our Current GNN

---

## Phase 7: Hyperparameter Tuning (3 days)
### Priority: MEDIUM | Impact: MEDIUM | Complexity: LOW

### Key Parameters from Paper (Section 4.2):

```python
RMGANETS_CONFIG = {
    # Att-GCM
    'num_heads': 8,           # h̄ in Eq 6
    'threshold_high': 0.7,    # T₁ in Eq 7
    'threshold_low': 0.3,     # T₂ in Eq 7

    # Architecture
    'hidden_dim': 160,
    'num_layers': 2,          # First + second Att-GCM
    'dropout': 0.2,

    # Loss function
    'lambda_branch': 0.25,    # λ in Eq 14
    'epsilon_dqn': 0.4,       # ε in Eq 14
    'focal_gamma': 0.2,       # γ in Eq 15

    # Training
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 500,

    # Optimizer
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingWarmRestarts',
    'restart_factor': 2,
    'min_lr': 1e-6
}
```

### Sensitivity Analysis (from paper findings):

**Critical to tune (high impact):**
- T₁, T₂ thresholds → Affects subgraph quality
- λ, ε loss weights → Affects branch balance

**Less critical (low impact):**
- Number of heads (8 is optimal per paper)
- Hidden dim (160 is optimal per paper)

### Deliverable:
- Grid search over (T₁, T₂, λ, ε)
- Config file: `configs/rmganets_optimal.json`

---

## Success Metrics & Expected Results

### Based on RMGANets Paper Results (Table 1):

**Elliptic Dataset:**
- Baseline GAT: F1=93.67%, Acc=97.87%
- RMGANets: F1=93.85%, Acc=97.96% → **+0.18% F1, +0.09% Acc**

**RD-Elliptic(0.4) Dataset (with noise):**
- Baseline GAT: F1=81.56%, Acc=94.44%
- RMGANets: F1=85.27%, Acc=95.54% → **+3.71% F1, +1.10% Acc**

### Our Expected Results (AMLNet Dataset):

**Conservative Estimate:**
- Current (GAT): F1 ~85%, Acc ~94%
- With RMGANets: F1 ~88-90%, Acc ~95-96% → **+3-5% F1, +1-2% Acc**

**Best Case (if fusion works well):**
- F1 ~91-93%, Acc ~96-97% → **+6-8% F1, +2-3% Acc**

---

## Timeline Summary

| Phase | Duration | Complexity | Impact |
|-------|----------|------------|--------|
| 1. Att-GCM + Subgraph Splitting | 2 weeks | Medium | High |
| 2. To-GCM | 1 week | Medium | Medium |
| 3. HyGCM | 1 week | Medium | Medium |
| 4. Feature Fusion Module | 1 week | Low | **Very High** |
| 5. Multi-Branch Loss | 3 days | Low | Medium |
| 6. Integration & Testing | 1 week | High | **Very High** |
| 7. Hyperparameter Tuning | 3 days | Low | Medium |
| **Total** | **6-7 weeks** | | |

---

## Risk Mitigation

### High-Risk Items:
1. **Threshold tuning (T₁, T₂)**: May require extensive grid search
   - **Mitigation**: Start with paper values (0.7, 0.3), use validation set

2. **Feature fusion compatibility**: Our 20-dim features vs paper's 166-dim
   - **Mitigation**: Normalize features before fusion, use batch norm

3. **DQN integration**: Paper embeds in last layer, we use separate module
   - **Mitigation**: Implement both approaches, A/B test

### Low-Risk Items:
- Attention mechanisms (PyG has native support)
- Multi-branch loss (straightforward implementation)
- Subgraph splitting (simple masking operation)

---

## Recommendation: GO / NO-GO Decision Points

### After Phase 1 (Att-GCM):
- **GO** if: Subgraph split improves F1 by >1%
- **NO-GO** if: No improvement or regression

### After Phase 4 (Fusion):
- **GO** if: Fusion improves F1 by >3% (as per paper)
- **NO-GO** if: Improvement <1%

### After Phase 6 (Integration):
- **GO** if: RMGANets outperforms current GAT by >2% F1
- **NO-GO** if: No significant improvement

---

## Next Steps

1. **Review this plan** → Discuss, refine, get approval
2. **Phase 1 kickoff** → Implement Att-GCM + subgraph splitting
3. **Early checkpoint** → Test on 10K transactions, verify subgraphs work
4. **Iterate** → Adjust based on ablation results

**Ready to proceed?** Let me know which phase to start with, or if you want to modify the plan.
