# LaTeX Compilation Guide

## Files Created

1. **`FRAMEWORK_ARCHITECTURE.tex`** - Main paper (17 pages)
   - Complete framework description
   - Module reference with equations
   - Performance benchmarks
   - Implementation details

2. **`architecture_diagrams.tex`** - Standalone diagrams (5 figures)
   - Full system architecture
   - RMGANets module flow
   - Multi-branch loss components
   - Training pipeline
   - Performance bar chart

## Compilation Instructions

### Option 1: Compile Main Document

```bash
cd rl_money_laundering
pdflatex FRAMEWORK_ARCHITECTURE.tex
pdflatex FRAMEWORK_ARCHITECTURE.tex  # Run twice for references
```

Output: `FRAMEWORK_ARCHITECTURE.pdf` (~17 pages)

### Option 2: Compile Diagrams Only

```bash
cd rl_money_laundering
pdflatex architecture_diagrams.tex
```

Output: `architecture_diagrams.pdf` (5 separate pages, one per diagram)

### Option 3: Use Overleaf (Recommended)

1. Go to https://www.overleaf.com
2. Create new project
3. Upload both `.tex` files
4. Compile with `pdfLaTeX`

## What's Included

### Main Document Structure

```
§1  Introduction
    - Framework philosophy
    - Key contributions

§2  Framework Architecture
    - System overview (6 layers)
    - Factory pattern design

§3  Module Reference
    - Data Layer (loaders, graph construction)
    - Feature Extraction (20-dim breakdown)
    - Graph Encoding (SAGE, GAT, RMGANets)
    - Multi-Branch Loss (paper + improvements)
    - RL Layer (environment, QR-DQN)

§4  Performance Benchmarks
    - Expected results (Table 3)
    - Ablation study (Table 4)

§5  Implementation Details
    - Hyperparameters (Table 5)
    - Code statistics (Table 6)

§6  Usage Example
    - Quick start code

§7  Roadmap
    - Development phases

References (4 papers)
```

### Diagrams Included

1. **Full System Architecture** (Page 1)
   - 6 layers with factory pattern
   - Component breakdown
   - Data flow arrows

2. **RMGANets Module Flow** (Page 2)
   - Att-GCM → subgraph split
   - To-GCM + HyGCM parallel processing
   - Feature fusion
   - All equations referenced

3. **Multi-Branch Loss Components** (Page 3)
   - 4 loss components
   - Our 3 improvements highlighted
   - Complete equation

4. **Training Pipeline** (Page 4)
   - Full data flow
   - RL loop
   - Experience replay
   - Checkpointing

5. **Performance Bar Chart** (Page 5)
   - 8 models compared
   - 75% → 95% progression
   - Baselines vs our framework

## All Paper References Included

### RMGANets Paper
- **Att-GCM**: Equations 1-7 (§3.2.1)
- **To-GCM**: Equations 8-9 (§3.2.2)
- **HyGCM**: Equations 10-11 (§3.2.3)
- **Fusion**: Equation 12 (§3.2.4)
- **Multi-Branch Loss**: Equation 14 (§3.3)
- **Hyperparameters**: Section 4.2
- **Ablation Studies**: Tables 4-6

### AMLNet Paper
- **Algorithm 2**: Temporal feature extraction
- **Dataset**: Section 3.1
- **Features**: Section 4.2
- **Baselines**: Table 3

### RL Papers
- **QR-DQN**: Dabney et al. 2018
- **Prioritized Replay**: Schaul et al. 2016

## Key Highlights in Document

### Mathematical Equations

✅ All RMGANets equations (1-14) with LaTeX formatting
✅ Subgraph splitting formulas
✅ Loss function breakdowns
✅ QR-DQN quantile regression

### Tables

✅ Table 1: Dataset loaders
✅ Table 2: Network feature complexity
✅ Table 3: RMGANets modules (with LOC)
✅ Table 4: Performance comparison
✅ Table 5: Ablation results
✅ Table 6: Hyperparameters
✅ Table 7: Code statistics
✅ Table 8: Roadmap

### Algorithms

✅ Algorithm 1: Temporal feature extraction (AMLNet)

### Code Listings

✅ Factory pattern example
✅ Quick start example

## Customization

### Add Your Name

Line 23:
```latex
\author{Your Name Here}
```

### Adjust Paper Size

Line 8:
```latex
\documentclass[11pt,a4paper]{article}  % or letter
```

### Change Colors

Lines 17-20:
```latex
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
```

## Usage in Presentation

These diagrams can be exported as individual PDFs:

```bash
# Extract page 1 (system architecture)
pdftk architecture_diagrams.pdf cat 1 output system_architecture.pdf

# Extract page 2 (RMGANets flow)
pdftk architecture_diagrams.pdf cat 2 output rmganets_flow.pdf
```

Then include in PowerPoint/Keynote or LaTeX Beamer presentations.

## Citation

```bibtex
@software{aml_rl_framework,
  title={AML-RL: A Modular Framework for Graph-Based Anti-Money Laundering},
  author={Your Name},
  year={2025},
  note={Built on PyTorch Geometric + Gymnasium}
}
```

## Tips

1. **Compile twice** to resolve references
2. **Use XeLaTeX** if you have Unicode issues
3. **Check TikZ** if diagrams don't render (install `tikz` package)
4. **View in Overleaf** for easiest editing

---

**Status**: Ready to compile!
**Pages**: 17 (main) + 5 (diagrams)
**Equations**: All RMGANets (1-14) + improvements
**Tables**: 8 comprehensive tables
**Figures**: 5 TikZ diagrams
