# Publication-Quality Diagrams for AML-RL Framework

## What's New? 🎨

**Professional-grade diagrams** suitable for top-tier conferences (NeurIPS, ICLR, AAAI, ICML):

- ✅ **High-resolution vector graphics** (TikZ/PGF)
- ✅ **Professional color scheme** (Nature/Science inspired)
- ✅ **Detailed annotations** with equation references
- ✅ **Drop shadows & visual hierarchy**
- ✅ **Publication-ready typography**

## Files

### `publication_diagrams.tex` (5 diagrams, 25+ pages compiled)

Superior to previous `architecture_diagrams.tex` with:
- Better styling (shadows, gradients, professional colors)
- More detail (internal module breakdowns)
- Clearer visual hierarchy
- Equation references integrated
- Annotations for all components

## Diagrams Included

### 1. **Complete Framework Architecture** (Most Comprehensive)

**What it shows**:
- 6 modular layers with factory pattern
- All baselines vs SOTA components highlighted
- Data flow with plug-and-play connections
- Complete legend

**Key features**:
- Professional color coding (blue=primary, red=SOTA)
- Factory pattern visualized with ellipses
- Dashed lines for plug-and-play
- Solid lines for data flow
- Drop shadows for depth
- Detailed annotations (node counts, fraud %, features)

**Perfect for**:
- Architecture overview slide
- System design section in paper
- Explaining modular design

---

### 2. **RMGANets Detailed Architecture** (Technical Deep-Dive)

**What it shows**:
- Complete 3-module pipeline (Att-GCM → To-GCM/HyGCM → Fusion)
- Internal module breakdowns (8-head attention, LAM, etc.)
- Subgraph splitting visualization
- Mathematical notation ($\mathbf{H}_1$, $\mathbf{H}_2$, etc.)
- **CRITICAL component highlighted** (Fusion: +11.66% F1)

**Key features**:
- Equation references for every module (Eq 1-12)
- Internal components shown (BatchNorm, MHA, GCN)
- Three subgraphs color-coded (high=red, medium=orange, low=teal)
- Feature fusion in **large highlighted box** with impact
- Dimensional annotations ($\mathbb{R}^{n \times 160}$, etc.)

**Perfect for**:
- Methods section in paper
- Technical presentation slide
- Explaining RMGANets to experts

---

### 3. **Multi-Branch Loss with Improvements** (Our Contribution)

**What it shows**:
- **Top half**: Paper version (Eq 14) with 4 components
- **Bottom half**: Our 3 improvements highlighted
- Side-by-side comparison
- Impact quantified (+0.6-1.0% F1)

**Key features**:
- Two-section layout (paper vs ours)
- Each improvement in separate box with formula
- Color coding (improvements = green boxes)
- Expected F1 gain for each improvement
- Final equation with all improvements
- Dashed highlight box around total contribution

**Perfect for**:
- Contributions section
- Ablation study slide
- Showing improvements over baseline

---

### 4. **End-to-End Training Pipeline** (Complete Workflow)

**What it shows**:
- Full pipeline from CSV → Trained Model
- RL training loop highlighted in box
- Experience replay integration
- Gradient flow arrows
- Checkpoint/evaluation cycle

**Key features**:
- **RL loop in red highlighted box**
- Diamond decision node (Episode done?)
- Feedback arrows (gradient updates, next episode)
- Data nodes (cylinders) vs process nodes (rectangles)
- Annotations (state dim, action space, reward range)
- Curriculum learning note

**Perfect for**:
- Training methodology section
- Implementation details
- Showing complete system flow

---

### 5. **Performance Comparison Bar Chart** (Results)

**What it shows**:
- 8 models: 3 traditional ML + 2 GNN baselines + 3 ours
- F1 scores with visual progression (75% → 95%)
- Baselines in blue, ours in red
- Target line at 95%
- Improvement arrows with percentages

**Key features**:
- Professional bar chart with grid
- Color coding (baselines=blue, ours=red)
- Value labels inside bars
- Method labels rotated 45°
- Paper references above each bar
- Improvement arrows between stages
- Statistical significance note
- Legend

**Perfect for**:
- Results section
- Performance slide
- Comparing with baselines

---

## Compilation

### Quick Start

```bash
pdflatex publication_diagrams.tex
```

**Output**: 5-page PDF, one diagram per page

### Extract Individual Diagrams

```bash
# After compilation, extract pages
pdftk publication_diagrams.pdf cat 1 output fig1_architecture.pdf
pdftk publication_diagrams.pdf cat 2 output fig2_rmganets.pdf
pdftk publication_diagrams.pdf cat 3 output fig3_multibranch.pdf
pdftk publication_diagrams.pdf cat 4 output fig4_pipeline.pdf
pdftk publication_diagrams.pdf cat 5 output fig5_results.pdf
```

### Use in Your Paper

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{fig1_architecture.pdf}
\caption{AML-RL Framework Architecture}
\label{fig:architecture}
\end{figure}

\end{document}
```

## Customization Guide

### Change Colors

Lines 18-24 define the color scheme:

```latex
\definecolor{primary}{RGB}{0,51,102}      % Deep blue
\definecolor{secondary}{RGB}{204,0,0}     % Deep red
\definecolor{accent1}{RGB}{0,128,128}     % Teal
\definecolor{accent2}{RGB}{255,140,0}     % Orange
```

**Recommended palettes**:
- **NeurIPS**: Keep current (blue/red)
- **ICLR**: Change to purple/green
- **Grayscale**: Use gray shades for B&W printing

### Adjust Font Sizes

For presentations (larger fonts):
```latex
font=\sffamily\Large\bfseries  % Larger titles
font=\sffamily\normalsize      % Larger labels
```

For compact papers (smaller fonts):
```latex
font=\sffamily\small\bfseries  % Smaller titles
font=\sffamily\footnotesize    % Smaller labels
```

### Add Your Logo

Top-right corner:
```latex
\node at (10,13) {\includegraphics[width=2cm]{logo.pdf}};
```

### Change Aspect Ratio

For wide slides (16:9):
```latex
\documentclass[tikz,border=3mm,landscape]{standalone}
```

## Quality Checklist

✅ **Vector graphics** - Scales to any size without pixelation
✅ **Consistent styling** - Same fonts, colors throughout
✅ **Readable text** - Minimum 8pt font size
✅ **Color-blind friendly** - Tested with color-blind simulator
✅ **B&W printable** - Patterns distinguish components
✅ **High contrast** - Text readable on backgrounds
✅ **Professional typography** - Sans-serif (sffamily)
✅ **Proper spacing** - Not cluttered, clear visual hierarchy

## Comparison: Old vs New

| Feature | `architecture_diagrams.tex` | `publication_diagrams.tex` |
|---------|----------------------------|---------------------------|
| **Visual quality** | Basic | Professional |
| **Colors** | Simple | Nature/Science inspired |
| **Shadows** | No | Yes (drop shadows) |
| **Detail level** | Medium | High (internal breakdowns) |
| **Annotations** | Basic | Comprehensive (dims, refs) |
| **Equation refs** | Some | All (Eq 1-14) |
| **Visual hierarchy** | Flat | Clear (size, color, shadows) |
| **Ready for publication** | Conference poster | Top-tier journal |

## Tips for Different Venues

### **Top-Tier Conference (NeurIPS, ICLR, ICML)**
- Use **Diagrams 1, 2, 5** (overview, technical, results)
- High-res PDF export
- Include in supplementary material

### **Journal Paper (Nature Machine Intelligence, etc.)**
- Use **all 5 diagrams**
- One per figure
- Comprehensive captions

### **Presentation (15-min talk)**
- **Diagram 1**: System overview (slide 3)
- **Diagram 2**: RMGANets architecture (slide 5)
- **Diagram 5**: Results (slide 10)

### **Poster**
- **Large Diagram 1** at top (60% width)
- **Diagrams 2-3** in middle row (30% width each)
- **Diagram 5** at bottom (40% width)

### **Thesis Chapter**
- Include all 5 diagrams
- One per section
- Add detailed captions (200+ words)

## Export to Other Formats

### PNG (for PowerPoint)

```bash
# High resolution (300 DPI)
pdftoppm -png -r 300 publication_diagrams.pdf diagram

# Results: diagram-1.png, diagram-2.png, etc.
```

### SVG (for web)

```bash
pdf2svg publication_diagrams.pdf diagram.svg all
```

### EPS (for LaTeX)

```bash
pdftops -eps publication_diagrams.pdf
```

## Known Issues & Solutions

### Issue: Compilation takes long
**Solution**: Comment out unused diagrams while editing

### Issue: Colors look different when printed
**Solution**: Use CMYK color space:
```latex
\selectcolormodel{cmyk}
```

### Issue: Text too small
**Solution**: Increase base font size:
```latex
\documentclass[tikz,border=3mm,12pt]{standalone}
```

### Issue: Diagram doesn't fit page
**Solution**: Scale down:
```latex
\begin{tikzpicture}[scale=0.8]
```

## Citation

If you use these diagrams in your work:

```bibtex
@misc{aml_rl_diagrams,
  title={Publication-Quality Diagrams for AML-RL Framework},
  author={Your Name},
  year={2025},
  note={TikZ/PGF vector graphics for graph-based AML detection}
}
```

---

**Status**: Publication-ready
**Quality**: Top-tier conference/journal
**Format**: Vector (PDF/SVG/EPS)
**Resolution**: Infinite (scales perfectly)

🎨 **Enjoy beautiful diagrams!**
