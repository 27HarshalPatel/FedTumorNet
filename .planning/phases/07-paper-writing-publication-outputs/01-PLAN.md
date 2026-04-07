---
phase: 7
plan: 01
title: "Paper Writing & Publication-Ready Outputs"
wave: 1
depends_on: ["phase-5", "phase-6"]
files_modified:
  - paper/main.tex
  - paper/references.bib
  - paper/figures/
  - paper/tables/
  - scripts/generate_paper_figures.py
requirements_addressed: [REQ-007]
autonomous: false
---

# Phase 7: Paper Writing & Publication-Ready Outputs

<objective>
Generate all publication-quality figures, compile results into LaTeX tables, and draft the research paper in IEEE/MICCAI format targeting top-tier venues.
</objective>

## Tasks

<task id="7.1" title="Figure Generation Script">
<read_first>
- outputs/ablations/
- outputs/dp_experiments/
- outputs/fairness/
</read_first>

<action>
Create `scripts/generate_paper_figures.py`:

Generate the following publication figures (300 DPI, PDF format):

1. **Figure 1: System Architecture** — Mermaid/TikZ diagram of the federated learning pipeline
2. **Figure 2: Non-IID Data Distribution** — Stacked bar chart showing class distribution across clients for α ∈ {0.1, 0.5, 1.0}
3. **Figure 3: Convergence Curves** — Global accuracy vs rounds for FedAvg/FedProx with centralized baseline
4. **Figure 4: Privacy-Accuracy Tradeoff** — Test accuracy vs ε with error bars + centralized baseline
5. **Figure 5: Fairness Heatmap** — Sites × tumor types accuracy heatmap
6. **Figure 6: Grad-CAM Comparison** — Centralized vs FL vs DP-FL attention maps
7. **Figure 7: Ablation Bar Charts** — Multi-panel figure for architecture + client scaling + local epochs

Style requirements:
- Font: Computer Modern (LaTeX-compatible)
- Color palette: colorblind-safe (seaborn "colorblind")
- Grid: light gray, no top/right spines
- Legend: outside plot area where possible
- All text size ≥ 8pt for readability at column-width
</action>

<acceptance_criteria>
- `scripts/generate_paper_figures.py` generates at least 7 figures
- Figures saved as PDF to `paper/figures/`
- Figures use colorblind-safe palette
- DPI is 300 or higher
</acceptance_criteria>
</task>

<task id="7.2" title="Paper Structure & LaTeX Template">
<read_first>
- outputs/ablations/ (for results)
</read_first>

<action>
Create `paper/main.tex` with the following structure:

```latex
% IEEE/MICCAI format
\documentclass[conference]{IEEEtran}

\title{FedTumorNet: A Fairness-Constrained Federated Learning Framework 
       for Privacy-Preserving Brain Tumor Classification}

\begin{abstract}
% ~150 words covering:
% 1. Problem: hospitals can't share MRI data
% 2. Method: FL + DP + fairness constraints
% 3. Results: X% accuracy, ε=Y privacy, Z fairness index
% 4. Significance: first to combine FL + DP + fairness for tumor classification
\end{abstract}

% Section 1: Introduction (motivation, contributions)
% Section 2: Related Work (FL in medical imaging, DP-FL, fairness in ML)
% Section 3: Methodology
%   3.1 Problem Formulation
%   3.2 Federated Learning Framework
%   3.3 Differential Privacy Integration
%   3.4 Fairness-Aware Evaluation
% Section 4: Experimental Setup
%   4.1 Dataset (Kaggle Brain Tumor MRI, 7K images, 4 classes)
%   4.2 Non-IID Simulation (Dirichlet partitioning)
%   4.3 Implementation Details
%   4.4 Baselines and Metrics
% Section 5: Results
%   5.1 Centralized vs Federated Performance
%   5.2 Privacy-Accuracy Tradeoff
%   5.3 Fairness Analysis
%   5.4 Ablation Studies
%   5.5 Explainability Analysis
% Section 6: Discussion
% Section 7: Conclusion
```

Create `paper/references.bib` with citations for:
- McMahan et al. (FedAvg, 2017)
- Li et al. (FedProx, 2020)
- Abadi et al. (Deep Learning with DP, 2016)
- Sheller et al. (Federated Learning for Brain Tumors, 2020)
- FeTS Challenge papers (2022-2024)
- Opacus, Flower framework citations
</action>

<acceptance_criteria>
- `paper/main.tex` exists with IEEE format class
- `paper/main.tex` contains all 7 sections
- `paper/references.bib` contains at least 15 references
- Paper title contains "Federated" and "Privacy"
</acceptance_criteria>
</task>

<task id="7.3" title="Results Tables in LaTeX">
<read_first>
- outputs/ablations/ (for experimental results)
</read_first>

<action>
Create LaTeX tables in `paper/tables/`:

1. **Table 1: Dataset Statistics** — per-class counts, train/val/test splits
2. **Table 2: Centralized vs Federated Comparison** — accuracy, AUC, F1 for each strategy
3. **Table 3: Privacy-Accuracy Tradeoff** — ε vs accuracy for FedAvg + FedProx
4. **Table 4: Fairness Metrics** — Jain index, fairness gap per strategy
5. **Table 5: Ablation Results** — architecture, non-IID, clients, epochs

All tables: mean ± std format, bold best results, include p-values where applicable.
</action>

<acceptance_criteria>
- `paper/tables/` directory contains at least 5 `.tex` files
- Tables use `\begin{table}` LaTeX environment
- Tables include `mean ± std` notation
- Best results bolded with `\textbf{}`
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. All 7 paper figures generated as PDF at 300 DPI
2. LaTeX paper compiles without errors
3. At least 5 results tables in LaTeX format
4. References include ≥15 citations
5. Abstract is ≤150 words and covers problem/method/results/significance
</must_haves>
