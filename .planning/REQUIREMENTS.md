# Requirements

## REQ-001: Data Pipeline
- Download and preprocess Kaggle Brain Tumor MRI Dataset
- Implement data augmentation (random flip, rotation, color jitter)
- Create non-IID data partitioning strategies (Dirichlet distribution α ∈ {0.1, 0.5, 1.0})
- Split into per-site train/val/test sets preserving class ratios
- Standard image preprocessing (resize 224×224, normalize ImageNet stats)

## REQ-002: Centralized Baseline
- Train ResNet-50 with centralized data as performance ceiling
- Implement standard training loop with early stopping
- Report accuracy, AUC-ROC, per-class F1, confusion matrix
- Serve as benchmark for all FL experiments

## REQ-003: Federated Learning Pipeline
- Implement Flower-based FL server with configurable aggregation strategies
- Implement Flower client with local training logic
- Support FedAvg, FedProx (μ parameter), and SCAFFOLD
- Configurable number of clients (3, 5), local epochs, communication rounds
- Log per-round global and per-client metrics

## REQ-004: Differential Privacy Integration
- Integrate Opacus for per-sample gradient clipping + noise injection
- Support configurable privacy budgets ε ∈ {0.5, 1.0, 2.0, 5.0, ∞}
- Track privacy accountant (RDP → ε,δ conversion)
- Evaluate accuracy-privacy tradeoff curves

## REQ-005: Fairness & Site-Level Analysis
- Compute per-site accuracy, AUC-ROC after global model convergence
- Calculate Jain fairness index across sites
- Analyze model bias across tumor types per site
- Generate per-site Grad-CAM visualizations for interpretability

## REQ-006: Ablation Studies
- Model architecture ablation: ResNet-50 vs EfficientNet-B0 vs ViT-Small
- Non-IID severity ablation: Dirichlet α ∈ {0.1, 0.5, 1.0}
- Number of clients ablation: 3 vs 5 sites
- Local epochs ablation: 1 vs 3 vs 5
- Communication rounds vs accuracy convergence curves

## REQ-007: Publication-Ready Outputs
- LaTeX-quality figures (convergence curves, privacy-accuracy tradeoffs, fairness heatmaps)
- Comprehensive results tables with mean ± std over 3 seeds
- Grad-CAM attention visualizations per tumor class
- Statistical significance tests (paired t-test / McNemar's)
- Paper draft in IEEE/MICCAI format
