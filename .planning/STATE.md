# Project State

## Current Milestone: v1.0 — Core Federated Learning Pipeline

## Decisions
- **Framework**: Flower (flwr) for federated orchestration
- **Privacy**: Opacus for differential privacy
- **Dataset**: Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar) — 7,023 images, 4 classes (glioma, meningioma, pituitary, no tumor)
- **Backbone**: ResNet-50 (primary), EfficientNet-B0 and ViT-Small (ablation)
- **FL Strategies**: FedAvg (baseline), FedProx (non-IID robust), SCAFFOLD (variance reduction)
- **Privacy Budget**: ε ∈ {0.5, 1.0, 2.0, 5.0, ∞}
- **Simulated Sites**: 3-5 hospitals with varying data distributions
- **Evaluation**: Accuracy, AUC-ROC, F1, Jain fairness index, communication cost

## History
- 2026-04-07: Project initialized with federated tumor classification scope
