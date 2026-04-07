# Roadmap — FedTumorNet v1.0

## Phase 1: Data Pipeline & Environment Setup
**Goal:** Set up the development environment, download the Kaggle Brain Tumor MRI dataset, and build the data pipeline with non-IID partitioning.
**Requirements:** REQ-001
**Deliverables:** Working data loaders, non-IID partitioning, preprocessing pipeline, EDA notebook

## Phase 2: Centralized Baseline Model
**Goal:** Train a centralized ResNet-50 classifier as the performance ceiling benchmark.
**Requirements:** REQ-002
**Deliverables:** Trained centralized model, baseline metrics (accuracy, AUC-ROC, F1), confusion matrix

## Phase 3: Federated Learning Pipeline
**Goal:** Implement the core FL pipeline using Flower with FedAvg, FedProx, and SCAFFOLD strategies.
**Requirements:** REQ-003
**Deliverables:** Working FL server + clients, convergence curves, FL vs centralized comparison

## Phase 4: Differential Privacy Integration
**Goal:** Add Opacus-based differential privacy to FL training and evaluate privacy-accuracy tradeoffs.
**Requirements:** REQ-004
**Deliverables:** DP-FL pipeline, privacy budget curves, ε-accuracy tradeoff analysis

## Phase 5: Fairness Analysis & Explainability
**Goal:** Evaluate per-site fairness, Grad-CAM interpretability, and site-level bias analysis.
**Requirements:** REQ-005
**Deliverables:** Fairness metrics, Grad-CAM visualizations, per-site performance reports

## Phase 6: Ablation Studies & Experiments
**Goal:** Run comprehensive ablation studies across architectures, non-IID levels, and hyperparameters.
**Requirements:** REQ-006
**Deliverables:** Ablation tables, convergence comparisons, statistical significance tests

## Phase 7: Paper Writing & Publication-Ready Outputs
**Goal:** Generate publication-quality figures, results tables, and draft the research paper.
**Requirements:** REQ-007
**Deliverables:** LaTeX paper draft, camera-ready figures, supplementary materials
