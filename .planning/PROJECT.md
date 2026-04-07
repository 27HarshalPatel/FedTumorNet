# FedTumorNet — Federated Tumor Classification with Privacy Guarantees

## Vision
Build a privacy-preserving deep learning system that enables multiple simulated hospitals to collaboratively train a brain tumor classifier without sharing raw patient MRI data. The system uses federated learning with differential privacy to ensure patient data protection while achieving performance comparable to centralized training.

## Problem Statement
- Brain tumor classification requires large, diverse datasets for robust AI models
- HIPAA/GDPR prevents hospitals from sharing raw patient MRI data
- Individual hospitals have insufficient data for training reliable classifiers
- Existing FL research lacks rigorous fairness evaluation across heterogeneous sites

## Key Goals
1. Implement a federated learning pipeline using Flower framework + PyTorch
2. Simulate multi-hospital scenarios with realistic non-IID data distributions
3. Integrate differential privacy (ε-DP) to quantify privacy guarantees
4. Evaluate fairness across participating sites using Jain fairness index
5. Benchmark FL approaches (FedAvg, FedProx, SCAFFOLD) against centralized training
6. Produce a publishable paper with rigorous evaluation

## Tech Stack
- **Deep Learning**: PyTorch, torchvision
- **Federated Learning**: Flower (flwr)
- **Privacy**: Opacus (differential privacy for PyTorch)
- **Models**: ResNet-50, EfficientNet-B0, ViT-Small
- **Dataset**: Kaggle Brain Tumor MRI Dataset (7K+ images, 4 classes)
- **Visualization**: matplotlib, seaborn, Grad-CAM
- **Experiment Tracking**: Weights & Biases / TensorBoard

## Success Criteria
- FL model achieves ≥90% of centralized model accuracy
- Differential privacy (ε ≤ 2.0) with <5% accuracy drop
- Fairness gap <3% across simulated hospital sites
- Paper-quality figures and ablation studies
