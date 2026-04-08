# FedTumorNet 🧠

**Privacy-Preserving Federated Learning for Brain Tumor Classification**

A research implementation of federated learning with differential privacy for multi-hospital brain tumor MRI classification. Targets publication at MICCAI / NeurIPS Health Track.

---

## Overview

| | |
|---|---|
| **Dataset** | [Kaggle Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7,023 images, 4 classes |
| **Classes** | Glioma · Meningioma · No Tumor · Pituitary |
| **FL Framework** | [Flower (flwr)](https://flower.ai) |
| **DP Library** | [Opacus](https://opacus.ai) |
| **Backbone** | ResNet-50 (primary) · EfficientNet-B0 · ViT-Small |
| **Non-IID** | Dirichlet partitioning (α ∈ {0.1, 0.5, 1.0}) |

## Project Structure

```
Medical/
├── configs/                  # YAML configs for data, training, FL, DP, ablations
├── src/
│   ├── data/                 # Download, preprocessing, partitioning, dataset
│   ├── models/               # ResNet, EfficientNet, ViT backbones + training loop
│   ├── fl/                   # Flower client, server, strategies, utilities
│   ├── privacy/              # DP client (Opacus), privacy accountant
│   └── utils/                # Metrics, fairness (Jain index), Grad-CAM, experiments
├── scripts/                  # CLI runners for each phase
├── notebooks/                # EDA notebook
├── tests/                    # pytest test suite
├── outputs/                  # Checkpoints, figures, results
└── paper/                    # LaTeX manuscript
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API key)
python -m src.data.download

# 3. Run EDA
python notebooks/01_eda.py

# 4. Train centralized baseline
python scripts/train_centralized.py --config configs/train_config.yaml

# 5. Run federated learning (FedAvg)
python scripts/run_federated.py --strategy fedavg --alpha 0.5 --num_clients 3

# 6. Run with differential privacy
python scripts/run_dp_federated.py --epsilon 2.0

# 7. Run full DP sweep
python scripts/run_dp_federated.py --sweep

# 8. Run unit tests
pytest tests/test_data_pipeline.py -v
```

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Pipeline & Environment Setup | ✅ |
| 2 | Centralized Baseline Model | ✅ |
| 3 | Federated Learning Pipeline | ✅ |
| 4 | Differential Privacy Integration | ✅ |
| 5 | Fairness Analysis & Explainability | ✅ |
| 6 | Ablation Studies & Experiments | ✅ |
| 7 | Paper Writing & Publication Outputs | 🔄 |

## Key Metrics

- **FL vs Centralized**: Federated model achieves ≥90% of centralized accuracy
- **Privacy**: DP with ε=2.0 incurs <5% accuracy drop
- **Fairness**: Jain index >0.95, fairness gap <3% across sites

## Citation

```bibtex
@article{fedtumornet2026,
  title={FedTumorNet: Fairness-Constrained Federated Learning for
         Privacy-Preserving Brain Tumor Classification},
  author={...},
  journal={...},
  year={2026}
}
```

## ⚠️ Disclaimer

This system is for **research purposes only** and must not be used for clinical diagnosis.
