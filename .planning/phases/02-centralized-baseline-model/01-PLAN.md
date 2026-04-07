---
phase: 2
plan: 01
title: "Centralized Baseline Model"
wave: 1
depends_on: ["phase-1"]
files_modified:
  - src/models/resnet.py
  - src/models/train.py
  - src/models/evaluate.py
  - src/utils/metrics.py
  - configs/train_config.yaml
  - scripts/train_centralized.py
  - tests/test_model.py
requirements_addressed: [REQ-002]
autonomous: true
---

# Phase 2: Centralized Baseline Model

<objective>
Train a centralized ResNet-50 classifier on the full Brain Tumor MRI dataset as the performance ceiling. This serves as the upper-bound benchmark against which all federated learning experiments are compared.
</objective>

## Tasks

<task id="2.1" title="Model Architecture & Config">
<read_first>
- configs/data_config.yaml
- src/data/dataset.py
</read_first>

<action>
Create `src/models/resnet.py`:
1. `get_resnet50(num_classes=4, pretrained=True)` — loads torchvision ResNet-50 with ImageNet weights, replaces final FC layer with `nn.Linear(2048, num_classes)`
2. `get_efficientnet_b0(num_classes=4, pretrained=True)` — EfficientNet-B0 backbone (for later ablation)
3. `get_vit_small(num_classes=4, pretrained=True)` — ViT-Small 16x16 backbone (for later ablation)
4. `get_model(model_name, num_classes=4)` — factory function dispatching by name

Create `configs/train_config.yaml`:
```yaml
training:
  model: "resnet50"
  num_classes: 4
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler: "cosine"
  num_epochs: 50
  early_stopping_patience: 10
  seed: 42
  device: "auto"  # auto-detect cuda/mps/cpu

logging:
  save_dir: "outputs/checkpoints"
  log_interval: 10
  use_wandb: false
```
</action>

<acceptance_criteria>
- `src/models/resnet.py` contains `def get_resnet50`
- `src/models/resnet.py` contains `nn.Linear(2048, num_classes)` or equivalent
- `src/models/resnet.py` contains `def get_model`
- `configs/train_config.yaml` contains `model: "resnet50"`
- `configs/train_config.yaml` contains `early_stopping_patience: 10`
</acceptance_criteria>
</task>

<task id="2.2" title="Metrics & Evaluation Utilities">
<read_first>
- src/models/resnet.py
</read_first>

<action>
Create `src/utils/metrics.py`:
1. `compute_accuracy(preds, labels)` → float
2. `compute_auc_roc(probs, labels, num_classes=4)` → per-class AUC + macro AUC using `sklearn.metrics.roc_auc_score` with `multi_class='ovr'`
3. `compute_f1(preds, labels, num_classes=4)` → per-class F1 + macro F1
4. `compute_confusion_matrix(preds, labels, class_names)` → saves seaborn heatmap
5. `compute_classification_report(preds, labels, class_names)` → returns sklearn classification_report dict
6. `MetricsTracker` class — accumulates per-epoch metrics, supports `save_to_json(path)` and `plot_training_curves(save_path)`
</action>

<acceptance_criteria>
- `src/utils/metrics.py` contains `def compute_accuracy`
- `src/utils/metrics.py` contains `def compute_auc_roc`
- `src/utils/metrics.py` contains `roc_auc_score`
- `src/utils/metrics.py` contains `class MetricsTracker`
- `src/utils/metrics.py` contains `confusion_matrix`
</acceptance_criteria>
</task>

<task id="2.3" title="Training Loop & Evaluation">
<read_first>
- src/models/resnet.py
- src/utils/metrics.py
- src/data/dataset.py
</read_first>

<action>
Create `src/models/train.py`:
1. `train_one_epoch(model, dataloader, optimizer, criterion, device)` → returns avg loss, accuracy
2. `evaluate(model, dataloader, criterion, device)` → returns loss, accuracy, all predictions & probabilities
3. `train_centralized(config)`:
   - Loads data using `get_centralized_dataloaders()`
   - Creates model, optimizer (Adam), scheduler (CosineAnnealingLR), criterion (CrossEntropyLoss)
   - Training loop with early stopping based on validation loss
   - Saves best model checkpoint to `outputs/checkpoints/centralized_best.pt`
   - Logs training/validation metrics per epoch
   - Returns MetricsTracker with full history

Create `scripts/train_centralized.py`:
- CLI entry point: `python scripts/train_centralized.py --config configs/train_config.yaml`
- After training, runs final evaluation on test set
- Saves: confusion matrix, classification report, training curves, AUC-ROC curve
- Prints final table:
  ```
  ┌─────────────────────────────────────────────┐
  │ Centralized Baseline Results                │
  ├─────────────┬───────┬───────┬───────┬───────┤
  │ Metric      │ Glioma│ Menin │ NoTum │ Pitui │
  ├─────────────┼───────┼───────┼───────┼───────┤
  │ Precision   │ 0.XX  │ 0.XX  │ 0.XX  │ 0.XX  │
  │ Recall      │ 0.XX  │ 0.XX  │ 0.XX  │ 0.XX  │
  │ F1-Score    │ 0.XX  │ 0.XX  │ 0.XX  │ 0.XX  │
  │ AUC-ROC     │ 0.XX  │ 0.XX  │ 0.XX  │ 0.XX  │
  └─────────────┴───────┴───────┴───────┴───────┘
  ```
</action>

<acceptance_criteria>
- `src/models/train.py` contains `def train_one_epoch`
- `src/models/train.py` contains `def evaluate`
- `src/models/train.py` contains `def train_centralized`
- `src/models/train.py` contains `CosineAnnealingLR` or equivalent scheduler
- `src/models/train.py` contains early stopping logic
- `scripts/train_centralized.py` saves model to `outputs/checkpoints/centralized_best.pt`
- `scripts/train_centralized.py` generates confusion matrix figure
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. Centralized ResNet-50 achieves ≥92% test accuracy on Brain Tumor MRI dataset
2. Confusion matrix saved to `outputs/figures/centralized_confusion_matrix.png`
3. Training convergence curves (loss + accuracy) saved
4. Per-class AUC-ROC > 0.95
5. Model checkpoint saved to `outputs/checkpoints/centralized_best.pt`
</must_haves>
