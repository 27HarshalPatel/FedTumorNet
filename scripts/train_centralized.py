"""Centralized baseline training script.
Usage (from anywhere):
  python scripts/train_centralized.py
  python scripts/train_centralized.py --config configs/train_config.yaml
"""
import os, sys
from pathlib import Path

# Ensure CWD is always the project root, regardless of where this script is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Train centralized ResNet-50 baseline")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("FedTumorNet — Centralized Baseline Training")
    print("=" * 60)
    print(f"Model: {config['training']['model']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")

    from src.models.train import train_centralized
    model, tracker, results = train_centralized(config)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {results['test_acc']:.4f}")
    print(f"Macro AUC-ROC : {results['auc']['macro']:.4f}")
    print(f"Macro F1      : {results['f1']['macro']:.4f}")
    print(f"Best Val Acc  : {tracker.best_val_acc():.4f} @ epoch {tracker.best_epoch()+1}")
    print("\n✅ Centralized training complete!")
    print("  Outputs:")
    print("  - outputs/checkpoints/centralized_best.pt")
    print("  - outputs/figures/centralized_confusion_matrix.png")
    print("  - outputs/figures/centralized_training_curves.png")

if __name__ == "__main__":
    main()
