"""Metrics utilities: accuracy, AUC-ROC, F1, confusion matrix, tracker."""
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                              confusion_matrix, accuracy_score)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

def compute_accuracy(preds, labels) -> float:
    return float(accuracy_score(labels, preds))

def compute_auc_roc(probs, labels, num_classes=4) -> Dict:
    """Per-class + macro AUC-ROC (OVR strategy)."""
    import torch
    if hasattr(probs, "numpy"): probs = probs.numpy()
    if hasattr(labels, "numpy"): labels = labels.numpy()
    try:
        macro = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        per_cls = {}
        for c in range(num_classes):
            bin_labels = (np.array(labels) == c).astype(int)
            per_cls[CLASS_NAMES[c]] = float(roc_auc_score(bin_labels, np.array(probs)[:,c]))
        return {"macro": macro, "per_class": per_cls}
    except Exception as e:
        return {"macro": 0.0, "per_class": {n: 0.0 for n in CLASS_NAMES}, "error": str(e)}

def compute_f1(preds, labels, num_classes=4) -> Dict:
    macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_cls = {CLASS_NAMES[c]: float(f1_score(labels, preds, labels=[c],
               average="macro", zero_division=0)) for c in range(num_classes)}
    return {"macro": macro, "per_class": per_cls}

def compute_confusion_matrix(preds, labels, class_names=CLASS_NAMES, save_path=None):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm

def compute_classification_report(preds, labels, class_names=CLASS_NAMES) -> Dict:
    return classification_report(labels, preds, target_names=class_names,
                                 output_dict=True, zero_division=0)

class MetricsTracker:
    """Accumulates per-epoch training/validation metrics."""
    def __init__(self):
        self.history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    def record(self, phase: str, loss: float, acc: float):
        self.history[f"{phase}_loss"].append(loss)
        self.history[f"{phase}_acc"].append(acc)

    def best_val_acc(self) -> float:
        return max(self.history["val_acc"]) if self.history["val_acc"] else 0.0

    def best_epoch(self) -> int:
        return int(np.argmax(self.history["val_acc"])) if self.history["val_acc"] else 0

    def save_to_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: json.dump(self.history, f, indent=2)

    def plot_training_curves(self, save_path: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        ax1.plot(epochs, self.history["train_loss"], label="Train", color="#2196F3")
        ax1.plot(epochs, self.history["val_loss"],   label="Val",   color="#F44336")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs, self.history["train_acc"], label="Train", color="#2196F3")
        ax2.plot(epochs, self.history["val_acc"],   label="Val",   color="#F44336")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
        plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Curves saved → {save_path}")
