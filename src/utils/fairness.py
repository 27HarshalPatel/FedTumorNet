"""Fairness metrics: Jain index, per-site analysis, fairness heatmap."""
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from src.utils.metrics import compute_accuracy, compute_auc_roc, CLASS_NAMES
from src.models.train import get_device, evaluate as eval_model

def jain_fairness_index(accuracies: List[float]) -> float:
    """Jain's fairness index: J = (Σxi)² / (n * Σxi²). Ranges [1/n, 1]."""
    x = np.array(accuracies, dtype=float)
    n = len(x)
    if n == 0: return 0.0
    denom = n * np.sum(x**2)
    return float(np.sum(x)**2 / denom) if denom > 0 else 0.0

def fairness_gap(site_metrics: Dict) -> float:
    """Max accuracy - min accuracy across sites."""
    accs = [m["accuracy"] for m in site_metrics.values()]
    return float(max(accs) - min(accs)) if accs else 0.0

def compute_site_metrics(model: nn.Module, site_dataloaders: Dict,
                         device=None) -> Dict:
    """Evaluate global model on each site's test DataLoader."""
    if device is None: device = get_device()
    criterion = nn.CrossEntropyLoss()
    site_metrics = {}
    for site_id, loader in site_dataloaders.items():
        loss, acc, preds, labels, probs = eval_model(model, loader, criterion, device)
        auc_result = compute_auc_roc(np.array(probs), labels)
        per_class_acc = {}
        labels_arr = np.array(labels); preds_arr = np.array(preds)
        for c, name in enumerate(CLASS_NAMES):
            mask = labels_arr == c
            per_class_acc[name] = float(np.mean(preds_arr[mask] == c)) if mask.sum() > 0 else 0.0
        site_metrics[site_id] = {
            "accuracy": acc, "loss": loss,
            "auc": auc_result.get("macro", 0.0), "per_class_acc": per_class_acc,
        }
    return site_metrics

def plot_fairness_heatmap(site_metrics: Dict, class_names=CLASS_NAMES, save_path=None):
    """Heatmap: sites (rows) × classes (cols) with per-class accuracy."""
    import pandas as pd
    data = {}
    for site_id, m in site_metrics.items():
        data[f"Hospital {site_id+1}"] = m["per_class_acc"]
    df = pd.DataFrame(data).T.reindex(columns=class_names)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df)*1.2)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title("Per-Site Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tumor Type"); ax.set_ylabel("Hospital Site")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Fairness heatmap → {save_path}")
    plt.close(fig)

def plot_fairness_comparison(strategy_metrics: Dict, save_path=None):
    """Grouped bar comparing Jain index across FL strategies."""
    strategies = list(strategy_metrics.keys())
    jain_scores = [jain_fairness_index([m["accuracy"] for m in sm.values()])
                   for sm in strategy_metrics.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("colorblind", len(strategies))
    bars = ax.bar(strategies, jain_scores, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="{:.3f}", padding=3)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Jain Fairness Index")
    ax.set_title("Fairness Across FL Strategies", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
