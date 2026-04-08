"""Privacy-accuracy tradeoff visualization."""
from pathlib import Path
from typing import Dict, List
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_accuracy_privacy_tradeoff(results: Dict[float, Dict], save_path: str,
                                    baseline_acc: float = None):
    """X=epsilon, Y=test accuracy with error bars + baseline.
    results: {epsilon: {"mean": float, "std": float}}
    """
    epsilons = sorted(results.keys())
    means = [results[e]["mean"] for e in epsilons]
    stds  = [results[e].get("std", 0) for e in epsilons]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("colorblind")
    ax.errorbar(epsilons, means, yerr=stds, fmt="o-", color=colors[0],
                linewidth=2, markersize=8, capsize=5, label="DP-FL")
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color=colors[1], linestyle="--",
                   linewidth=2, label="Non-DP FL Baseline")
    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Privacy-Accuracy Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tradeoff plot → {save_path}")
