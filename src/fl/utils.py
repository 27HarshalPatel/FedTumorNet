"""FL visualization and results utilities."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_fl_convergence(history: dict, save_path: str, title="FL Convergence"):
    """Plot global accuracy and loss over rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette("colorblind")
    for ax, key, label in zip(axes, ["val_loss","val_acc"], ["Loss","Accuracy"]):
        vals = history.get(key, [])
        if vals:
            ax.plot(range(1, len(vals)+1), vals, color=colors[0], linewidth=2)
            ax.set_title(f"Global {label}"); ax.set_xlabel("Round")
            ax.grid(alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot → {save_path}")

def plot_per_client_performance(client_metrics: dict, save_path: str):
    """Bar chart of per-client accuracy."""
    clients = list(client_metrics.keys())
    accs    = [client_metrics[c].get("val_acc", 0) for c in clients]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("colorblind", len(clients))
    ax.bar([f"Hospital {c+1}" for c in clients], accs, color=colors, edgecolor="white")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Per-Client Performance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_fl_results(results: dict, config: dict, save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = {"config": config, "results": results}
    p = Path(save_dir) / "fl_results.json"
    with open(p, "w") as f: json.dump(out, f, indent=2, default=str)
    print(f"Results saved → {p}")
