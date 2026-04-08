"""Exploratory Data Analysis for Brain Tumor MRI Dataset.
Run: python notebooks/01_eda.py
Outputs: outputs/figures/eda/*.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from PIL import Image

from src.data.dataset import load_image_paths_and_labels, CLASS_NAMES, CLASS_MAP
from src.data.partition import dirichlet_partition, visualize_partition, print_partition_stats
from src.data.preprocessing import get_eval_transforms

OUT_DIR = Path("outputs/figures/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")
palette = sns.color_palette("colorblind", 4)

DATA_DIR = "data/raw/Training"

def eda_class_distribution(paths, labels):
    counts = Counter(labels)
    total = len(labels)
    print("\n📊 Class Distribution:")
    print("-" * 40)
    for c, name in enumerate(CLASS_NAMES):
        pct = counts[c] / total * 100
        print(f"  {name:15s} {counts[c]:5d} ({pct:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(CLASS_NAMES, [counts[c] for c in range(4)], color=palette, edgecolor="white")
    ax1.set_title("Sample Count per Class"); ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=15)

    ax2.pie([counts[c] for c in range(4)], labels=CLASS_NAMES, colors=palette,
            autopct="%1.1f%%", startangle=140)
    ax2.set_title("Class Distribution")
    plt.suptitle("Brain Tumor MRI — Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = str(OUT_DIR / "class_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

def eda_sample_grid(paths, labels):
    num_per_class = 4
    fig, axes = plt.subplots(4, num_per_class, figsize=(num_per_class * 3, 4 * 3))
    for row, cls_name in enumerate(CLASS_NAMES):
        cls_paths = [p for p, l in zip(paths, labels) if l == row][:num_per_class]
        for col, ip in enumerate(cls_paths):
            img = Image.open(ip).convert("RGB").resize((224, 224))
            axes[row, col].imshow(img); axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(cls_name, fontsize=11, fontweight="bold")
    plt.suptitle("Sample Images per Tumor Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = str(OUT_DIR / "sample_grid.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

def eda_partition_comparison(labels):
    alphas = [0.1, 0.5, 1.0]
    print("\n📊 Non-IID Partition Comparison (3 clients):")
    for alpha in alphas:
        partitions = dirichlet_partition(labels, num_clients=3, alpha=alpha, seed=42)
        print(f"\n  α={alpha}:")
        print_partition_stats(partitions, labels, CLASS_NAMES)
        visualize_partition(
            partitions, labels, CLASS_NAMES,
            save_path=str(OUT_DIR / f"partition_alpha_{alpha}.png"),
            title=f"Data Distribution (Dirichlet α={alpha})"
        )

def main():
    print("="*60)
    print("FedTumorNet — Exploratory Data Analysis")
    print("="*60)

    if not Path(DATA_DIR).exists():
        print(f"\n⚠ Data not found at {DATA_DIR}")
        print("  Run first: python -m src.data.download")
        return

    paths, labels = load_image_paths_and_labels(DATA_DIR)
    print(f"\nTotal images: {len(paths)}")

    eda_class_distribution(paths, labels)
    print("\n📊 Sample grid:"); eda_sample_grid(paths, labels)
    eda_partition_comparison(labels)

    print(f"\n✅ EDA complete! Figures → {OUT_DIR}/")

if __name__ == "__main__":
    main()
