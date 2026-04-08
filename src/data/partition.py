"""Non-IID partitioning strategies for Federated Learning simulation."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

def dirichlet_partition(labels, num_clients, alpha, seed=42, min_samples=10):
    """Dirichlet non-IID: lower alpha = more heterogeneous."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        props = (np.cumsum(props) * len(idx)).astype(int)
        splits = np.split(idx, props[:-1])
        for cid in range(num_clients):
            if cid < len(splits): client_indices[cid].extend(splits[cid].tolist())
    # Ensure minimum samples per client
    for i in range(num_clients):
        if len(client_indices[i]) < min_samples:
            largest = max(range(num_clients), key=lambda x: len(client_indices[x]))
            needed = min_samples - len(client_indices[i])
            transfer = client_indices[largest][-needed:]
            client_indices[largest] = client_indices[largest][:-needed]
            client_indices[i].extend(transfer)
    return [list(x) for x in client_indices]

def pathological_partition(labels, num_clients, classes_per_client=2, seed=42):
    """Each client gets only a fixed number of classes."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    class_assignments = []
    for _ in range(num_clients):
        sel = rng.choice(num_classes, size=min(classes_per_client, num_classes), replace=False)
        class_assignments.append(sel.tolist())
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = np.where(labels == c)[0].tolist(); rng.shuffle(idx)
        assigned = [i for i, cs in enumerate(class_assignments) if c in cs]
        if not assigned: assigned = [int(rng.integers(0, num_clients))]
        splits = np.array_split(idx, len(assigned))
        for cid, sp in zip(assigned, splits): client_indices[cid].extend(sp.tolist())
    return client_indices

def iid_partition(labels, num_clients, seed=42):
    """Uniform IID split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels)); rng.shuffle(idx)
    return [sp.tolist() for sp in np.array_split(idx, num_clients)]

def visualize_partition(partition_indices, labels, class_names, save_path=None,
                        title="Data Distribution Across Clients"):
    labels = np.array(labels)
    nc = len(class_names); n = len(partition_indices)
    dist = np.zeros((n, nc))
    for i, idx in enumerate(partition_indices):
        cl = labels[idx]
        for c in range(nc): dist[i, c] = np.sum(cl == c)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("colorblind", nc)
    bottom = np.zeros(n)
    for c in range(nc):
        ax.bar(range(n), dist[:, c], 0.6, bottom=bottom, label=class_names[c],
               color=colors[c], edgecolor="white", linewidth=0.5)
        bottom += dist[:, c]
    ax.set_xlabel("Client (Simulated Hospital)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"Hospital {i+1}" for i in range(n)])
    ax.legend(title="Tumor Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)

def print_partition_stats(partition_indices, labels, class_names):
    labels = np.array(labels)
    print("\n" + "="*70 + "\nPARTITION STATISTICS\n" + "="*70)
    for i, idx in enumerate(partition_indices):
        cl = labels[idx]; counts = Counter(cl.tolist()); total = len(idx)
        print(f"\nClient {i} (Hospital {i+1}): {total} samples\n" + "-"*50)
        for c, name in enumerate(class_names):
            cnt = counts.get(c, 0); pct = cnt/total*100 if total else 0
            print(f"  {name:15s} {cnt:5d} ({pct:5.1f}%) " + "█"*int(pct/2))
