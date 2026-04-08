"""Privacy budget tracker for DP-FL experiments."""
import json
from pathlib import Path
from typing import Dict, List
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

class PrivacyTracker:
    """Tracks cumulative epsilon per client per round."""
    def __init__(self):
        self.records: List[Dict] = []

    def record(self, client_id: int, round_num: int, epsilon: float, delta: float):
        self.records.append({"client_id": client_id, "round": round_num,
                              "epsilon": epsilon, "delta": delta})

    def get_total_epsilon(self, client_id: int) -> float:
        eps = [r["epsilon"] for r in self.records if r["client_id"] == client_id]
        return max(eps) if eps else 0.0

    def check_budget(self, client_id: int, max_epsilon: float) -> bool:
        return self.get_total_epsilon(client_id) <= max_epsilon

    def save_report(self, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f: json.dump(self.records, f, indent=2)
        print(f"Privacy report → {save_path}")

    def plot_privacy_curve(self, save_path: str):
        if not self.records: return
        import pandas as pd
        df = pd.DataFrame(self.records)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("colorblind")
        for i, cid in enumerate(df["client_id"].unique()):
            sub = df[df["client_id"] == cid].sort_values("round")
            ax.plot(sub["round"], sub["epsilon"], label=f"Hospital {cid+1}",
                    color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel("FL Round"); ax.set_ylabel("Cumulative ε")
        ax.set_title("Privacy Budget Consumption Over Rounds")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
