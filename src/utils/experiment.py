"""Experiment runner utilities for ablation studies."""
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats

class ExperimentRunner:
    """Manages ablation experiment sweeps across seeds."""

    def __init__(self, save_dir="outputs/ablations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_results(self, raw_results: List[Dict]) -> Dict:
        """Compute mean ± std across seeds for each metric."""
        if not raw_results: return {}
        keys = [k for k in raw_results[0].keys() if isinstance(raw_results[0][k], (int, float))]
        agg = {}
        for k in keys:
            vals = [r[k] for r in raw_results if k in r]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return agg

    def statistical_test(self, results_a: List[float], results_b: List[float]) -> Dict:
        """Paired t-test between two result sets."""
        if len(results_a) != len(results_b) or len(results_a) < 2:
            return {"p_value": None, "significant": None, "note": "insufficient data"}
        t_stat, p_val = stats.ttest_rel(results_a, results_b)
        return {"t_statistic": float(t_stat), "p_value": float(p_val),
                "significant": p_val < 0.05}

    def to_latex_table(self, rows: List[Dict], caption="Results", label="tab:results") -> str:
        """Generate LaTeX table from list of result dicts."""
        if not rows: return ""
        cols = list(rows[0].keys())
        header = " & ".join(f"\\textbf{{{c}}}" for c in cols)
        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "  \\begin{tabular}{" + "l" * len(cols) + "}",
            "    \\hline",
            f"    {header} \\\\",
            "    \\hline",
        ]
        for row in rows:
            cells = []
            for k in cols:
                v = row[k]
                if isinstance(v, dict) and "mean" in v:
                    cells.append(f"{v['mean']:.3f} $\\pm$ {v['std']:.3f}")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            lines.append("    " + " & ".join(cells) + " \\\\")
        lines += ["    \\hline", "  \\end{tabular}", "\\end{table}"]
        return "\n".join(lines)

    def save_results(self, results: List[Dict], ablation_name: str):
        """Save results to CSV, JSON, and LaTeX."""
        out_dir = self.save_dir / ablation_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # CSV
        if results:
            keys = list(results[0].keys())
            with open(out_dir / "results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(results)

        # LaTeX
        latex = self.to_latex_table(results, caption=f"Ablation: {ablation_name}",
                                     label=f"tab:{ablation_name}")
        with open(out_dir / "table.tex", "w") as f: f.write(latex)

        print(f"Results saved → {out_dir}/")
