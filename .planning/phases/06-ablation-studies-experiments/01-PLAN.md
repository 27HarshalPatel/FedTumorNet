---
phase: 6
plan: 01
title: "Ablation Studies & Experiments"
wave: 1
depends_on: ["phase-3", "phase-4", "phase-5"]
files_modified:
  - scripts/run_ablations.py
  - src/utils/experiment.py
  - configs/ablation_config.yaml
requirements_addressed: [REQ-006]
autonomous: true
---

# Phase 6: Ablation Studies & Experiments

<objective>
Run comprehensive ablation studies varying model architecture, non-IID severity, number of clients, local epochs, and FL strategy. Generate publication-ready comparison tables with statistical significance tests.
</objective>

## Tasks

<task id="6.1" title="Experiment Configuration">
<read_first>
- configs/fl_config.yaml
- configs/dp_config.yaml
</read_first>

<action>
Create `configs/ablation_config.yaml`:
```yaml
ablations:
  seeds: [42, 123, 456]  # 3 seeds for mean ± std
  
  architecture:
    models: ["resnet50", "efficientnet_b0", "vit_small"]
    fixed:
      strategy: "fedavg"
      alpha: 0.5
      num_clients: 3
      local_epochs: 3
      num_rounds: 50

  noniid_severity:
    alpha_values: [0.1, 0.5, 1.0, 10.0]  # 10.0 ≈ IID
    fixed:
      model: "resnet50"
      strategy: "fedavg"
      num_clients: 3

  num_clients:
    client_counts: [2, 3, 5, 8]
    fixed:
      model: "resnet50"
      strategy: "fedavg"
      alpha: 0.5

  local_epochs:
    epoch_values: [1, 3, 5, 10]
    fixed:
      model: "resnet50"
      strategy: "fedavg"
      alpha: 0.5
      num_clients: 3

  strategy_comparison:
    strategies: ["fedavg", "fedprox"]
    fixed:
      model: "resnet50"
      alpha: 0.5
      num_clients: 3

  privacy_accuracy:
    epsilon_values: [0.5, 1.0, 2.0, 5.0]
    fixed:
      model: "resnet50"
      strategy: "fedavg"
      alpha: 0.5
      num_clients: 3
```

Create `src/utils/experiment.py`:
1. `ExperimentRunner` class:
   - `run_single(config, seed)` → runs one FL experiment, returns metrics dict
   - `run_ablation(ablation_name, config)` → sweeps parameter, runs over all seeds
   - `aggregate_results(raw_results)` → computes mean ± std
   - `statistical_test(results_a, results_b)` → paired t-test, returns p-value
   - `to_latex_table(results_df, caption, label)` → LaTeX table string
   - `save_results(results, save_dir)` → CSV + JSON + LaTeX
</action>

<acceptance_criteria>
- `configs/ablation_config.yaml` contains `seeds: [42, 123, 456]`
- `configs/ablation_config.yaml` defines at least 5 ablation categories
- `src/utils/experiment.py` contains `class ExperimentRunner`
- `src/utils/experiment.py` contains `def statistical_test`
- `src/utils/experiment.py` contains `def to_latex_table`
</acceptance_criteria>
</task>

<task id="6.2" title="Ablation Runner Script">
<read_first>
- src/utils/experiment.py
- configs/ablation_config.yaml
</read_first>

<action>
Create `scripts/run_ablations.py`:

1. CLI: `python scripts/run_ablations.py --ablation architecture` or `--all`
2. For each ablation:
   - Runs experiments across parameter sweep × seeds
   - Generates comparison figure (bar chart with error bars)
   - Generates LaTeX table
   - Saves raw results to `outputs/ablations/{ablation_name}/`
3. Key paper figures generated:
   - **Architecture comparison**: Bar chart (ResNet-50 vs EfficientNet vs ViT)
   - **Non-IID severity**: Line plot accuracy vs α
   - **Client scaling**: Accuracy vs number of clients
   - **Strategy comparison**: Grouped bars (FedAvg vs FedProx) with Jain index
   - **Privacy-accuracy tradeoff**: Accuracy vs ε with centralized baseline
   - **Convergence comparison**: Overlay of convergence curves for different strategies

4. After all ablations, generate **summary table**:
   ```
   | Experiment | Best Config | Accuracy | AUC | Fairness | ε |
   |------------|-------------|----------|-----|----------|---|
   ```
</action>

<acceptance_criteria>
- `scripts/run_ablations.py` accepts `--ablation` argument
- `scripts/run_ablations.py` accepts `--all` for full sweep
- Script saves figures to `outputs/ablations/`
- Script generates LaTeX tables
- At least 5 ablation experiments defined
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. All ablation experiments complete over 3 seeds
2. Results tables include mean ± std for all metrics
3. Statistical significance tests performed (paired t-test)
4. At least 6 publication-quality figures generated
5. LaTeX-formatted tables saved for paper inclusion
</must_haves>
