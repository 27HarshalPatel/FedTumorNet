---
phase: 4
plan: 01
title: "Differential Privacy Integration"
wave: 1
depends_on: ["phase-3"]
files_modified:
  - src/privacy/dp_client.py
  - src/privacy/accountant.py
  - src/privacy/utils.py
  - configs/dp_config.yaml
  - scripts/run_dp_federated.py
  - tests/test_dp.py
requirements_addressed: [REQ-004]
autonomous: true
---

# Phase 4: Differential Privacy Integration

<objective>
Integrate Opacus-based differential privacy into the federated learning pipeline. Implement per-sample gradient clipping and noise injection with configurable privacy budgets (ε). Evaluate the accuracy-privacy tradeoff across multiple ε values and produce key paper figures.
</objective>

## Tasks

<task id="4.1" title="DP Configuration">
<read_first>
- configs/fl_config.yaml
</read_first>

<action>
Create `configs/dp_config.yaml`:
```yaml
differential_privacy:
  enabled: true
  epsilon_values: [0.5, 1.0, 2.0, 5.0]  # privacy budgets to sweep
  delta: 1e-5                    # δ parameter (set to 1/N)
  max_grad_norm: 1.0             # per-sample gradient clipping bound
  noise_multiplier: null         # auto-computed from ε, δ, epochs
  mechanism: "gaussian"          # gaussian or laplace
  accountant: "rdp"              # RDP accountant for tight composition

# Combined with FL config
federation:
  num_rounds: 50
  local_epochs: 3
  strategy: "fedavg"
```
</action>

<acceptance_criteria>
- `configs/dp_config.yaml` contains `epsilon_values: [0.5, 1.0, 2.0, 5.0]`
- `configs/dp_config.yaml` contains `max_grad_norm: 1.0`
- `configs/dp_config.yaml` contains `accountant: "rdp"`
</acceptance_criteria>
</task>

<task id="4.2" title="DP-Enabled Flower Client">
<read_first>
- src/fl/client.py
- configs/dp_config.yaml
</read_first>

<action>
Create `src/privacy/dp_client.py`:

1. **`DPBrainTumorClient(BrainTumorClient)`** — extends the base client with DP:
   - In `__init__`: wraps model, optimizer, and dataloader with Opacus:
     ```python
     from opacus import PrivacyEngine
     privacy_engine = PrivacyEngine()
     model, optimizer, dataloader = privacy_engine.make_private(
         module=model,
         optimizer=optimizer,
         data_loader=train_loader,
         noise_multiplier=noise_multiplier,
         max_grad_norm=max_grad_norm,
     )
     ```
   - `fit()`: trains with DP-SGD (Opacus handles gradient clipping + noise injection automatically)
   - After each local training round, queries `privacy_engine.get_epsilon(delta)` to track spent budget
   - Returns additional metrics: `{"epsilon_spent": ..., "delta": ...}`

2. **`compute_noise_multiplier(target_epsilon, delta, num_epochs, sample_rate)`**:
   - Uses Opacus `get_noise_multiplier()` to compute the required noise level for target ε
   - Returns float noise_multiplier

3. **`dp_client_fn(context)`** — factory that creates DPBrainTumorClient
</action>

<acceptance_criteria>
- `src/privacy/dp_client.py` contains `class DPBrainTumorClient`
- `src/privacy/dp_client.py` contains `PrivacyEngine`
- `src/privacy/dp_client.py` contains `make_private`
- `src/privacy/dp_client.py` contains `max_grad_norm`
- `src/privacy/dp_client.py` contains `get_epsilon` or `epsilon`
</acceptance_criteria>
</task>

<task id="4.3" title="Privacy Accountant & Tracking">
<read_first>
- src/privacy/dp_client.py
</read_first>

<action>
Create `src/privacy/accountant.py`:

1. **`PrivacyTracker`** class:
   - Tracks cumulative ε spent per client per round
   - `record(client_id, round_num, epsilon, delta)` — logs privacy budget consumed
   - `get_total_epsilon(client_id)` → total ε for a client after all rounds
   - `check_budget(client_id, max_epsilon)` → bool, whether budget exceeded
   - `save_report(save_path)` → JSON with per-client, per-round ε tracking
   - `plot_privacy_curve(save_path)` → line plot of cumulative ε vs rounds per client

Create `src/privacy/utils.py`:
1. `plot_accuracy_privacy_tradeoff(results_dict, save_path)`:
   - X-axis: ε (privacy budget)
   - Y-axis: Test accuracy
   - Adds horizontal line for non-DP baseline
   - Includes error bars (mean ± std over seeds)
   - Publication-quality matplotlib figure

2. `plot_privacy_budget_over_rounds(tracker, save_path)`:
   - Shows cumulative privacy loss over FL rounds
</action>

<acceptance_criteria>
- `src/privacy/accountant.py` contains `class PrivacyTracker`
- `src/privacy/accountant.py` contains `def record`
- `src/privacy/accountant.py` contains `def get_total_epsilon`
- `src/privacy/utils.py` contains `def plot_accuracy_privacy_tradeoff`
- Plotting functions use `matplotlib`
</acceptance_criteria>
</task>

<task id="4.4" title="DP-FL Runner & Sweep Script">
<read_first>
- src/privacy/dp_client.py
- src/privacy/accountant.py
- scripts/run_federated.py
</read_first>

<action>
Create `scripts/run_dp_federated.py`:

1. CLI: `python scripts/run_dp_federated.py --epsilon 1.0 --strategy fedavg --alpha 0.5`
2. Also supports sweep mode: `python scripts/run_dp_federated.py --sweep` which runs all ε ∈ {0.5, 1.0, 2.0, 5.0, ∞}
3. For each ε value:
   - Computes noise_multiplier
   - Runs FL with DP clients
   - Records final accuracy, AUC, per-client performance, total ε
4. After sweep, generates:
   - **Key Paper Figure**: Accuracy vs ε tradeoff curve
   - Results table saved as CSV and LaTeX
   - Per-ε convergence curves overlay
</action>

<acceptance_criteria>
- `scripts/run_dp_federated.py` accepts `--epsilon` argument
- `scripts/run_dp_federated.py` contains sweep logic for multiple ε values
- Script generates accuracy-privacy tradeoff figure
- Script saves results to `outputs/dp_experiments/`
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. DP-FL pipeline runs end-to-end with ε=2.0 without crashing
2. Accuracy with ε=2.0 is within 5% of non-DP FL baseline
3. Accuracy-privacy tradeoff curve generated covering ε ∈ {0.5, 1.0, 2.0, 5.0}
4. Privacy accountant correctly tracks cumulative ε per round
5. Noise multiplier computed correctly from target ε via Opacus
</must_haves>
