---
phase: 3
plan: 01
title: "Federated Learning Pipeline"
wave: 1
depends_on: ["phase-2"]
files_modified:
  - src/fl/server.py
  - src/fl/client.py
  - src/fl/strategies.py
  - src/fl/utils.py
  - configs/fl_config.yaml
  - scripts/run_federated.py
  - tests/test_fl_pipeline.py
requirements_addressed: [REQ-003]
autonomous: true
---

# Phase 3: Federated Learning Pipeline

<objective>
Implement the core federated learning pipeline using the Flower framework. Build FL server with configurable aggregation strategies (FedAvg, FedProx, SCAFFOLD), FL client with local training logic, and end-to-end federation simulation across 3-5 simulated hospital sites.
</objective>

## Tasks

<task id="3.1" title="FL Configuration">
<read_first>
- configs/data_config.yaml
- configs/train_config.yaml
</read_first>

<action>
Create `configs/fl_config.yaml`:
```yaml
federation:
  num_clients: 3
  num_rounds: 50
  fraction_fit: 1.0          # fraction of clients selected per round
  fraction_evaluate: 1.0
  min_fit_clients: 3
  min_evaluate_clients: 3
  min_available_clients: 3

client:
  local_epochs: 3
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "sgd"            # SGD preferred in FL for stability
  momentum: 0.9

strategy:
  name: "fedavg"              # fedavg | fedprox | scaffold
  fedprox_mu: 0.01            # proximal term coefficient for FedProx
  scaffold_lr: 1.0            # SCAFFOLD correction learning rate

data:
  partition_method: "dirichlet"
  dirichlet_alpha: 0.5
  seed: 42

model:
  name: "resnet50"
  num_classes: 4
  pretrained: true

logging:
  save_dir: "outputs/fl_experiments"
  log_per_round: true
  save_global_model: true
```
</action>

<acceptance_criteria>
- `configs/fl_config.yaml` contains `strategy: name: "fedavg"`
- `configs/fl_config.yaml` contains `fedprox_mu: 0.01`
- `configs/fl_config.yaml` contains `num_rounds: 50`
- `configs/fl_config.yaml` contains `local_epochs: 3`
</acceptance_criteria>
</task>

<task id="3.2" title="Flower Client Implementation">
<read_first>
- src/models/train.py
- src/models/resnet.py
- src/data/dataset.py
- configs/fl_config.yaml
</read_first>

<action>
Create `src/fl/client.py`:

1. **`BrainTumorClient(fl.client.NumPyClient)`**:
   - `__init__(self, model, train_loader, val_loader, config)` — stores local data and model
   - `get_parameters(config)` → returns model weights as list of numpy arrays
   - `set_parameters(parameters)` → loads numpy arrays into model state_dict
   - `fit(parameters, config)`:
     - Sets parameters from server
     - Trains locally for `config.local_epochs` epochs
     - If strategy == "fedprox": adds proximal term `μ/2 * ||w - w_global||²` to the loss
     - Returns updated parameters, num_samples, metrics dict `{"train_loss": ..., "train_acc": ...}`
   - `evaluate(parameters, config)`:
     - Sets parameters, evaluates on local validation set
     - Returns loss, num_samples, metrics dict `{"val_acc": ..., "val_auc": ...}`

2. **`client_fn(context: Context)`** — Flower client factory function:
   - Reads `context.node_config["partition-id"]` to determine which client's data to load
   - Creates model, loads client-specific data partition
   - Returns `BrainTumorClient.to_client()`

3. **FedProx proximal term implementation**:
   ```python
   def fedprox_loss(model, global_params, mu):
       proximal_term = 0.0
       for local_param, global_param in zip(model.parameters(), global_params):
           proximal_term += ((local_param - global_param) ** 2).sum()
       return mu / 2 * proximal_term
   ```
</action>

<acceptance_criteria>
- `src/fl/client.py` contains `class BrainTumorClient`
- `src/fl/client.py` contains `fl.client.NumPyClient` or `flwr.client.NumPyClient`
- `src/fl/client.py` contains `def get_parameters`
- `src/fl/client.py` contains `def fit`
- `src/fl/client.py` contains `def evaluate`
- `src/fl/client.py` contains `def client_fn`
- `src/fl/client.py` contains `fedprox_loss` or proximal term computation
</acceptance_criteria>
</task>

<task id="3.3" title="FL Server & Aggregation Strategies">
<read_first>
- src/fl/client.py
- configs/fl_config.yaml
</read_first>

<action>
Create `src/fl/strategies.py`:

1. **`get_strategy(config)`** — factory function returning Flower Strategy:
   - `"fedavg"` → `fl.server.strategy.FedAvg(...)` with weighted averaging
   - `"fedprox"` → `fl.server.strategy.FedProx(proximal_mu=config.mu)`
   - Custom evaluation function `evaluate_fn` that tests global model on centralized test set each round

2. **Custom metrics aggregation**:
   ```python
   def weighted_average(metrics):
       # Aggregate per-client metrics weighted by num_samples
       accuracies = [m["val_acc"] * n for n, m in metrics]
       examples = [n for n, m in metrics]
       return {"val_acc": sum(accuracies) / sum(examples)}
   ```

Create `src/fl/server.py`:

1. **`create_server_app(config)`**:
   - Creates `ServerApp` with configured strategy
   - Sets `num_rounds` from config
   - Configures on_fit_config_fn to pass `local_epochs`, `learning_rate` to clients
   - Global model evaluation on centralized test set after each round

2. **Per-round logging**:
   - After each round, log: round number, aggregated loss, aggregated accuracy, per-client metrics
   - Save to `outputs/fl_experiments/{strategy}_{alpha}/round_metrics.json`
</action>

<acceptance_criteria>
- `src/fl/strategies.py` contains `def get_strategy`
- `src/fl/strategies.py` contains `FedAvg`
- `src/fl/strategies.py` contains `FedProx` or `proximal_mu`
- `src/fl/strategies.py` contains `def weighted_average`
- `src/fl/server.py` contains `def create_server_app` or `ServerApp`
- `src/fl/server.py` contains `num_rounds`
</acceptance_criteria>
</task>

<task id="3.4" title="FL Runner Script">
<read_first>
- src/fl/server.py
- src/fl/client.py
- src/data/partition.py
</read_first>

<action>
Create `scripts/run_federated.py`:

1. CLI: `python scripts/run_federated.py --config configs/fl_config.yaml --strategy fedavg --alpha 0.5 --num_clients 3`
2. Flow:
   - Loads config, creates non-IID partitioned data
   - Initializes Flower simulation: `fl.simulation.start_simulation()`
   - Runs for configured number of rounds
   - After completion:
     - Saves global model checkpoint
     - Plots convergence curve (global accuracy vs. round)
     - Plots per-client accuracy over rounds
     - Saves all metrics to JSON

3. Also create `src/fl/utils.py`:
   - `plot_fl_convergence(metrics_history, save_path)` — accuracy & loss vs rounds
   - `plot_per_client_performance(client_metrics, save_path)` — bar chart per client
   - `save_fl_results(results, config, save_dir)` — comprehensive results dump
</action>

<acceptance_criteria>
- `scripts/run_federated.py` contains `fl.simulation.start_simulation` or equivalent
- `scripts/run_federated.py` accepts `--strategy` argument
- `scripts/run_federated.py` accepts `--alpha` argument
- `src/fl/utils.py` contains `def plot_fl_convergence`
- Running `python scripts/run_federated.py --strategy fedavg` completes without error
</acceptance_criteria>
</task>

## Verification

<must_haves>
1. FedAvg with 3 clients converges within 50 rounds
2. FL accuracy reaches ≥85% of centralized baseline
3. Per-round metrics logged to JSON
4. Convergence curves saved as PNG
5. Both FedAvg and FedProx strategies work end-to-end
</must_haves>
