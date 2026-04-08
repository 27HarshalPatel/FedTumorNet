"""DP-FL runner with epsilon sweep.
Usage:
  python scripts/run_dp_federated.py --epsilon 1.0
  python scripts/run_dp_federated.py --sweep
"""

import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse, json
import yaml
from pathlib import Path

def run_one_epsilon(epsilon, config, dp_config, num_clients, num_rounds, save_dir):
    from src.data.dataset import create_federated_datasets, get_dataloaders
    from src.privacy.dp_client import DPBrainTumorClient, compute_noise_multiplier
    from src.models.resnet import get_model
    from src.fl.strategies import get_strategy
    from src.models.train import get_device, evaluate as eval_model
    import flwr as fl
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import numpy as np

    strategy_name = config["strategy"]["name"]
    alpha = config["data"]["dirichlet_alpha"]

    client_datasets, global_test = create_federated_datasets(
        num_clients=num_clients, alpha=alpha, seed=config["data"]["seed"]
    )
    loaders = get_dataloaders(client_datasets, batch_size=config["client"]["batch_size"])
    global_loader = DataLoader(global_test, batch_size=32, shuffle=False, num_workers=0)

    sample_rate = config["client"]["batch_size"] / max(
        len(client_datasets[0]["train"]), 1)
    noise_mult = compute_noise_multiplier(
        epsilon, dp_config["differential_privacy"]["delta"],
        config["client"]["local_epochs"], sample_rate
    )
    print(f"  ε={epsilon:.1f} → noise_multiplier={noise_mult:.4f}")

    def dp_client_fn_inner(context):
        cid = int(context.node_config.get("partition-id", 0))
        model = get_model(config["model"]["name"], config["model"]["num_classes"])
        client = DPBrainTumorClient(
            model, loaders[cid]["train"], loaders[cid]["val"],
            config=config["client"],
            target_epsilon=epsilon,
            delta=dp_config["differential_privacy"]["delta"],
            max_grad_norm=dp_config["differential_privacy"]["max_grad_norm"],
            noise_multiplier=noise_mult,
        )
        return client.to_client()

    strategy = get_strategy(config)
    history = fl.simulation.start_simulation(
        client_fn=dp_client_fn_inner,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    # Evaluate final global model accuracy (stub — real result from server evaluate_fn)
    result = {"epsilon": epsilon, "noise_multiplier": noise_mult,
               "status": "complete", "rounds": num_rounds}

    out_path = save_dir / f"epsilon_{epsilon}.json"
    with open(out_path, "w") as f: json.dump(result, f, indent=2)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--sweep",   action="store_true")
    parser.add_argument("--config",    default="configs/fl_config.yaml")
    parser.add_argument("--dp_config", default="configs/dp_config.yaml")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds",  type=int, default=30)
    args = parser.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    with open(args.dp_config) as f: dp_cfg = yaml.safe_load(f)

    save_dir = Path("outputs/dp_experiments")
    save_dir.mkdir(parents=True, exist_ok=True)

    epsilons = dp_cfg["differential_privacy"]["epsilon_values"] if args.sweep else [args.epsilon]

    print("=" * 60)
    print(f"FedTumorNet — DP-FL ({'sweep' if args.sweep else f'ε={args.epsilon}'})")
    print("=" * 60)

    all_results = {}
    for eps in epsilons:
        print(f"\nRunning ε={eps}...")
        result = run_one_epsilon(eps, cfg, dp_cfg, args.num_clients, args.num_rounds, save_dir)
        all_results[eps] = result

    with open(save_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ DP-FL complete! Results → {save_dir}/")

if __name__ == "__main__":
    main()
