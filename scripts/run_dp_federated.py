"""DP-FL runner with epsilon sweep.
Usage:
  python scripts/run_dp_federated.py --epsilon 1.0
  python scripts/run_dp_federated.py --sweep
"""

import os, sys
from pathlib import Path

# ── Suppress noisy TF/CUDA/protobuf/Ray warnings before any imports ──────────
os.environ.setdefault("RAY_memory_usage_threshold",       "0.98")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",             "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",            "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS",                   "1")
os.environ.setdefault("FLWR_TELEMETRY_ENABLED",           "0")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
warnings.filterwarnings("ignore", category=FutureWarning,      module="ray")

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

    # Only load global test set in main process; client data is created lazily
    # inside each Ray actor to avoid serialising ALL partitions into EVERY worker.
    _, global_test = create_federated_datasets(
        num_clients=num_clients, alpha=alpha, seed=config["data"]["seed"]
    )
    global_loader = DataLoader(global_test, batch_size=32, shuffle=False,
                               num_workers=0, pin_memory=False)

    # Estimate sample rate for noise multiplier from a quick partition
    _tmp_datasets, _ = create_federated_datasets(
        num_clients=num_clients, alpha=alpha, seed=config["data"]["seed"]
    )
    sample_rate = config["client"]["batch_size"] / max(
        len(_tmp_datasets[0]["train"]), 1)
    del _tmp_datasets  # free immediately

    noise_mult = compute_noise_multiplier(
        epsilon, dp_config["differential_privacy"]["delta"],
        config["client"]["local_epochs"], sample_rate
    )
    print(f"  ε={epsilon:.1f} → noise_multiplier={noise_mult:.4f}")

    # Snapshot lightweight config — do NOT capture heavy data in the closure
    _cfg = config
    _dp_cfg = dp_config
    _eps = epsilon
    _nm = noise_mult

    def dp_client_fn_inner(context):
        cid = int(context.node_config.get("partition-id", 0))

        # Each actor creates ONLY its own shard
        client_datasets, _ = create_federated_datasets(
            num_clients=_cfg["federation"]["num_clients"],
            alpha=_cfg["data"]["dirichlet_alpha"],
            seed=_cfg["data"]["seed"],
        )
        loaders = get_dataloaders(
            {cid: client_datasets[cid]},
            batch_size=_cfg["client"]["batch_size"],
            num_workers=0,
        )

        model = get_model(_cfg["model"]["name"], _cfg["model"]["num_classes"])
        client = DPBrainTumorClient(
            model, loaders[cid]["train"], loaders[cid]["val"],
            config=_cfg["client"],
            target_epsilon=_eps,
            delta=_dp_cfg["differential_privacy"]["delta"],
            max_grad_norm=_dp_cfg["differential_privacy"]["max_grad_norm"],
            noise_multiplier=_nm,
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
