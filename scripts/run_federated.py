"""Federated learning runner — compatible with flwr >= 1.20.
Usage:
  python scripts/run_federated.py                              (FedAvg, alpha=0.5)
  python scripts/run_federated.py --strategy fedprox --alpha 0.1 --num_clients 3
"""
import os, sys

# ── Suppress noisy TF/CUDA/protobuf/Ray warnings before any imports ──────────
os.environ.setdefault("RAY_memory_usage_threshold",       "0.98")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",             "3")      # hide cuFFT/cuDNN errors
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",            "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")  # MessageFactory fix
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")   # hide Ray FutureWarning
os.environ.setdefault("RAY_DEDUP_LOGS",                   "1")
os.environ.setdefault("FLWR_TELEMETRY_ENABLED",           "0")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
warnings.filterwarnings("ignore", category=FutureWarning,      module="ray")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse, json, yaml
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="FedTumorNet — Federated simulation")
    parser.add_argument("--config",      default="configs/fl_config.yaml")
    parser.add_argument("--strategy",    default="fedavg", choices=["fedavg","fedprox"])
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--num_clients", type=int,   default=3)
    parser.add_argument("--num_rounds",  type=int,   default=10)   # start small
    parser.add_argument("--num_cpus",    type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override from CLI
    config["strategy"]["name"]          = args.strategy
    config["data"]["dirichlet_alpha"]   = args.alpha
    config["federation"]["num_clients"] = args.num_clients
    config["federation"]["num_rounds"]  = args.num_rounds

    # Write updated config so client_fn reads the CLI overrides
    import tempfile, atexit
    tmp_cfg = Path("configs/_fl_config_run.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(config, f)
    atexit.register(lambda: tmp_cfg.unlink(missing_ok=True))
    # Point client_fn at the temp config
    os.environ["FL_CONFIG"] = str(tmp_cfg)

    save_dir = Path("outputs/fl_experiments") / f"{args.strategy}_alpha{args.alpha}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FedTumorNet — Federated Learning [{args.strategy.upper()}]")
    print("=" * 60)
    print(f"Clients: {args.num_clients} | Rounds: {args.num_rounds} | α={args.alpha}")

    # ── Build data for server-side global test ────────────────────────────────
    # Only load the global test set here; client data is created lazily inside
    # each Ray actor to avoid serialising ALL partitions into EVERY worker.
    from src.data.dataset import create_federated_datasets
    _, global_test = create_federated_datasets(
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=config["data"]["seed"],
    )
    global_test_loader = DataLoader(global_test, batch_size=32,
                                    shuffle=False, num_workers=0,
                                    pin_memory=False)

    # ── Build ServerApp ───────────────────────────────────────────────────────
    from src.fl.server import create_server_app
    server_app = create_server_app(config, global_test_loader)

    # ── Build ClientApp ───────────────────────────────────────────────────────
    # IMPORTANT: The client_fn closure must NOT capture large objects (datasets,
    # dataloaders).  Ray pickles the closure to each ClientAppActor; capturing
    # all dataloaders tripled total memory and caused OOM.  Instead, we pass
    # only the lightweight `config` dict and let each actor create its own
    # partition on the fly.
    from src.fl.client import BrainTumorClient
    from src.models.resnet import get_model
    from src.data.dataset import create_federated_datasets as _create_fed, get_dataloaders

    # Snapshot only the tiny config dict — no heavy data
    _cfg = config  # ~1 KB serialised

    # Actor-level dataset cache — keyed by (num_clients, alpha, seed, cid).
    # Flower calls client_fn once per selected client per round; without caching
    # create_federated_datasets() would run 3 × num_rounds = 30 times.
    _dataset_cache: dict = {}

    def _client_fn(context):
        cid = int(context.node_config.get("partition-id", 0))
        cache_key = (_cfg["federation"]["num_clients"],
                     _cfg["data"]["dirichlet_alpha"],
                     _cfg["data"]["seed"], cid)

        if cache_key not in _dataset_cache:
            # First call for this (cid, config): build partition silently
            client_datasets, _ = _create_fed(
                num_clients=_cfg["federation"]["num_clients"],
                alpha=_cfg["data"]["dirichlet_alpha"],
                seed=_cfg["data"]["seed"],
                verbose=False,   # suppress per-actor print/save
            )
            _dataset_cache[cache_key] = get_dataloaders(
                {cid: client_datasets[cid]},
                batch_size=_cfg["client"]["batch_size"],
                num_workers=0,
            )

        loaders = _dataset_cache[cache_key]
        model = get_model(_cfg["model"]["name"], _cfg["model"]["num_classes"],
                          pretrained=_cfg["model"]["pretrained"])
        client = BrainTumorClient(
            model=model,
            train_loader=loaders[cid]["train"],
            val_loader=loaders[cid]["val"],
            config=_cfg["client"],
            strategy=_cfg["strategy"]["name"],
            mu=_cfg["strategy"].get("fedprox_mu", 0.01),
        )
        return client.to_client()

    import flwr as fl
    client_app = fl.client.ClientApp(client_fn=_client_fn)

    # ── Run simulation ────────────────────────────────────────────────────────
    backend_cfg = {"client_resources": {"num_cpus": args.num_cpus, "num_gpus": 0.0}}
    print(f"\nStarting simulation with Ray backend…\n")

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config=backend_cfg,
    )

    # Save run metadata
    meta = {"strategy": args.strategy, "alpha": args.alpha,
            "num_clients": args.num_clients, "num_rounds": args.num_rounds}
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ FL simulation complete! Results → {save_dir}/")


if __name__ == "__main__":
    main()
