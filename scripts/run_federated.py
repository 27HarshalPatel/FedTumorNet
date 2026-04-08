"""Federated learning runner — compatible with flwr >= 1.20.
Usage:
  python scripts/run_federated.py                              (FedAvg, alpha=0.5)
  python scripts/run_federated.py --strategy fedprox --alpha 0.1 --num_clients 3
"""
import os, sys
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
    from src.data.dataset import create_federated_datasets
    client_datasets, global_test = create_federated_datasets(
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=config["data"]["seed"],
    )
    global_test_loader = DataLoader(global_test, batch_size=32,
                                    shuffle=False, num_workers=0)

    # ── Build ServerApp ───────────────────────────────────────────────────────
    from src.fl.server import create_server_app
    server_app = create_server_app(config, global_test_loader)

    # ── Build ClientApp ───────────────────────────────────────────────────────
    from src.fl.client import BrainTumorClient, get_weights, set_weights
    from src.fl.strategies import get_strategy
    from src.models.resnet import get_model
    from src.data.dataset import get_dataloaders

    loaders = get_dataloaders(client_datasets, batch_size=config["client"]["batch_size"],
                              num_workers=0)

    def _client_fn(context):
        cid = int(context.node_config.get("partition-id", 0))
        model = get_model(config["model"]["name"], config["model"]["num_classes"],
                          pretrained=config["model"]["pretrained"])
        client = BrainTumorClient(
            model=model,
            train_loader=loaders[cid]["train"],
            val_loader=loaders[cid]["val"],
            config=config["client"],
            strategy=config["strategy"]["name"],
            mu=config["strategy"].get("fedprox_mu", 0.01),
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
