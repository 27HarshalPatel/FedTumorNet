"""Federated learning runner.
Usage: python scripts/run_federated.py --strategy fedavg --alpha 0.5 --num_clients 3
"""

import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run FL simulation with Flower")
    parser.add_argument("--config",      default="configs/fl_config.yaml")
    parser.add_argument("--strategy",    default="fedavg", choices=["fedavg","fedprox"])
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--num_clients", type=int,   default=3)
    parser.add_argument("--num_rounds",  type=int,   default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override from CLI
    config["strategy"]["name"]          = args.strategy
    config["data"]["dirichlet_alpha"]   = args.alpha
    config["federation"]["num_clients"] = args.num_clients
    config["federation"]["num_rounds"]  = args.num_rounds

    save_dir = Path(config["logging"]["save_dir"]) / f"{args.strategy}_alpha{args.alpha}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FedTumorNet — Federated Learning [{args.strategy.upper()}]")
    print("=" * 60)
    print(f"Clients: {args.num_clients} | Rounds: {args.num_rounds} | α={args.alpha}")

    # Prepare data
    from src.data.dataset import create_federated_datasets, get_dataloaders
    from torch.utils.data import DataLoader

    client_datasets, global_test = create_federated_datasets(
        num_clients=args.num_clients, alpha=args.alpha,
        seed=config["data"]["seed"],
    )
    global_test_loader = DataLoader(global_test, batch_size=32, shuffle=False, num_workers=0)

    # Build server + client apps
    from src.fl.server import create_server_app
    from src.fl.client import client_fn

    server_app = create_server_app(config, global_test_loader)

    # Run simulation
    import flwr as fl
    history = fl.simulation.run_serverapp(
        server_app=server_app,
        num_supernodes=args.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}},
    ) if hasattr(fl.simulation, "run_serverapp") else None

    # Fallback to start_simulation API
    if history is None:
        from src.fl.strategies import get_strategy
        strategy = get_strategy(config)
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )

    # Save results
    results = {
        "strategy": args.strategy,
        "alpha": args.alpha,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "distributed_fit": str(history.metrics_distributed_fit if history else {}),
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    from src.fl.utils import plot_fl_convergence, save_fl_results
    # Plot if metrics available
    if history and history.metrics_distributed_fit:
        rounds_data = {}
        for k, v in history.metrics_distributed_fit.items():
            rounds_data[k] = [x[1] for x in v]
        plot_fl_convergence(rounds_data, str(save_dir / "convergence.png"),
                            title=f"{args.strategy.upper()} Convergence (α={args.alpha})")

    print(f"\n✅ FL simulation complete! Results → {save_dir}/")

if __name__ == "__main__":
    main()
