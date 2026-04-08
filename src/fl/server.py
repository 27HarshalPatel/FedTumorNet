"""Flower server setup — compatible with flwr >= 1.20 (ServerApp API)."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.server import ServerConfig, ServerApp, ServerAppComponents
from flwr.common import Context
from src.fl.strategies import get_strategy, weighted_average
from src.fl.client import get_weights, set_weights
from src.models.resnet import get_model
from src.models.train import get_device, evaluate as eval_model


def make_evaluate_fn(global_test_loader, config: Dict):
    """Server-side evaluation called after each aggregation round."""
    device = get_device()
    model  = get_model(config["model"]["name"], config["model"]["num_classes"],
                       pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()

    def evaluate_fn(server_round: int, parameters, cfg):
        set_weights(model, parameters)
        loss, acc, _, _, _ = eval_model(model, global_test_loader, criterion, device)
        print(f"  [Round {server_round}] Global test → loss={loss:.4f}, acc={acc:.4f}")
        return loss, {"val_acc": acc, "round": server_round}

    return evaluate_fn


def server_fn(context: Context) -> ServerAppComponents:
    """Factory called once by the Flower runtime to build server components."""
    import yaml
    from src.data.dataset import create_federated_datasets
    from torch.utils.data import DataLoader

    with open("configs/fl_config.yaml") as f:
        fl_config = yaml.safe_load(f)

    num_rounds = int(context.run_config.get(
        "num-rounds", fl_config["federation"]["num_rounds"]
    ))

    # Build global test loader
    client_datasets, global_test = create_federated_datasets(
        num_clients=fl_config["federation"]["num_clients"],
        alpha=fl_config["data"]["dirichlet_alpha"],
        seed=fl_config["data"]["seed"],
    )
    global_test_loader = DataLoader(global_test, batch_size=32,
                                    shuffle=False, num_workers=0)

    evaluate_fn = make_evaluate_fn(global_test_loader, fl_config)

    def on_fit_config_fn(server_round: int) -> Dict:
        return {
            "local_epochs":  fl_config["client"]["local_epochs"],
            "learning_rate": fl_config["client"]["learning_rate"],
            "round":         server_round,
        }

    strategy = get_strategy(fl_config, evaluate_fn=evaluate_fn)
    strategy.on_fit_config_fn = on_fit_config_fn
    config   = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


def create_server_app(fl_config: Dict, global_test_loader=None) -> ServerApp:
    """Build a Flower ServerApp — used by the external runner."""
    evaluate_fn = (make_evaluate_fn(global_test_loader, fl_config)
                   if global_test_loader is not None else None)

    def on_fit_config_fn(server_round: int) -> Dict:
        return {
            "local_epochs":  fl_config["client"]["local_epochs"],
            "learning_rate": fl_config["client"]["learning_rate"],
            "round":         server_round,
        }

    strategy = get_strategy(fl_config, evaluate_fn=evaluate_fn)
    strategy.on_fit_config_fn = on_fit_config_fn
    server_config = ServerConfig(num_rounds=fl_config["federation"]["num_rounds"])

    def _server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=strategy, config=server_config)

    return ServerApp(server_fn=_server_fn)
