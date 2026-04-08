"""Flower server setup with configurable aggregation strategies."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.server import ServerConfig, ServerApp
from flwr.server.strategy import FedAvg
from src.fl.strategies import get_strategy, weighted_average
from src.fl.client import get_weights, set_weights
from src.models.resnet import get_model
from src.models.train import get_device, evaluate as eval_model

def make_evaluate_fn(global_test_loader, model_name="resnet50", num_classes=4):
    """Returns a server-side evaluation function run after each aggregation round."""
    device = get_device()
    model  = get_model(model_name, num_classes, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()

    def evaluate_fn(server_round: int, parameters, config):
        set_weights(model, parameters)
        loss, acc, preds, labels, probs = eval_model(model, global_test_loader, criterion, device)
        print(f"  [Server Round {server_round}] Global test → loss={loss:.4f}, acc={acc:.4f}")
        return loss, {"val_acc": acc, "round": server_round}

    return evaluate_fn

def create_server_app(fl_config: Dict, global_test_loader=None):
    """Build a Flower ServerApp with the configured strategy."""
    evaluate_fn = None
    if global_test_loader is not None:
        evaluate_fn = make_evaluate_fn(
            global_test_loader,
            model_name=fl_config["model"]["name"],
            num_classes=fl_config["model"]["num_classes"],
        )

    def on_fit_config_fn(server_round: int) -> Dict:
        return {
            "local_epochs": fl_config["client"]["local_epochs"],
            "learning_rate": fl_config["client"]["learning_rate"],
            "round": server_round,
        }

    strategy = get_strategy(fl_config, evaluate_fn=evaluate_fn)
    strategy.on_fit_config_fn = on_fit_config_fn

    server_config = ServerConfig(num_rounds=fl_config["federation"]["num_rounds"])
    return ServerApp(strategy=strategy, config=server_config)
