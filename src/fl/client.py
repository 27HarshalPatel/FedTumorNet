"""Flower FL Client — compatible with flwr >= 1.20 (ClientApp API)."""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from src.models.resnet import get_model
from src.models.train import get_device, evaluate as eval_model
from src.utils.metrics import compute_accuracy


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, params: List[np.ndarray]):
    state = model.state_dict()
    for key, val in zip(state.keys(), params):
        state[key] = torch.tensor(val)
    model.load_state_dict(state, strict=True)


def fedprox_loss(model: nn.Module, global_params: List[np.ndarray], mu: float) -> torch.Tensor:
    """Proximal term for FedProx: mu/2 * ||w - w_global||^2"""
    proximal = 0.0
    for local_p, g in zip(model.parameters(), global_params):
        proximal += ((local_p - torch.tensor(g).to(local_p.device)) ** 2).sum()
    return (mu / 2) * proximal


class BrainTumorClient(NumPyClient):
    """Flower NumPy client wrapping local training on one hospital's shard."""

    def __init__(self, model, train_loader, val_loader, config: Dict,
                 strategy: str = "fedavg", mu: float = 0.01):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.strategy     = strategy
        self.mu           = mu
        self.device       = get_device()
        self.model.to(self.device)
        self.criterion    = nn.CrossEntropyLoss()

    def get_parameters(self, config) -> List[np.ndarray]:
        return get_weights(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        set_weights(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs",
                           self.config.get("local_epochs", 3)))
        lr = float(config.get("learning_rate",
                   self.config.get("learning_rate", 0.001)))

        optimizer = SGD(self.model.parameters(), lr=lr,
                        momentum=self.config.get("momentum", 0.9),
                        weight_decay=self.config.get("weight_decay", 1e-4))

        global_params = ([p.copy() for p in parameters]
                         if self.strategy == "fedprox" else None)

        self.model.train()
        total_loss = total_correct = total_samples = 0

        for _ in range(local_epochs):
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                if self.strategy == "fedprox" and global_params is not None:
                    loss += fedprox_loss(self.model, global_params, self.mu)
                loss.backward()
                optimizer.step()
                total_loss    += loss.item() * len(labels)
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += len(labels)

        metrics = {
            "train_loss": total_loss  / max(total_samples, 1),
            "train_acc":  total_correct / max(total_samples, 1),
        }
        return self.get_parameters({}), total_samples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        loss, acc, _, _, _ = eval_model(
            self.model, self.val_loader, self.criterion, self.device
        )
        return loss, len(self.val_loader.dataset), {"val_acc": acc}


# ── ClientApp factory (flwr >= 1.20) ────────────────────────────────────────

def client_fn(context: Context) -> fl.client.Client:
    """Called once per client node per round by the Flower runtime."""
    import yaml
    from src.data.dataset import create_federated_datasets, get_dataloaders

    with open("configs/fl_config.yaml") as f:
        cfg = yaml.safe_load(f)

    partition_id = int(context.node_config.get("partition-id", 0))
    num_clients  = cfg["federation"]["num_clients"]
    alpha        = cfg["data"]["dirichlet_alpha"]
    strategy     = cfg["strategy"]["name"]
    mu           = cfg["strategy"].get("fedprox_mu", 0.01)

    client_datasets, _ = create_federated_datasets(
        num_clients=num_clients,
        alpha=alpha,
        seed=cfg["data"]["seed"],
    )
    loaders = get_dataloaders(client_datasets,
                              batch_size=cfg["client"]["batch_size"],
                              num_workers=0)   # 0 workers avoids Ray/fork issues

    model = get_model(cfg["model"]["name"], cfg["model"]["num_classes"],
                      pretrained=cfg["model"]["pretrained"])

    client = BrainTumorClient(
        model=model,
        train_loader=loaders[partition_id]["train"],
        val_loader=loaders[partition_id]["val"],
        config=cfg["client"],
        strategy=strategy,
        mu=mu,
    )
    return client.to_client()


# ClientApp wrapping the factory
client_app = ClientApp(client_fn=client_fn)
