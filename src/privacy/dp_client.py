"""DP-enabled Flower client using Opacus for per-sample gradient clipping."""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import flwr as fl
from flwr.common import Context
from src.fl.client import BrainTumorClient, get_weights, set_weights
from src.models.resnet import get_model
from src.models.train import get_device, evaluate as eval_model

def compute_noise_multiplier(target_epsilon: float, delta: float,
                              num_epochs: int, sample_rate: float) -> float:
    """Compute Opacus noise multiplier for target (epsilon, delta)."""
    try:
        from opacus.accountants.utils import get_noise_multiplier
        return float(get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=num_epochs,
            accountant="rdp",
        ))
    except Exception as e:
        print(f"Noise multiplier computation failed: {e}. Using default 1.0")
        return 1.0

class DPBrainTumorClient(BrainTumorClient):
    """FL client with Opacus differential privacy (DP-SGD)."""

    def __init__(self, model, train_loader, val_loader, config: Dict,
                 target_epsilon: float = 2.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0, noise_multiplier: float = None):
        super().__init__(model, train_loader, val_loader, config)
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = None
        self._epsilon_spent = 0.0

        # Compute noise multiplier if not provided
        sample_rate = 1.0 / max(len(train_loader), 1)
        local_epochs = config.get("local_epochs", 3)
        self.noise_multiplier = noise_multiplier or compute_noise_multiplier(
            target_epsilon, delta, local_epochs, sample_rate
        )

    def fit(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", self.config.get("local_epochs", 3)))
        lr = float(config.get("learning_rate", self.config.get("learning_rate", 0.001)))

        optimizer = SGD(self.model.parameters(), lr=lr,
                        momentum=self.config.get("momentum", 0.9),
                        weight_decay=self.config.get("weight_decay", 1e-4))

        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine(accountant="rdp")
            dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            self.privacy_engine = privacy_engine
        except Exception as e:
            print(f"Opacus init failed: {e}. Falling back to non-DP training.")
            dp_model, dp_optimizer, dp_loader = self.model, optimizer, self.train_loader

        dp_model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for _ in range(local_epochs):
            for imgs, labels in dp_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                dp_optimizer.zero_grad()
                logits = dp_model(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                dp_optimizer.step()
                total_loss += loss.item() * len(labels)
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += len(labels)

        # Track privacy budget
        if self.privacy_engine is not None:
            try:
                self._epsilon_spent = float(
                    self.privacy_engine.get_epsilon(self.delta)
                )
            except Exception:
                self._epsilon_spent = self.target_epsilon

        metrics = {
            "train_loss": total_loss / max(total_samples, 1),
            "train_acc":  total_correct / max(total_samples, 1),
            "epsilon_spent": self._epsilon_spent,
            "delta": self.delta,
        }
        return get_weights(self.model), total_samples, metrics

def dp_client_fn(context: Context) -> fl.client.Client:
    """Factory for DP-enabled Flower clients."""
    from src.data.dataset import create_federated_datasets, get_dataloaders
    import yaml

    with open("configs/fl_config.yaml") as f:  cfg = yaml.safe_load(f)
    with open("configs/dp_config.yaml") as f:  dp  = yaml.safe_load(f)

    partition_id = int(context.node_config.get("partition-id", 0))
    num_clients  = cfg["federation"]["num_clients"]
    alpha        = cfg["data"]["dirichlet_alpha"]

    # Create all partitions (deterministic), then keep only this client's shard
    client_datasets, _ = create_federated_datasets(
        num_clients=num_clients, alpha=alpha, seed=cfg["data"]["seed"]
    )
    loaders = get_dataloaders(
        {partition_id: client_datasets[partition_id]},  # only this client's data
        batch_size=cfg["client"]["batch_size"],
        num_workers=0,
    )
    model   = get_model(cfg["model"]["name"], cfg["model"]["num_classes"])

    client = DPBrainTumorClient(
        model=model,
        train_loader=loaders[partition_id]["train"],
        val_loader=loaders[partition_id]["val"],
        config=cfg["client"],
        target_epsilon=dp["differential_privacy"]["epsilon_values"][1],
        delta=dp["differential_privacy"]["delta"],
        max_grad_norm=dp["differential_privacy"]["max_grad_norm"],
    )
    return client.to_client()
