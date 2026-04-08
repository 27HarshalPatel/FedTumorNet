"""Flower aggregation strategies: FedAvg and FedProx."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics with sample-count weighting."""
    total = sum(n for n, _ in metrics)
    agg = {}
    for key in (metrics[0][1].keys() if metrics else []):
        agg[key] = sum(m[key] * n for n, m in metrics) / total if total > 0 else 0.0
    return agg

def get_strategy(config: Dict, evaluate_fn=None):
    """Factory: returns configured Flower strategy."""
    name = config["strategy"]["name"].lower()
    mu   = config["strategy"].get("fedprox_mu", 0.01)

    common = dict(
        fraction_fit=config["federation"]["fraction_fit"],
        fraction_evaluate=config["federation"]["fraction_evaluate"],
        min_fit_clients=config["federation"]["min_fit_clients"],
        min_evaluate_clients=config["federation"]["min_evaluate_clients"],
        min_available_clients=config["federation"]["min_available_clients"],
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate_fn,
    )

    if name == "fedprox":
        return FedProx(proximal_mu=mu, **common)
    else:  # fedavg (default)
        return FedAvg(**common)
