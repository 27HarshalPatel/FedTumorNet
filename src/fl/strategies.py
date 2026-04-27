"""Flower aggregation strategies: FedAvg and FedProx."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate scalar metrics with sample-count weighting.

    Robust to: clients reporting different metric keys, non-numeric values,
    and zero total weight (e.g. all clients had empty val shards).
    """
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total <= 0:
        return {}
    # Only aggregate keys present in every client's metrics dict and numeric.
    common_keys = set.intersection(*[set(m.keys()) for _, m in metrics])
    agg: Metrics = {}
    for key in common_keys:
        try:
            agg[key] = sum(float(m[key]) * n for n, m in metrics) / total
        except (TypeError, ValueError):
            continue
    return agg

def get_strategy(config: Dict, evaluate_fn=None, initial_parameters=None):
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
    if initial_parameters is not None:
        common["initial_parameters"] = initial_parameters

    if name == "fedprox":
        return FedProx(proximal_mu=mu, **common)
    else:  # fedavg (default)
        return FedAvg(**common)
