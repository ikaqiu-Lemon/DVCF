"""Selected gCastle adapters; no model or device is loaded at import time."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .graph_conversion import prepare_scaled_data, seed_torch


def dag_gnn(matrix, *, seed: int, w_threshold: float, epochs: int,
            standardize: bool | str, zero_tolerance: float,
            scale_ratio_threshold: float, device_type: str, options: Mapping) -> np.ndarray:
    from castle.algorithms import DAG_GNN
    seed_torch(seed)
    values = prepare_scaled_data(matrix, standardize=standardize,
                                 zero_tolerance=zero_tolerance, scale_ratio_threshold=scale_ratio_threshold)
    model = DAG_GNN(epochs=epochs, graph_threshold=w_threshold,
                    device_type=device_type, **dict(options))
    model.learn(values)
    return (np.asarray(model.causal_matrix) != 0).astype(int)


def gae(matrix, *, seed: int, w_threshold: float, epochs: int,
        standardize: bool | str, zero_tolerance: float,
        scale_ratio_threshold: float, device_type: str, options: Mapping) -> np.ndarray:
    from castle.algorithms import GAE
    seed_torch(seed)
    values = prepare_scaled_data(matrix, standardize=standardize,
                                 zero_tolerance=zero_tolerance, scale_ratio_threshold=scale_ratio_threshold)
    model = GAE(input_dim=values.shape[1], epochs=epochs, graph_thresh=w_threshold,
                device_type=device_type, **dict(options))
    model.learn(values)
    return (np.asarray(model.causal_matrix) != 0).astype(int)


def golem(matrix, *, seed: int, lambda_1: float, lambda_2: float,
          w_threshold: float, num_iter: int, standardize: bool | str,
          zero_tolerance: float, scale_ratio_threshold: float,
          device_type: str, options: Mapping) -> np.ndarray:
    from castle.algorithms import GOLEM
    seed_torch(seed)
    values = prepare_scaled_data(matrix, standardize=standardize,
                                 zero_tolerance=zero_tolerance, scale_ratio_threshold=scale_ratio_threshold)
    model = GOLEM(lambda_1=lambda_1, lambda_2=lambda_2, num_iter=num_iter,
                  graph_thres=w_threshold, device_type=device_type, **dict(options))
    model.learn(values)
    return (np.asarray(model.causal_matrix) != 0).astype(int)
