"""In-memory data validation and graph-encoding conversions."""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
import numpy as np


def require_parameters(parameters: Mapping, names: Sequence[str]) -> None:
    if not isinstance(parameters, Mapping):
        raise TypeError("Please input a parameter mapping.")
    for name in names:
        if name not in parameters:
            raise ValueError(f"<Please input your {name}>")
        _reject_placeholder(parameters[name], name)


def _reject_placeholder(value, name: str) -> None:
    if isinstance(value, str) and value.strip().startswith("<Please input"):
        raise ValueError(f"<Please input your {name}>")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_placeholder(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_placeholder(child, f"{name}[{index}]")


def validate_data(matrix) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("Expected a non-empty two-dimensional observation matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Observation matrix must contain finite numeric values.")
    return values


def validate_variables(variables: Sequence[str], size: int) -> tuple[str, ...]:
    names = tuple(variables)
    if len(names) != size or len(set(names)) != size:
        raise ValueError("Variables must be unique and match the data column order.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Variable names must be non-empty strings.")
    return names


def binary_adjacency(matrix, variable_count: int) -> np.ndarray:
    values = np.asarray(matrix)
    if values.shape != (variable_count, variable_count):
        raise ValueError(f"Expected a {variable_count} by {variable_count} graph matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Graph matrix must be finite.")
    adjacency = (values != 0).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def causallearn_graph_to_adj(graph_matrix) -> np.ndarray:
    """Keep causal-learn tail-to-arrow edges; row i denotes source i."""
    graph = np.asarray(graph_matrix)
    return ((graph.T == 1) & (graph == -1)).astype(int)


def pag_to_adjacency(graph_matrix) -> np.ndarray:
    """Retain tail/circle-to-arrow pairs; this is a projection, not a DAG guarantee."""
    graph = np.asarray(graph_matrix)
    adjacency = (((graph == -1) | (graph == 2)) & (graph.T == 1)).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def named_edges_to_adjacency(edges, variables: Sequence[str]) -> np.ndarray:
    positions = {name: index for index, name in enumerate(variables)}
    adjacency = np.zeros((len(positions), len(positions)), dtype=int)
    for parent, child in edges:
        adjacency[positions[parent], positions[child]] = 1
    return adjacency


def adjacency_to_edges(matrix, variables: Sequence[str]) -> frozenset[tuple[str, str]]:
    adjacency = binary_adjacency(matrix, len(variables))
    return frozenset((variables[i], variables[j]) for i, j in zip(*np.nonzero(adjacency)))


def prepare_scaled_data(matrix, *, standardize: bool | str,
                        zero_tolerance: float, scale_ratio_threshold: float) -> np.ndarray:
    """Apply the original scale-to-median rule with explicit thresholds."""
    values = validate_data(matrix)
    if zero_tolerance <= 0 or scale_ratio_threshold <= 0:
        raise ValueError("Scaling thresholds must be positive.")
    if standardize not in (True, False, "auto"):
        raise ValueError("standardize must be True, False, or 'auto'.")
    do_standardize = standardize is True
    if standardize == "auto":
        scales = np.std(values, axis=0)
        positive = scales[scales > zero_tolerance]
        do_standardize = len(positive) >= 2 and positive.max() / np.median(positive) > scale_ratio_threshold
    if do_standardize:
        scales = np.std(values, axis=0)
        scales[scales < zero_tolerance] = 1.0
        values = (values - values.mean(axis=0)) / scales
    return values


def seed_numeric(seed: int) -> None:
    """Seed only the numerical libraries needed by non-neural adapters."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("<Please input your integer random seed>")
    random.seed(int(seed))
    np.random.seed(int(seed))


def seed_torch(seed: int) -> None:
    """Import PyTorch only when a neural discovery algorithm is selected."""
    seed_numeric(seed)
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def break_cycles(adjacency, weights=None) -> np.ndarray:
    """Remove the weakest residual edge until Kahn's traversal covers the graph."""
    graph = binary_adjacency(adjacency, len(adjacency))
    edge_weights = graph.astype(float) if weights is None else np.asarray(weights, dtype=float)
    if edge_weights.shape != graph.shape or not np.isfinite(edge_weights).all():
        raise ValueError("Cycle-removal weights must match the finite graph matrix.")
    size = graph.shape[0]
    for _ in range(size * size):
        in_degree = graph.sum(axis=0).copy()
        queue = [node for node in range(size) if in_degree[node] == 0]
        visited = set()
        while queue:
            node = queue.pop(0)
            visited.add(node)
            for child in range(size):
                if graph[node, child] > 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        if len(visited) == size:
            return graph
        residual = sorted(set(range(size)) - visited)
        candidates = [(edge_weights[i, j], i, j) for i in residual for j in residual if graph[i, j] > 0]
        if not candidates:
            raise RuntimeError("Unable to remove a cyclic edge.")
        _, source, target = min(candidates)
        graph[source, target] = 0
    return graph
