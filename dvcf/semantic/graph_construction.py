"""Score-ranked semantic graph construction with budgets and DAG constraints."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from typing import Mapping, Sequence

from dvcf.interfaces import Edge


@dataclass(frozen=True)
class GraphParameters:
    target_edges_ratio: float
    threshold_min: float
    max_out_degree_factor: float
    minimum_out_degree_cap: int

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in (self.target_edges_ratio, self.threshold_min, self.max_out_degree_factor)):
            raise ValueError("Graph parameters must be finite.")
        if self.target_edges_ratio <= 0 or self.max_out_degree_factor <= 0:
            raise ValueError("Edge ratio and degree factor must be positive.")
        if not 0 <= self.threshold_min <= 1 or not isinstance(self.minimum_out_degree_cap, int) or self.minimum_out_degree_cap < 0:
            raise ValueError("Invalid threshold or minimum out-degree cap.")


def has_path(edges: set[Edge], source: str, target: str) -> bool:
    stack = [source]
    seen: set[str] = set()
    adjacency: dict[str, list[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
    return False


def construct_graph(
    variables: Sequence[str], scores: Mapping[Edge, float], *, parameters: GraphParameters
) -> frozenset[Edge]:
    """Apply the source top-K → degree-cap pipeline; score ties use variable order."""
    names = tuple(variables)
    if len(names) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Provide at least two distinct variable names.")
    pairs = [(left, right) for left in names for right in names if left != right]
    if any(edge not in scores for edge in pairs):
        raise ValueError("Provide scores for every non-self ordered pair.")
    if any(not isfinite(float(scores[edge])) or not 0 <= scores[edge] <= 1 for edge in pairs):
        raise ValueError("Semantic generation scores must be finite and in [0, 1].")
    candidates = sorted((edge for edge in pairs if scores[edge] >= parameters.threshold_min),
                        key=lambda edge: scores[edge], reverse=True)
    budget = max(len(names), int(parameters.target_edges_ratio * len(names)))
    selected: set[Edge] = set()
    for edge in candidates:
        if len(selected) >= budget:
            break
        if not has_path(selected, edge[1], edge[0]):
            selected.add(edge)
    cap = max(parameters.minimum_out_degree_cap, ceil(sqrt(len(names)) * parameters.max_out_degree_factor))
    indices = {name: index for index, name in enumerate(names)}
    for source in names:
        outgoing = sorted((edge for edge in selected if edge[0] == source),
                          key=lambda edge: (scores[edge], indices[edge[1]]), reverse=True)
        selected.difference_update(outgoing[cap:])
    return frozenset(selected)
