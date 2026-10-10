"""Deterministic graph and numerical primitives used by the fusion core."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .interfaces import Edge, StatisticalGraph


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def stable_mean(values: Iterable[float]) -> float:
    items = sorted(float(v) for v in values)
    return math.fsum(items) / len(items) if items else 0.0


def percentile(values: Iterable[float], fraction: float) -> float:
    items = sorted(values)
    if not 0 <= fraction <= 1:
        raise ValueError("quantile fraction must be in [0, 1]")
    if not items:
        return 0.0
    position = (len(items) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return items[low] if low == high else items[low] * (1 - (position - low)) + items[high] * (position - low)


def median(values: Iterable[float]) -> float:
    return percentile(values, 1 / 2)


def population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def dot(weights: Sequence[float], values: Sequence[float]) -> float:
    if len(weights) != len(values):
        raise ValueError("weights and features must have the same length")
    result = 0.0
    for weight, value in zip(weights, values):
        result += weight * value
    return result


def has_path(edges: Iterable[Edge], source: str, target: str) -> bool:
    ordered = sorted(edges)
    stack, seen = [source], set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(v for u, v in ordered if u == node and v not in seen)
    return False


def cycle_repair(edges: Iterable[Edge]) -> frozenset[Edge]:
    """RF repair: ascending endpoint tuples, unused pairs, then DAG admission."""
    selected, used_pairs = set(), set()
    for edge in sorted(edges):
        pair = tuple(sorted(edge))
        if pair in used_pairs or has_path(selected, edge[1], edge[0]):
            continue
        selected.add(edge)
        used_pairs.add(pair)
    return frozenset(selected)


def prune_with_label_key(edges: Iterable[Edge]) -> frozenset[Edge]:
    """Structural-anchor repair uses the original source->target string order."""
    selected = set()
    for edge in sorted(edges, key=lambda item: f"{item[0]}->{item[1]}"):
        if not has_path(selected, edge[1], edge[0]):
            selected.add(edge)
    return frozenset(selected)


def is_acyclic(edges: Iterable[Edge]) -> bool:
    current = set()
    for source, target in sorted(edges):
        if has_path(current, target, source):
            return False
        current.add((source, target))
    return True


def source_jaccard(source_id: str, edges: frozenset[Edge], sources: Sequence[StatisticalGraph]) -> float:
    values = []
    for other in sorted(sources, key=lambda item: item.source_id):
        if other.source_id == source_id:
            continue
        union = edges | other.edges
        values.append(len(edges & other.edges) / len(union) if union else 0.0)
    return sum(values) / len(values) if values else 0.0


def profile(value: float, target: float, width: float) -> float:
    return clamp(1 - abs(value - target) / width) if width > 0 else 0.0
