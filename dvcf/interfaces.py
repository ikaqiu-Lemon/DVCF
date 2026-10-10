"""In-memory inputs shared by the DVCF core and external source adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

Edge = tuple[str, str]


@dataclass(frozen=True)
class StatisticalGraph:
    source_id: str
    algorithm_family: str
    edges: frozenset[Edge]


@dataclass(frozen=True)
class DirectionalView:
    scores: Mapping[Edge, float]
    base_adjacency: frozenset[Edge]
    citation_adjacency: frozenset[Edge]
    base_predicted_edges: frozenset[Edge]
    citation_predicted_edges: frozenset[Edge]


@dataclass(frozen=True)
class StructuralView:
    base_graph: frozenset[Edge]
    citation_graph: frozenset[Edge]


@dataclass(frozen=True)
class DirectionalAssessment:
    records: Mapping[Edge, Mapping[str, Any]]
    context: Mapping[str, Any]
    aggregate_edges: frozenset[Edge]


def validate_variables(variables: Sequence[str]) -> tuple[str, ...]:
    result = tuple(variables)
    if len(result) < 2 or any(not isinstance(v, str) or not v for v in result):
        raise ValueError("variables must contain at least two nonempty names")
    if len(set(result)) != len(result):
        raise ValueError("variable names must be unique")
    return result


def validate_edges(edges, variables: Sequence[str]) -> frozenset[Edge]:
    names = set(variables)
    result = frozenset(edges)
    for edge in result:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("each edge must be a (source, target) tuple")
        source, target = edge
        if source not in names or target not in names or source == target:
            raise ValueError(f"edge {edge!r} must connect two distinct input variables")
    return result


def validate_sources(sources: Sequence[StatisticalGraph], variables: Sequence[str]) -> tuple[StatisticalGraph, ...]:
    result = tuple(sorted(sources, key=lambda item: item.source_id))
    if not result:
        raise ValueError("at least one statistical graph is required")
    if len({item.source_id for item in result}) != len(result):
        raise ValueError("statistical source_id values must be unique")
    for item in result:
        if not item.source_id or not item.algorithm_family:
            raise ValueError("each statistical source requires source_id and algorithm_family")
        validate_edges(item.edges, variables)
    return result
