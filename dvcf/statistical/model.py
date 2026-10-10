"""Generate caller-selected statistical graphs from an in-memory observation matrix."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import inspect

from ..interfaces import StatisticalGraph
from .profile import resolve_profile
from .graph_conversion import adjacency_to_edges, require_parameters, validate_data, validate_variables


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    algorithm: str
    algorithm_family: str


_ADAPTERS = {
    "pc": ("causal_learn", "pc"),
    "fci": ("causal_learn", "fci"),
    "ges": ("causal_learn", "ges"),
    "ges_bdeu": ("causal_learn", "ges_bdeu"),
    "direct_lingam": ("causal_learn", "direct_lingam"),
    "hill_climb": ("pgmpy_search", "hill_climb"),
    "bdeu_tabu": ("pgmpy_search", "bdeu_tabu"),
    "mmhc": ("pgmpy_search", "mmhc"),
    "dag_gnn": ("gcastle", "dag_gnn"),
    "gae": ("gcastle", "gae"),
    "golem": ("gcastle", "golem"),
    "dagma": ("dagma", "dagma"),
    "bootstrap_reci": ("bootstrap_reci", "bootstrap_reci"),
}


def discover_statistical_graphs(matrix, variables: Sequence[str], *, rules: Mapping,
                                 portfolios: Mapping[str, Sequence[SourceSpec]],
                                 parameters: Mapping[str, Mapping], seed: int) -> list[StatisticalGraph]:
    """Parameters are keyed by source_id; portfolios map profile_id to SourceSpec.

    The caller supplies every portfolio, experiment parameter, and seed. The
    adapter registry lists supported algorithms, not a preset source selection.
    Each output edge (u, v) means u -> v in the supplied variable order.
    """
    values = validate_data(matrix)
    names = validate_variables(variables, values.shape[1])
    profile = resolve_profile(values, rules)
    if profile.profile_id not in portfolios:
        raise ValueError(f"<Please input your portfolio for {profile.profile_id}>")
    selections = tuple(portfolios[profile.profile_id])
    if not selections:
        raise ValueError("Please input a non-empty statistical-source portfolio.")
    source_ids = [selection.source_id for selection in selections]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Statistical source_id values must be unique.")
    result = []
    for selection in selections:
        if not selection.source_id or not selection.algorithm_family:
            raise ValueError("Please input non-empty source and algorithm-family identifiers.")
        require_parameters(
            {"source_id": selection.source_id, "algorithm_family": selection.algorithm_family},
            ("source_id", "algorithm_family"),
        )
        if selection.algorithm not in _ADAPTERS:
            raise ValueError(f"Unsupported algorithm: {selection.algorithm}")
        module_name, function_name = _ADAPTERS[selection.algorithm]
        function = getattr(importlib.import_module(f".{module_name}", __package__), function_name)
        if selection.source_id not in parameters:
            raise ValueError(f"<Please input your parameters for {selection.source_id}>")
        supplied = dict(parameters[selection.source_id])
        if "seed" in supplied or "variables" in supplied:
            raise ValueError("Supply seed and variables at the generation interface, not inside source parameters.")
        required = [key for key in inspect.signature(function).parameters
                    if key not in {"matrix", "seed", "variables"}]
        require_parameters(supplied, required)
        supplied["seed"] = seed
        if "variables" in inspect.signature(function).parameters:
            supplied["variables"] = names
        adjacency = function(values, **supplied)
        result.append(StatisticalGraph(
            source_id=selection.source_id,
            algorithm_family=selection.algorithm_family,
            edges=adjacency_to_edges(adjacency, names),
        ))
    return result
