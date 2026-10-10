"""PGMPY hill-climbing and MMHC adapters with explicit scoring inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from .graph_conversion import named_edges_to_adjacency, seed_numeric, validate_data, validate_variables


def _frame(matrix, variables, discrete: bool):
    import pandas as pd
    values = validate_data(matrix)
    names = validate_variables(variables, values.shape[1])
    if discrete:
        values = values.astype(int)
    return pd.DataFrame(values, columns=names), names


def hill_climb(matrix, *, variables: Sequence[str], seed: int, scoring_method,
               discrete: bool, max_indegree: int | None, max_iter: int,
               options: Mapping) -> np.ndarray:
    from pgmpy.estimators import HillClimbSearch
    seed_numeric(seed)
    data, names = _frame(matrix, variables, discrete)
    graph = HillClimbSearch(data).estimate(
        scoring_method=scoring_method, max_indegree=max_indegree,
        max_iter=max_iter, show_progress=False, **dict(options),
    )
    return named_edges_to_adjacency(graph.edges(), names)


def bdeu_tabu(matrix, *, variables: Sequence[str], seed: int,
              equivalent_sample_size: float, max_indegree: int | None,
              tabu_length: int, max_iter: int, options: Mapping) -> np.ndarray:
    """BDeu-scored HillClimbSearch with a tabu list, not BOSS."""
    from pgmpy.estimators import HillClimbSearch
    try:
        from pgmpy.estimators import BDeu
    except ImportError:
        from pgmpy.estimators import BDeuScore as BDeu
    seed_numeric(seed)
    data, names = _frame(matrix, variables, True)
    score = BDeu(data, equivalent_sample_size=equivalent_sample_size)
    graph = HillClimbSearch(data).estimate(
        scoring_method=score, max_indegree=max_indegree, tabu_length=tabu_length,
        max_iter=max_iter, show_progress=False, **dict(options),
    )
    return named_edges_to_adjacency(graph.edges(), names)


def mmhc(matrix, *, variables: Sequence[str], seed: int,
         equivalent_sample_size: float, significance_level: float,
         options: Mapping) -> np.ndarray:
    from pgmpy.estimators import MmhcEstimator
    try:
        from pgmpy.estimators import BDeu
    except ImportError:
        from pgmpy.estimators import BDeuScore as BDeu
    seed_numeric(seed)
    data, names = _frame(matrix, variables, True)
    score = BDeu(data, equivalent_sample_size=equivalent_sample_size)
    graph = MmhcEstimator(data).estimate(
        scoring_method=score, significance_level=significance_level, **dict(options),
    )
    return named_edges_to_adjacency(graph.edges(), names)
