"""Lazy adapters for causal-learn; every experimental setting is an input."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .graph_conversion import (
    causallearn_graph_to_adj, pag_to_adjacency, seed_numeric, validate_data,
)


def pc(matrix, *, seed: int, alpha: float, independence_test: str,
       uc_rule: int, uc_priority: int, options: Mapping) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc as inner_pc
    seed_numeric(seed)
    result = inner_pc(validate_data(matrix), alpha=alpha, indep_test=independence_test,
                      uc_rule=uc_rule, uc_priority=uc_priority,
                      show_progress=False, **dict(options))
    return causallearn_graph_to_adj(result.G.graph)


def fci(matrix, *, seed: int, independence_test: str, alpha: float,
        depth: int, options: Mapping) -> np.ndarray:
    from causallearn.search.ConstraintBased.FCI import fci as inner_fci
    seed_numeric(seed)
    graph, _ = inner_fci(validate_data(matrix), independence_test_method=independence_test,
                         alpha=alpha, depth=depth, verbose=False,
                         show_progress=False, **dict(options))
    return pag_to_adjacency(graph.graph)


def ges(matrix, *, seed: int, score_func: str, subsample_n: int | str | None,
        subsample_trigger: int, auto_subsample_size: int, options: Mapping) -> np.ndarray:
    """The automatic subsampling rule uses explicit trigger and target sizes."""
    from causallearn.search.ScoreBased.GES import ges as inner_ges
    seed_numeric(seed)
    values = validate_data(matrix)
    if subsample_n == "auto":
        if subsample_trigger <= 0 or auto_subsample_size <= 0:
            raise ValueError("Automatic subsampling thresholds must be positive.")
        subsample_n = auto_subsample_size if values.shape[0] > subsample_trigger else None
    if subsample_n is not None:
        if not isinstance(subsample_n, int) or subsample_n <= 0:
            raise ValueError("subsample_n must be a positive integer or None.")
        if values.shape[0] > subsample_n:
            rows = np.random.default_rng(seed).choice(values.shape[0], size=subsample_n, replace=False)
            values = values[rows]
    return causallearn_graph_to_adj(inner_ges(values, score_func=score_func, **dict(options))["G"].graph)


def ges_bdeu(matrix, *, seed: int, options: Mapping) -> np.ndarray:
    """The BDeu-scored GES variant; this is not a separate FGS implementation."""
    from causallearn.search.ScoreBased.GES import ges as inner_ges
    seed_numeric(seed)
    result = inner_ges(validate_data(matrix), score_func="local_score_BDeu", **dict(options))
    return causallearn_graph_to_adj(result["G"].graph)


def direct_lingam(matrix, *, seed: int, weight_threshold: float,
                  zero_variance_noise_scale: float, zero_variance_seed: int,
                  adjacency_layout: str,
                  options: Mapping) -> np.ndarray:
    """adjacency_layout explicitly declares the library matrix's row convention.

    'row_target' transposes the library result to source-by-target;
    'row_source' preserves its row direction. This choice has no default.
    """
    from causallearn.search.FCMBased.lingam.direct_lingam import DirectLiNGAM
    seed_numeric(seed)
    values = validate_data(matrix).copy()
    if weight_threshold < 0 or zero_variance_noise_scale < 0:
        raise ValueError("Weight and noise thresholds must be non-negative.")
    constant = np.std(values, axis=0) == 0
    if constant.any():
        if zero_variance_noise_scale == 0:
            raise ValueError("Constant columns require a positive noise scale or caller preprocessing.")
        values[:, constant] += np.random.default_rng(zero_variance_seed).normal(
            scale=zero_variance_noise_scale, size=(values.shape[0], int(constant.sum())),
        )
    model = DirectLiNGAM(**dict(options))
    model.fit(values)
    weights = np.asarray(model.adjacency_matrix_)
    if adjacency_layout == "row_target":
        weights = weights.T
    elif adjacency_layout != "row_source":
        raise ValueError("adjacency_layout must be 'row_source' or 'row_target'.")
    return (np.abs(weights) > weight_threshold).astype(int)
