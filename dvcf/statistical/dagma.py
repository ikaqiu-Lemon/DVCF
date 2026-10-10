"""Linear DAGMA adapter with caller-owned fitting settings."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .graph_conversion import prepare_scaled_data, seed_numeric


def dagma(matrix, *, seed: int, lambda1: float, w_threshold: float,
          standardize: bool | str, zero_tolerance: float,
          scale_ratio_threshold: float, model_options: Mapping,
          fit_options: Mapping) -> np.ndarray:
    from dagma.linear import DagmaLinear
    seed_numeric(seed)
    values = prepare_scaled_data(matrix, standardize=standardize,
                                 zero_tolerance=zero_tolerance, scale_ratio_threshold=scale_ratio_threshold)
    model = DagmaLinear(loss_type="l2", **dict(model_options))
    weights = model.fit(values, lambda1=lambda1, w_threshold=w_threshold, **dict(fit_options))
    return (np.asarray(weights) != 0).astype(int)
