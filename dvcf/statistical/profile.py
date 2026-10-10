"""Select an observable-data profile using caller-supplied rules."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from .graph_conversion import require_parameters, validate_data


@dataclass(frozen=True)
class DataProfile:
    sample_count: int
    variable_count: int
    sample_to_variable_ratio: float
    discrete_variable_count: int
    continuous_variable_count: int
    discrete_variable_ratio: float
    profile_id: str


def resolve_profile(matrix: np.ndarray, rules: Mapping) -> DataProfile:
    """All cardinality, ratio, dimension, and sample thresholds are required."""
    keys = (
        "max_discrete_unique", "max_discrete_unique_ratio", "discrete_dataset_ratio",
        "small_discrete_variable_max", "tiny_continuous_variable_max",
        "large_sample_min", "small_continuous_variable_max",
        "large_sample_low_dimensional_max",
    )
    require_parameters(rules, keys)
    values = validate_data(matrix)
    sample_count, variable_count = values.shape
    discrete_count = sum(
        np.unique(values[:, index]).size <= int(rules["max_discrete_unique"])
        or np.unique(values[:, index]).size / sample_count < float(rules["max_discrete_unique_ratio"])
        for index in range(variable_count)
    )
    discrete_ratio = discrete_count / variable_count
    if discrete_ratio >= float(rules["discrete_dataset_ratio"]):
        profile_id = "discrete_small" if variable_count <= int(rules["small_discrete_variable_max"]) else "discrete_medium"
    elif variable_count <= int(rules["tiny_continuous_variable_max"]):
        profile_id = "continuous_tiny"
    elif sample_count < int(rules["large_sample_min"]) and variable_count <= int(rules["small_continuous_variable_max"]):
        profile_id = "continuous_low_dimensional"
    elif variable_count <= int(rules["large_sample_low_dimensional_max"]):
        profile_id = "continuous_large_sample_low_dimensional"
    else:
        profile_id = "continuous_large_sample_medium_dimensional"
    return DataProfile(
        sample_count, variable_count, sample_count / variable_count,
        discrete_count, variable_count - discrete_count, discrete_ratio, profile_id,
    )
