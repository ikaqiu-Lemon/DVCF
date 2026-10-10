"""Required parameter schema; this module contains no experiment defaults.

Vector positions are documented at their point of use. Supply numeric values
from the method specification or your own declared configuration. Placeholder
strings deliberately raise ValueError instead of producing an output graph.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


PARAMETER_SCHEMA = {
    "cross_view": (
        "text_profile_weights", "conflict_text_min", "conflict_data_max", "conflict_margin_max",
        "pair_support_min", "high_data_min", "high_data_with_text_min", "high_family_min",
        "addition_data_with_text_min", "addition_data_min", "addition_family_min",
        "anchor_density_target", "reliability_weights", "anchor_weights", "majority_weights",
        "preservation_weights", "channel_budget_ratios",
    ),
    "directional": (
        "decimal_places", "support_quantile", "high_confidence_quantile", "budget_quantile",
        "budget_reduction", "reverse_pressure_weights", "text_support_weights", "cross_support_weights",
        "role_bonus_weight", "margin_shift", "margin_scale", "signal_data_min", "signal_family_min",
        "weak_family_max", "evidence_weights", "membership_forward_offset", "membership_reverse_offset",
        "perturbation_weights", "weakness_weights", "weakness_support_target", "corroboration_weights",
        "corroboration_margin_scale", "core_quantile", "core_signal_min", "core_direction_margin",
        "core_stability_weights", "core_conflict_weights", "residual_support_weights",
        "residual_conflict_weights", "residual_offset", "zero_vote_max", "zero_vote_text_max",
        "minimum_correction", "admission_signal_min", "pair_arbitration_margin",
    ),
    "pf": (
        "sparse_density_quantile", "target_degree_quantile", "precision_min_score",
        "precision_max_reverse", "precision_min_family", "coverage_target", "coverage_width",
        "seed_weights", "coverage_weights", "addition_weights", "candidate_weights", "graph_weights",
        "addition_penalty", "budget_scales", "gain_cutoffs", "addition_min_family",
        "addition_min_text", "addition_max_reverse",
    ),
    "rf": (
        "statistical_weights", "text_weights", "sample_ratio_scale", "density_penalty",
        "statistical_min_score", "statistical_min_coverage", "statistical_max_reverse",
        "statistical_max_density", "coverage_min_score", "coverage_min_coverage",
        "coverage_max_density", "coverage_max_reverse", "coverage_min_family", "coverage_min_overlap",
        "text_min_score", "text_max_density", "text_min_support", "text_min_nontext",
        "text_max_reverse", "text_preference_margin", "fallback_density_target",
    ),
}


def placeholder_parameters() -> dict[str, dict[str, str]]:
    """Return a template to fill explicitly, never a runnable configuration."""
    return {group: {key: f"<Please input your {group}.{key}>" for key in keys}
            for group, keys in PARAMETER_SCHEMA.items()}


def require_group(parameters: Mapping, group: str) -> Mapping:
    if not isinstance(parameters, Mapping) or not isinstance(parameters.get(group), Mapping):
        raise ValueError(f"missing required parameter group: {group}")
    result = parameters[group]
    for key in PARAMETER_SCHEMA[group]:
        if key not in result:
            raise ValueError(f"missing required parameter: {group}.{key}")
        value = result[key]
        if value is None or isinstance(value, str):
            raise ValueError(f"fill {group}.{key} with a numeric value or numeric vector; placeholders are not executable")
    return result


def scalar(group: Mapping, key: str) -> float:
    value = group[key]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def integer(group: Mapping, key: str) -> int:
    value = scalar(group, key)
    if value != int(value) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return int(value)


def vector(group: Mapping, key: str, length: int) -> tuple[float, ...]:
    value = group[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{key} must be a numeric vector of length {length}")
    return tuple(scalar({key: item}, key) for item in value)


def quantile(group: Mapping, key: str) -> float:
    value = scalar(group, key)
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def positive(group: Mapping, key: str) -> float:
    value = scalar(group, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value
