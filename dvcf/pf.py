"""Precision-focused seed preservation, bounded augmentation, and selection."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .graph_utils import clamp, dot, has_path, is_acyclic, percentile, profile, source_jaccard, stable_mean
from .interfaces import DirectionalAssessment, Edge, StatisticalGraph, validate_sources, validate_variables
from .parameters import positive, quantile, require_group, scalar, vector


def _mean(rows, key):
    return stable_mean(row[key] for row in rows)


def _source_stats(source, sources, records, variables, params):
    rows = [records[edge] for edge in source.edges if edge in records]
    if not rows:
        return {"density": 0.0, "mean_reverse": 1.0, "mean_family": 0.0, "mean_text_support": 0.0,
                "mean_dropout": 0.0, "core_ratio": 0.0, "cross_source_jaccard": 0.0,
                "relative_coverage": 0.0, "confidence_score": 0.0}
    reverse = _mean(rows, "reverse_pressure")
    text = _mean(rows, "text_confidence_support")
    perturbation = _mean(rows, "dropout_component")
    core_fraction = stable_mean(float(row["integrated_core"]) for row in rows)
    return {
        "density": len(source.edges) / max(1, len(variables) * (len(variables) - 1)),
        "mean_reverse": reverse, "mean_family": _mean(rows, "source_family_count"),
        "mean_text_support": text, "mean_dropout": perturbation, "core_ratio": core_fraction,
        "cross_source_jaccard": source_jaccard(source.source_id, source.edges, sources),
        "relative_coverage": len(source.edges) / (max(len(item.edges) for item in sources) or 1),
        "confidence_score": dot(vector(params, "seed_weights", 4), (perturbation, core_fraction, text, 1 - reverse)),
    }


def _qualified(row, params):
    return (row["confidence_score"] >= scalar(params, "precision_min_score")
            and row["mean_reverse"] <= scalar(params, "precision_max_reverse")
            and row["mean_family"] >= scalar(params, "precision_min_family"))


def _choose_seeds(sources, stats, params):
    cutoff = percentile((stats[source.source_id]["density"] for source in sources), quantile(params, "sparse_density_quantile"))
    sparse = [source for source in sources if stats[source.source_id]["density"] <= cutoff and _qualified(stats[source.source_id], params)]
    precision = max(sparse, key=lambda source: (stats[source.source_id]["confidence_score"],
                                               stats[source.source_id]["cross_source_jaccard"],
                                               -stats[source.source_id]["density"], source.source_id)) if sparse else None
    def coverage_key(source):
        row = stats[source.source_id]
        balance = profile(row["relative_coverage"], scalar(params, "coverage_target"), positive(params, "coverage_width"))
        value = dot(vector(params, "coverage_weights", 7), (balance, row["mean_dropout"], row["core_ratio"],
                    min(row["mean_family"] / 3, 1.0), row["cross_source_jaccard"], row["mean_text_support"], 1 - row["mean_reverse"]))
        return value, row["confidence_score"], row["cross_source_jaccard"], source.source_id
    return precision, max(sources, key=coverage_key)


def _degree_deficits(edges, variables, target_degree):
    degrees = dict.fromkeys(variables, 0)
    for source, target in edges:
        degrees[source] += 1
        degrees[target] += 1
    return {v: clamp((target_degree - degrees[v]) / max(1.0, target_degree)) for v in variables}


def _refill_candidates(seed, records, variables, target_degree, params):
    deficits = _degree_deficits(seed, variables, target_degree)
    weights = vector(params, "addition_weights", 6)
    candidates = []
    for edge, row in records.items():
        if (edge in seed or row["source_family_count"] < scalar(params, "addition_min_family")
                or row["text_confidence_support"] < scalar(params, "addition_min_text")
                or row["reverse_pressure"] > scalar(params, "addition_max_reverse")
                or row["hard_reverse_guard_reject"]):
            continue
        endpoint = (deficits[edge[0]] + deficits[edge[1]]) / 2
        gain = dot(weights, (min(row["source_family_count"] / 3, 1.0), row["text_confidence_support"],
                            row["dropout_component"], row["cross_view_support"], 1 - row["reverse_pressure"], endpoint))
        candidates.append((edge, gain, row["source_family_count"], row["reverse_pressure"]))
    return sorted(candidates, key=lambda item: (item[1], item[2], -item[3], item[0][0], item[0][1]), reverse=True)


def _augment(seed, records, variables, target_degree, budget, minimum_gain, params):
    if not is_acyclic(seed):
        raise ValueError("PF requires an acyclic input seed to preserve every seed edge")
    selected, used_pairs, additions = set(seed), {tuple(sorted(edge)) for edge in seed}, 0
    # Gains are computed once against the original seed; do not update them after insertion.
    for edge, gain, _, _ in _refill_candidates(seed, records, variables, target_degree, params):
        pair = tuple(sorted(edge))
        if additions >= budget or gain < minimum_gain or pair in used_pairs or has_path(selected, edge[1], edge[0]):
            continue
        selected.add(edge)
        used_pairs.add(pair)
        additions += 1
    return frozenset(selected), additions


def _candidate_stats(edges, records, variables, target_degree, additions, params):
    rows = [records[edge] for edge in edges if edge in records]
    if rows:
        reverse = _mean(rows, "reverse_pressure")
        family = _mean(rows, "source_family_count")
        text = _mean(rows, "text_confidence_support")
        perturbation = _mean(rows, "dropout_component")
        confidence = dot(vector(params, "candidate_weights", 4), (perturbation, text, min(family / 3, 1.0), 1 - reverse))
    else:
        reverse = family = text = perturbation = confidence = 0.0
    deficits = _degree_deficits(edges, variables, target_degree)
    deficit = sum(deficits.values()) / max(1, len(variables))
    score = dot(vector(params, "graph_weights", 5), (confidence, 1 - reverse, min(family / 3, 1.0), 1 - deficit, perturbation))
    score -= scalar(params, "addition_penalty") * max(0, additions - math.sqrt(len(variables)))
    return {"confidence_score": confidence, "mean_reverse": reverse, "mean_family": family,
            "coverage_deficit": deficit, "graph_score": score}


def select_pf(
    variables: Sequence[str], statistical_graphs: Sequence[StatisticalGraph],
    assessment: DirectionalAssessment, parameters: Mapping,
) -> frozenset[Edge]:
    """Return PF's qualified precision candidate or the best structural candidate.

    Budget scales and gain cutoffs are ordered (precision, refill, balanced).
    Candidate identities survive even when two candidates have equal edge sets.
    """
    variables = validate_variables(variables)
    sources = validate_sources(statistical_graphs, variables)
    params = require_group(parameters, "pf")
    stats = {source.source_id: _source_stats(source, sources, assessment.records, variables, params) for source in sources}
    sparse, coverage = _choose_seeds(sources, stats, params)
    precision = sparse or coverage
    target_degree = max(1.0, percentile([len(source.edges) for source in sources] + [len(assessment.aggregate_edges)],
                                     quantile(params, "target_degree_quantile")) * 2 / max(1, len(variables)))
    scales = vector(params, "budget_scales", 3)
    cutoffs = vector(params, "gain_cutoffs", 3)
    candidates, summaries = {}, {}
    for index, (role, seed) in enumerate((("precision_preserve", precision.edges), ("recall_refill", coverage.edges),
                                        ("balanced", assessment.aggregate_edges))):
        if scales[index] < 0:
            raise ValueError("PF budget scales must be nonnegative")
        budget = max(1, round(math.sqrt(len(variables)) * scales[index]))
        edges, additions = _augment(seed, assessment.records, variables, target_degree, budget, cutoffs[index], params)
        candidates[role] = edges
        summaries[role] = _candidate_stats(edges, assessment.records, variables, target_degree, additions, params)
    if sparse is not None and _qualified(summaries["precision_preserve"], params):
        return candidates["precision_preserve"]
    chosen = max(summaries, key=lambda role: (summaries[role]["graph_score"], -summaries[role]["coverage_deficit"], role))
    return candidates[chosen]
