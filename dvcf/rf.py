"""Recall-focused complete-proposal scoring, ordered selection, and DAG repair."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .graph_utils import cycle_repair, dot, source_jaccard, stable_mean
from .interfaces import DirectionalAssessment, DirectionalView, Edge, StatisticalGraph, validate_edges, validate_sources, validate_variables
from .parameters import positive, require_group, scalar, vector


def _common_stats(edges, records, variables, aggregate, sample_count, params):
    rows = [records[edge] for edge in edges if edge in records]
    if rows:
        reverse = stable_mean(row["reverse_pressure"] for row in rows)
        family = stable_mean(row["source_family_count"] for row in rows)
        text = stable_mean(row["text_confidence_support"] for row in rows)
        perturbation = stable_mean(row["dropout_component"] for row in rows)
        nontext = stable_mean(float(row["data_vote_count"] > 0 or row["cross_view_support"] > 0) for row in rows)
        overlap = len(edges & aggregate) / len(edges) if edges else 0.0
    else:
        reverse = family = text = perturbation = nontext = overlap = 0.0
    return {"edges": edges, "edge_count": len(edges), "density": len(edges) / max(1, len(variables) * (len(variables) - 1)),
            "mean_reverse": reverse, "mean_family": family, "mean_text_support": text, "mean_dropout": perturbation,
            "fusion_alignment": overlap, "text_non_text_support_ratio": nontext,
            "sample_term": min(sample_count / max(1, len(variables)) / positive(params, "sample_ratio_scale"), 1.0)}


def _statistical_candidate(source, sources, assessment, variables, sample_count, params):
    row = _common_stats(source.edges, assessment.records, variables, assessment.aggregate_edges, sample_count, params)
    relative_coverage = len(source.edges) / max([len(item.edges) for item in sources] + [1])
    agreement = source_jaccard(source.source_id, source.edges, sources)
    score = dot(vector(params, "statistical_weights", 8), (relative_coverage, row["mean_dropout"], min(row["mean_family"] / 3, 1.0),
                agreement, 1 - row["mean_reverse"], row["mean_text_support"], row["fusion_alignment"], row["sample_term"]))
    score -= scalar(params, "density_penalty") * max(0.0, row["density"] - scalar(params, "statistical_max_density"))
    return {**row, "source_name": source.source_id, "relative_coverage": relative_coverage, "score": score}


def _text_candidate(view, assessment, variables, sample_count, params):
    # DirectView supplies this candidate; StructView is reserved for cross-view support.
    edges = view.citation_adjacency | view.citation_predicted_edges
    validate_edges(edges, variables)
    row = _common_stats(edges, assessment.records, variables, assessment.aggregate_edges, sample_count, params)
    compactness = 1 - min(row["density"] / positive(params, "text_max_density"), 1.0)
    score = dot(vector(params, "text_weights", 8), (row["mean_text_support"], row["text_non_text_support_ratio"], 1 - row["mean_reverse"],
                min(row["mean_family"] / 3, 1.0), row["fusion_alignment"], row["mean_dropout"], row["sample_term"], compactness))
    return {**row, "source_name": "text_gated", "score": score}


def _choose(statistical_candidates, text, params):
    best = max(statistical_candidates, key=lambda row: (row["score"], row["relative_coverage"], -row["mean_reverse"], -row["density"], row["source_name"]))
    source_ok = (best["score"] >= scalar(params, "statistical_min_score")
                 and best["relative_coverage"] >= scalar(params, "statistical_min_coverage")
                 and best["mean_reverse"] <= scalar(params, "statistical_max_reverse")
                 and best["density"] <= scalar(params, "statistical_max_density"))
    text_ok = (text["score"] >= scalar(params, "text_min_score")
               and text["density"] <= scalar(params, "text_max_density")
               and text["mean_text_support"] >= scalar(params, "text_min_support")
               and text["text_non_text_support_ratio"] >= scalar(params, "text_min_nontext")
               and text["mean_reverse"] <= scalar(params, "text_max_reverse"))
    coverage = [row for row in statistical_candidates
                if row["score"] >= scalar(params, "coverage_min_score")
                and row["relative_coverage"] >= scalar(params, "coverage_min_coverage")
                and row["density"] <= scalar(params, "coverage_max_density")
                and row["mean_reverse"] <= scalar(params, "coverage_max_reverse")
                and row["mean_family"] >= scalar(params, "coverage_min_family")
                and row["fusion_alignment"] >= scalar(params, "coverage_min_overlap")]
    if coverage and text_ok:
        best_coverage = max(coverage, key=lambda row: (row["relative_coverage"], row["score"], row["fusion_alignment"], -row["mean_reverse"], row["source_name"]))
        if best_coverage["edge_count"] > text["edge_count"] and best_coverage["mean_family"] >= text["mean_family"]:
            return best_coverage
    if text_ok and (not source_ok or text["score"] >= best["score"] + scalar(params, "text_preference_margin")):
        return text
    if source_ok:
        return best
    return max(statistical_candidates, key=lambda row: (row["score"], -abs(row["density"] - scalar(params, "fallback_density_target")),
                                                       -row["mean_reverse"], row["source_name"]))


def select_rf(
    variables: Sequence[str], statistical_graphs: Sequence[StatisticalGraph], directional_view: DirectionalView,
    assessment: DirectionalAssessment, sample_count: int, parameters: Mapping,
) -> frozenset[Edge]:
    """Score complete graphs, apply the four ordered branches, then repair the winner."""
    variables = validate_variables(variables)
    sources = validate_sources(statistical_graphs, variables)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a nonnegative integer")
    params = require_group(parameters, "rf")
    candidates = [_statistical_candidate(source, sources, assessment, variables, sample_count, params) for source in sources]
    text = _text_candidate(directional_view, assessment, variables, sample_count, params)
    selected = _choose(candidates, text, params)
    return cycle_repair(selected["edges"])
