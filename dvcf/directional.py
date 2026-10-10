"""Shared directional evidence, perturbed utilities, and DAG aggregation."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .graph_utils import clamp, dot, has_path, median, percentile, population_std
from .interfaces import DirectionalAssessment, DirectionalView, Edge, StatisticalGraph, validate_edges, validate_sources, validate_variables
from .parameters import integer, positive, quantile, require_group, scalar, vector


def _build_source_pool(variables, sources, view, support, params):
    decimals = integer(params, "decimal_places")
    number = lambda value: float(f"{value:.{decimals}f}")
    edges = [(s, t) for s in variables for t in variables if s != t]
    for item in (view.base_adjacency, view.citation_adjacency, view.base_predicted_edges, view.citation_predicted_edges):
        validate_edges(item, variables)
    validate_edges(view.scores, variables)
    confidence = {edge: float(view.scores.get(edge, 0.0)) for edge in edges}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidence.values()):
        raise ValueError("DirectView scores must be finite and in [0, 1]")
    for edge in edges:
        if edge not in support:
            raise ValueError(f"missing cross-view support for {edge!r}")
        for key in ("selected_count", "rule_count", "direction_support"):
            if key not in support[edge] or not math.isfinite(float(support[edge][key])) or support[edge][key] < 0:
                raise ValueError(f"invalid cross-view support field {key} for {edge!r}")
    primary = view.base_adjacency | view.base_predicted_edges
    gated = view.citation_adjacency | view.citation_predicted_edges
    positive_scores = [confidence[edge] for edge in sorted(confidence) if confidence[edge] > 0]
    support_cut = percentile(positive_scores, quantile(params, "support_quantile"))
    high_cut = percentile(positive_scores, quantile(params, "high_confidence_quantile"))
    high_primary = frozenset(edge for edge in primary if confidence[edge] >= high_cut)
    graph_supported = frozenset(edge for edge in edges if support[edge]["selected_count"] or support[edge]["direction_support"])
    members = set().union(*(item.edges for item in sources), primary, gated, high_primary, graph_supported)
    counts = [len(item.edges) for item in sources] + [len(primary), len(gated), len(high_primary)]
    floor = max(1, math.floor(median(counts)))
    ceiling = max(floor, math.ceil(percentile(counts, quantile(params, "budget_quantile"))) - integer(params, "budget_reduction"))
    reverse_w = vector(params, "reverse_pressure_weights", 6)
    text_w = vector(params, "text_support_weights", 5)
    cross_w = vector(params, "cross_support_weights", 3)
    rows = []
    for source, target in edges:
        edge, reverse = (source, target), (target, source)
        votes = sum(edge in item.edges for item in sources)
        reverse_votes = sum(reverse in item.edges for item in sources)
        base_flag, gated_flag = edge in primary, edge in gated
        predicted_flag = edge in view.base_predicted_edges
        gated_predicted_flag = edge in view.citation_predicted_edges
        cv, reverse_cv = support[edge], support[reverse]
        family = sum((bool(votes), any((base_flag, gated_flag, predicted_flag, gated_predicted_flag)) or confidence[edge] >= support_cut,
                      cv["selected_count"] > 0 or cv["direction_support"] > 0))
        agreement = votes + sum((base_flag, gated_flag, predicted_flag, gated_predicted_flag, confidence[edge] >= support_cut,
                                 cv["selected_count"] > 0, cv["direction_support"] > 0))
        reverse_agreement = reverse_votes + sum((reverse in primary, reverse in gated, reverse in view.base_predicted_edges,
                                                 reverse in view.citation_predicted_edges, confidence[reverse] >= support_cut,
                                                 reverse_cv["selected_count"] > 0, reverse_cv["direction_support"] > 0))
        reverse_pressure = clamp(dot(reverse_w, (reverse_votes, int(reverse in primary), int(reverse in gated), confidence[reverse],
                                                    reverse_cv["selected_count"], reverse_cv["direction_support"])))
        rows.append({
            "source": source, "target": target, "data_vote_count": votes, "reverse_data_vote_count": reverse_votes,
            "text_primary": base_flag, "text_gated": gated_flag, "text_predicted": predicted_flag,
            "text_gated_predicted": gated_predicted_flag, "text_confidence": number(confidence[edge]),
            "reverse_text_primary": reverse in primary, "reverse_text_gated": reverse in gated,
            "reverse_text_confidence": number(confidence[reverse]),
            "cross_view_selected_count": int(cv["selected_count"]), "cross_view_channel_count": int(cv["rule_count"]),
            "cross_view_direction_support": number(cv["direction_support"]),
            "source_family_count": family, "source_agreement_count": agreement,
            "direction_margin": number(agreement - reverse_agreement), "reverse_pressure": number(reverse_pressure),
            "text_confidence_support": number(clamp(dot(text_w, (confidence[edge], int(base_flag), int(gated_flag), int(predicted_flag), int(gated_predicted_flag))))),
            "cross_view_support": number(clamp(dot(cross_w, (cv["selected_count"], cv["rule_count"], cv["direction_support"])))),
            "source_pool_member": edge in members,
        })
    context = {
        "source_edge_counts": counts, "density_floor": floor, "density_ceiling": ceiling,
        "source_pool_conf_quantile": support_cut, "text_high_conf_quantile": high_cut,
        "membership_groups": {**{f"data_{item.source_id}": item.edges for item in sources},
                              "text_primary": primary, "text_gated": gated,
                              "high_conf_text_primary": high_primary, "cross_view_support": graph_supported},
    }
    return rows, context


def _platform_features(pool_rows, context, variables, source_count, params):
    decimals = integer(params, "decimal_places")
    number = lambda value: float(f"{value:.{decimals}f}")
    incoming, outgoing = dict.fromkeys(variables, 0), dict.fromkeys(variables, 0)
    for row in pool_rows:
        if row["source_pool_member"]:
            outgoing[row["source"]] += 1
            incoming[row["target"]] += 1
    tendencies = {v: outgoing[v] / (incoming[v] + outgoing[v]) if incoming[v] + outgoing[v] else 0.0 for v in variables}
    max_agreement = max(row["source_agreement_count"] for row in pool_rows) or 1
    evidence_w = vector(params, "evidence_weights", 10)
    perturbation_w = vector(params, "perturbation_weights", 6)
    weakness_w = vector(params, "weakness_weights", 6)
    corroboration_w = vector(params, "corroboration_weights", 7)
    preliminary = []
    for row in pool_rows:
        edge, reverse = (row["source"], row["target"]), (row["target"], row["source"])
        role = scalar(params, "role_bonus_weight") * (tendencies[edge[0]] - tendencies[edge[1]])
        agreement_fraction = row["source_agreement_count"] / max_agreement
        margin = clamp((row["direction_margin"] + scalar(params, "margin_shift")) / positive(params, "margin_scale"))
        signals = sum((row["text_confidence"] >= context["source_pool_conf_quantile"], row["text_gated"], row["text_primary"],
                       row["data_vote_count"] >= scalar(params, "signal_data_min"), row["cross_view_selected_count"] > 0,
                       row["cross_view_direction_support"] > 0, row["source_family_count"] >= scalar(params, "signal_family_min")))
        rescue = row["text_gated"] and row["text_confidence"] >= context["text_high_conf_quantile"] and (row["data_vote_count"] >= scalar(params, "signal_data_min") or row["cross_view_direction_support"] > 0)
        reject = row["reverse_text_gated"] and not row["text_gated"] and row["reverse_data_vote_count"] > row["data_vote_count"]
        weak = row["source_family_count"] <= scalar(params, "weak_family_max") and not rescue
        utility = dot(evidence_w[:-1], (row["text_confidence"], int(row["text_primary"]), int(row["text_gated"]), int(row["text_predicted"]),
                                      int(row["text_gated_predicted"]), row["data_vote_count"] / max(1, source_count), agreement_fraction,
                                      margin, row["cross_view_support"])) + role - evidence_w[-1] * row["reverse_pressure"]
        offset_utilities = []
        for _, group in sorted(context["membership_groups"].items()):
            drop = scalar(params, "membership_forward_offset") if edge in group else 0.0
            if reverse in group:
                drop -= scalar(params, "membership_reverse_offset")
            offset_utilities.append(utility - drop)
        minimum = min(offset_utilities) if offset_utilities else utility
        mean = sum(offset_utilities) / len(offset_utilities) if offset_utilities else utility
        deviation = population_std(offset_utilities)
        stability = minimum / utility if utility > 0 else 0.0
        perturbation = dot(perturbation_w, (utility, mean, minimum, clamp(stability), -deviation, -int(weak)))
        weakness = clamp(dot(weakness_w, (1 - agreement_fraction, row["reverse_pressure"], int(weak), int(reject),
                                         max(0.0, scalar(params, "weakness_support_target") - perturbation), 1 - clamp(stability))))
        addition = clamp(corroboration_w[0] * row["source_family_count"] / 3
                         + corroboration_w[1] * row["source_agreement_count"] / max_agreement
                         + corroboration_w[2] * row["text_confidence_support"]
                         + corroboration_w[3] * row["cross_view_support"]
                         + corroboration_w[4] * clamp(row["direction_margin"] / positive(params, "corroboration_margin_scale"))
                         + corroboration_w[5] * clamp(stability)
                         - corroboration_w[6] * row["reverse_pressure"])
        preliminary.append((row, utility, signals, rescue, weak, reject, stability, perturbation, weakness, addition))
    core_cut = percentile((item[7] for item in preliminary if item[7] > 0), quantile(params, "core_quantile"))
    by_edge = {(item[0]["source"], item[0]["target"]): item for item in preliminary}
    rows = []
    for row, utility, signals, rescue, weak, reject, stability, perturbation, weakness, addition in preliminary:
        reverse_perturbation = by_edge[(row["target"], row["source"])][7]
        core = row["source_pool_member"] and perturbation >= core_cut and signals >= scalar(params, "core_signal_min") and not reject and not weak and perturbation >= reverse_perturbation + scalar(params, "core_direction_margin")
        rows.append({**row, "support_signal_count": signals, "high_confidence_rescue": rescue,
                     "weak_single_source_dependence": weak, "hard_reverse_guard_reject": reject,
                     "integrated_utility": number(utility), "stability_ratio": number(stability),
                     "dropout_component": number(perturbation), "selected_weakness_score": number(weakness),
                     "unselected_addition_score": number(addition), "integrated_core": core})
    return rows


def _rank_directions(platform_rows, params):
    decimals = integer(params, "decimal_places")
    number = lambda value: float(f"{value:.{decimals}f}")
    core_stability_w = vector(params, "core_stability_weights", 4)
    core_conflict_w = vector(params, "core_conflict_weights", 3)
    residual_support_w = vector(params, "residual_support_weights", 5)
    residual_conflict_w = vector(params, "residual_conflict_weights", 3)
    rows = []
    for row in platform_rows:
        rank_score = max(row["integrated_utility"], row["dropout_component"], row["unselected_addition_score"])
        if row["integrated_core"]:
            stability = clamp(dot(core_stability_w, (row["integrated_utility"], row["dropout_component"], clamp(row["stability_ratio"]), 1 - row["selected_weakness_score"])))
            conflict = clamp(dot(core_conflict_w, (row["reverse_pressure"], row["selected_weakness_score"], int(row["hard_reverse_guard_reject"]))))
            correction = stability - conflict
        else:
            # Seven corroborating signals and three support categories are counts,
            # not tunable historical settings.
            support = clamp(dot(residual_support_w, (rank_score, min(1.0, row["support_signal_count"] / 7),
                                                    min(1.0, row["source_family_count"] / 3), row["text_confidence_support"], row["cross_view_support"])))
            conflict = clamp(dot(residual_conflict_w, (row["reverse_pressure"], int(row["hard_reverse_guard_reject"]), int(row["weak_single_source_dependence"]))))
            correction = support - conflict - scalar(params, "residual_offset")
        remove = row["data_vote_count"] <= scalar(params, "zero_vote_max") and row["text_confidence"] <= scalar(params, "zero_vote_text_max")
        rows.append({**row, "net_residual_gain": number(correction), "source_pool_rank_score": number(rank_score),
                     "residual_conflict_score": number(conflict), "final_score": number(rank_score + row["dropout_component"] + correction),
                     "precision_rule_remove": remove, "tie_break_key": f"{row['source']}->{row['target']}"})
    by_edge = {(row["source"], row["target"]): row for row in rows}
    for row in rows:
        reverse = by_edge[(row["target"], row["source"])]
        eligible = row["source_pool_member"] and not row["precision_rule_remove"] and row["net_residual_gain"] > scalar(params, "minimum_correction") and row["final_score"] > 0 and (row["support_signal_count"] >= scalar(params, "admission_signal_min") or row["high_confidence_rescue"]) and not row["hard_reverse_guard_reject"]
        margin = row["final_score"] - reverse["final_score"]
        tolerance = scalar(params, "pair_arbitration_margin")
        winner = eligible and (margin >= tolerance or (margin >= -tolerance and row["tie_break_key"] < reverse["tie_break_key"]))
        row.update(eligible=eligible, one_direction_pair_winner=winner)
    return rows


def _rank(row):
    return (row["final_score"], row["net_residual_gain"], row["source_pool_rank_score"], row["source_family_count"],
            -row["residual_conflict_score"], tuple(-ord(char) for char in row["source"]), tuple(-ord(char) for char in row["target"]))


def _aggregate(rows, context):
    by_pair = {}
    for row in rows:
        if row["eligible"] and row["one_direction_pair_winner"]:
            by_pair.setdefault(tuple(sorted((row["source"], row["target"]))), []).append(row)
    candidates = [sorted(items, key=_rank, reverse=True)[0] for items in by_pair.values()]
    selected, used_pairs = set(), set()
    for row in sorted(candidates, key=_rank, reverse=True):
        edge = (row["source"], row["target"])
        pair = tuple(sorted(edge))
        if pair in used_pairs or has_path(selected, edge[1], edge[0]) or len(selected) >= context["density_ceiling"]:
            continue
        selected.add(edge)
        used_pairs.add(pair)
    return frozenset(selected)


def assess_directional_evidence(
    variables: Sequence[str], statistical_graphs: Sequence[StatisticalGraph], directional_view: DirectionalView,
    cross_view_support: Mapping[Edge, Mapping[str, float]], parameters: Mapping,
) -> DirectionalAssessment:
    variables = validate_variables(variables)
    sources = validate_sources(statistical_graphs, variables)
    params = require_group(parameters, "directional")
    pool, context = _build_source_pool(variables, sources, directional_view, cross_view_support, params)
    platform = _platform_features(pool, context, variables, len(sources), params)
    ranked = _rank_directions(platform, params)
    return DirectionalAssessment({(row["source"], row["target"]): row for row in ranked}, context, _aggregate(ranked, context))
