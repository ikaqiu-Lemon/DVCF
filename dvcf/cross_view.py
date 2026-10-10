"""Four structural-support channels over statistical graphs and StructView."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .graph_utils import dot, has_path, prune_with_label_key
from .interfaces import Edge, StatisticalGraph, StructuralView, validate_edges, validate_sources, validate_variables
from .parameters import require_group, scalar, vector


CHANNELS = ("structural_reliability", "high_confidence_anchor", "conflict_veto_majority", "agreement_preservation")


def _pair_evidence(variables, sources, view):
    rows = []
    for i, source in enumerate(variables):
        for target in variables[i + 1:]:
            forward, reverse = (source, target), (target, source)
            fwd = [item for item in sources if forward in item.edges]
            rev = [item for item in sources if reverse in item.edges]
            rows.append({
                "forward": forward, "reverse": reverse,
                "data_forward": len(fwd), "data_reverse": len(rev),
                "data_pair": len({item.source_id for item in fwd + rev}),
                "family_forward": len({item.algorithm_family for item in fwd}),
                "family_reverse": len({item.algorithm_family for item in rev}),
                "family_pair": len({item.algorithm_family for item in fwd + rev}),
                "margin": abs(len(fwd) - len(rev)),
                "base_forward": int(forward in view.base_graph), "base_reverse": int(reverse in view.base_graph),
                "citation_forward": int(forward in view.citation_graph), "citation_reverse": int(reverse in view.citation_graph),
            })
    return rows


def _anchor(rows, variables, view, text_weight, params):
    majority = set()
    for row in rows:
        if row["data_pair"] >= scalar(params, "pair_support_min"):
            if row["data_forward"] > row["data_reverse"]:
                majority.add(row["forward"])
            elif row["data_reverse"] > row["data_forward"]:
                majority.add(row["reverse"])
    anchors = {
        "data": prune_with_label_key(majority),
        "text_primary": prune_with_label_key(view.base_graph),
        "text_evidence_gated": prune_with_label_key(view.citation_graph),
    }
    scores = []
    pair_count = max(len(variables) * max(len(variables) - 1, 1), 1)
    for name, edges in anchors.items():
        score = 0.0
        for row in rows:
            text_forward = row["base_forward"] + text_weight * row["citation_forward"]
            text_reverse = row["base_reverse"] + text_weight * row["citation_reverse"]
            if row["forward"] in edges:
                score += row["data_forward"] + row["family_forward"] + text_forward - text_reverse
            if row["reverse"] in edges:
                score += row["data_reverse"] + row["family_reverse"] + text_reverse - text_forward
        score -= abs(len(edges) / pair_count - scalar(params, "anchor_density_target"))
        scores.append((score, -len(edges), name, edges))
    return max(scores, key=lambda item: item[:3])[3]


def _select_channel(rows, variables, channel_index, text_weight, anchor, params):
    proposed = []
    reliability_w = vector(params, "reliability_weights", 5)
    anchor_w = vector(params, "anchor_weights", 5)
    majority_w = vector(params, "majority_weights", 4)
    preservation_w = vector(params, "preservation_weights", 6)
    for row in rows:
        d_f, d_r = row["data_forward"], row["data_reverse"]
        f_f, f_r = row["family_forward"], row["family_reverse"]
        t_f = row["base_forward"] + text_weight * row["citation_forward"]
        t_r = row["base_reverse"] + text_weight * row["citation_reverse"]
        veto_f = t_r >= scalar(params, "conflict_text_min") and d_f <= scalar(params, "conflict_data_max") and row["margin"] <= scalar(params, "conflict_margin_max")
        veto_r = t_f >= scalar(params, "conflict_text_min") and d_r <= scalar(params, "conflict_data_max") and row["margin"] <= scalar(params, "conflict_margin_max")
        if channel_index == 0:
            score_f = dot(reliability_w, (d_f, f_f, t_f, -t_r, row["margin"]))
            score_r = dot(reliability_w, (d_r, f_r, t_r, -t_f, row["margin"]))
            eligible_f = eligible_r = row["data_pair"] >= scalar(params, "pair_support_min")
        elif channel_index == 1:
            high_f = d_f >= scalar(params, "high_data_min") or (d_f >= scalar(params, "high_data_with_text_min") and t_f > 0) or f_f >= scalar(params, "high_family_min")
            high_r = d_r >= scalar(params, "high_data_min") or (d_r >= scalar(params, "high_data_with_text_min") and t_r > 0) or f_r >= scalar(params, "high_family_min")
            score_f = dot(anchor_w, (int(high_f), d_f, f_f, t_f, -t_r))
            score_r = dot(anchor_w, (int(high_r), d_r, f_r, t_r, -t_f))
            eligible_f, eligible_r = high_f, high_r
        elif channel_index == 2:
            score_f = dot(majority_w, (d_f, f_f, t_f, -t_r))
            score_r = dot(majority_w, (d_r, f_r, t_r, -t_f))
            eligible_f = d_f > d_r and row["data_pair"] >= scalar(params, "pair_support_min")
            eligible_r = d_r > d_f and row["data_pair"] >= scalar(params, "pair_support_min")
        else:
            anchor_f, anchor_r = row["forward"] in anchor, row["reverse"] in anchor
            add_f = (d_f >= scalar(params, "addition_data_with_text_min") and t_f > 0) or (d_f >= scalar(params, "addition_data_min") and f_f >= scalar(params, "addition_family_min"))
            add_r = (d_r >= scalar(params, "addition_data_with_text_min") and t_r > 0) or (d_r >= scalar(params, "addition_data_min") and f_r >= scalar(params, "addition_family_min"))
            score_f = dot(preservation_w, (int(anchor_f), int(add_f), d_f, f_f, t_f, -t_r))
            score_r = dot(preservation_w, (int(anchor_r), int(add_r), d_r, f_r, t_r, -t_f))
            eligible_f, eligible_r = anchor_f or add_f, anchor_r or add_r
        direction = None
        if eligible_f and not veto_f and (channel_index == 2 or score_f > score_r):
            direction, score = "forward", score_f
        elif eligible_r and not veto_r and (channel_index == 2 or score_r > score_f):
            direction, score = "reverse", score_r
        if direction is not None:
            proposed.append((row[direction], score, row["family_pair"], row["margin"]))
    ratio = vector(params, "channel_budget_ratios", len(CHANNELS))[channel_index]
    if ratio < 0:
        raise ValueError("channel budget ratios must be nonnegative")
    capacity = max(1, round(ratio * len(variables) * max(len(variables) - 1, 1)))
    selected = set()
    for edge, _, _, _ in sorted(proposed, key=lambda item: (-item[1], -item[2], -item[3], f"{item[0][0]}->{item[0][1]}")):
        if len(selected) >= capacity or has_path(selected, edge[1], edge[0]):
            continue
        selected.add(edge)
    return frozenset(selected)


def build_cross_view_evidence(
    variables: Sequence[str], statistical_graphs: Sequence[StatisticalGraph],
    structural_view: StructuralView, parameters: Mapping,
) -> dict[Edge, dict[str, float]]:
    """Return channel selections, participation counts, and directional support.

    Participation uses k_pair * k_direction per profile. Raw directional support
    is accumulated for every pair, including directions retained by no channel.
    """
    variables = validate_variables(variables)
    sources = validate_sources(statistical_graphs, variables)
    validate_edges(structural_view.base_graph, variables)
    validate_edges(structural_view.citation_graph, variables)
    params = require_group(parameters, "cross_view")
    rows = _pair_evidence(variables, sources, structural_view)
    support = {(a, b): {"selected_count": 0.0, "rule_count": 0.0, "direction_support": 0.0}
               for a in variables for b in variables if a != b}
    for text_weight in vector(params, "text_profile_weights", 2):
        anchor = _anchor(rows, variables, structural_view, text_weight, params)
        channels = [_select_channel(rows, variables, index, text_weight, anchor, params)
                    for index in range(len(CHANNELS))]
        for row in rows:
            forward_count = sum(row["forward"] in edges for edges in channels)
            reverse_count = sum(row["reverse"] in edges for edges in channels)
            pair_count = forward_count + reverse_count
            for direction, count in (("forward", forward_count), ("reverse", reverse_count)):
                item = support[row[direction]]
                item["selected_count"] += count
                item["rule_count"] += pair_count * count
                item["direction_support"] += row[f"data_{direction}"] + row[f"base_{direction}"] + row[f"citation_{direction}"]
    return support
