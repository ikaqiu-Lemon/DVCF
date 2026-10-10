"""Citation-to-context filtering with an explicit unknown-label policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from dvcf.interfaces import Edge

from .response_parser import PairJudgment, clean_evidence_label

CitationPolicy = Literal["reject_unknown", "valid_only"]


@dataclass(frozen=True)
class CitationDecision:
    raw_labels: tuple[str, ...]
    valid_labels: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    retained: bool


@dataclass(frozen=True)
class CitationFilterResult:
    edges: frozenset[Edge]
    decisions: Mapping[Edge, CitationDecision]


def filter_citations(
    edges: frozenset[Edge], judgments: Sequence[PairJudgment], *, policy: CitationPolicy
) -> CitationFilterResult:
    """Keep valid cited edges; the caller decides how unknown labels affect them.

    reject_unknown: at least one valid label and no unresolved labels.
    valid_only: at least one valid label; unresolved labels remain in the decision.
    This checks label correspondence, not semantic entailment of a direction.
    """
    if policy not in ("reject_unknown", "valid_only"):
        raise ValueError("Choose citation policy 'reject_unknown' or 'valid_only'.")
    pairs = {frozenset((item.var_a, item.var_b)): item for item in judgments}
    if len(pairs) != len(judgments):
        raise ValueError("Judgments must contain each unordered pair only once.")
    retained: set[Edge] = set()
    decisions: dict[Edge, CitationDecision] = {}
    for edge in sorted(edges):
        if edge[0] == edge[1]:
            raise ValueError("Self-loops are not citation-filterable causal directions.")
        judgment = pairs.get(frozenset(edge))
        if judgment is None:
            raise ValueError("A selected direction has no pair judgment.")
        cleaned = tuple(clean_evidence_label(label) for label in judgment.raw_labels)
        valid = tuple(label for label in cleaned if judgment.evidence_map.get(label))
        unknown = tuple(label for label in cleaned if not judgment.evidence_map.get(label))
        if valid != judgment.valid_labels or unknown != judgment.unknown_labels:
            raise ValueError("Citation records are inconsistent with raw labels and evidence mappings.")
        keep = bool(valid) and (policy == "valid_only" or not unknown)
        if keep:
            retained.add(edge)
        decisions[edge] = CitationDecision(judgment.raw_labels, valid, unknown, keep)
    return CitationFilterResult(frozenset(retained), decisions)
