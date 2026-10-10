"""Construct separate directional and structural products from supplied judgments."""

from __future__ import annotations

from typing import Sequence

from dvcf.interfaces import DirectionalView, StructuralView

from .citation_filter import CitationFilterResult, CitationPolicy, filter_citations
from .graph_construction import GraphParameters, construct_graph
from .judgments import directional_scores
from .response_parser import PairJudgment


def prepare_directional_view(
    variables: Sequence[str], judgments: Sequence[PairJudgment], *,
    graph_parameters: GraphParameters, citation_policy: CitationPolicy
) -> tuple[DirectionalView, CitationFilterResult]:
    scores = directional_scores(judgments)
    base = construct_graph(variables, scores, parameters=graph_parameters)
    citations = filter_citations(base, judgments, policy=citation_policy)
    # The generator converts each constructed graph to its corresponding edge list.
    # Preserve all four membership fields even when the generated sets coincide.
    view = DirectionalView(scores=scores, base_adjacency=base, citation_adjacency=citations.edges,
                           base_predicted_edges=frozenset(base), citation_predicted_edges=frozenset(citations.edges))
    return view, citations


def prepare_structural_view(
    variables: Sequence[str], judgments: Sequence[PairJudgment], *,
    graph_parameters: GraphParameters, citation_policy: CitationPolicy
) -> tuple[StructuralView, CitationFilterResult]:
    """Consume the structural product's own judgments; never reuse a DirectView implicitly."""
    scores = directional_scores(judgments)
    base = construct_graph(variables, scores, parameters=graph_parameters)
    citations = filter_citations(base, judgments, policy=citation_policy)
    return StructuralView(base_graph=base, citation_graph=citations.edges), citations


def prepare_views(
    variables: Sequence[str], *, direct_judgments: Sequence[PairJudgment],
    structural_judgments: Sequence[PairJudgment], direct_graph_parameters: GraphParameters,
    structural_graph_parameters: GraphParameters, direct_citation_policy: CitationPolicy,
    structural_citation_policy: CitationPolicy
) -> tuple[DirectionalView, StructuralView]:
    """Both independently prepared inputs are required, with no copy/alias fallback."""
    direct, _ = prepare_directional_view(variables, direct_judgments,
                                         graph_parameters=direct_graph_parameters,
                                         citation_policy=direct_citation_policy)
    structural, _ = prepare_structural_view(variables, structural_judgments,
                                           graph_parameters=structural_graph_parameters,
                                           citation_policy=structural_citation_policy)
    return direct, structural
