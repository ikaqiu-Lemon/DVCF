"""Pair-local evidence preparation and explicitly configured directional judgments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

from dvcf.interfaces import Edge

from .model_client import ModelClient, ModelParameters
from .relation_context import RelationContext
from .response_parser import PairJudgment, parse_pair_judgments
from .retrieval import ScoredChunk


@dataclass(frozen=True)
class LabeledEvidence:
    label: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class PairPromptInput:
    pair: Edge
    descriptions: Mapping[str, str]
    evidence: tuple[LabeledEvidence, ...]
    summary: Mapping[str, Any]
    relation_context: RelationContext


def select_pair_evidence(
    pair: Edge, evidence: Sequence[ScoredChunk], *, aliases: Mapping[str, Sequence[str]], max_items: int
) -> tuple[ScoredChunk, ...]:
    """Prioritize chunks mentioning both variables, then either one, retaining order."""
    if max_items <= 0:
        raise ValueError("max_items must be positive.")
    left_terms = {pair[0].lower(), *(term.lower() for term in aliases.get(pair[0], ()) if term)}
    right_terms = {pair[1].lower(), *(term.lower() for term in aliases.get(pair[1], ()) if term)}
    both, single = [], []
    seen: set[str] = set()
    for item in evidence:
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        text = item.chunk.text.lower()
        has_left = any(term in text for term in left_terms)
        has_right = any(term in text for term in right_terms)
        if has_left and has_right:
            both.append(item)
        elif has_left or has_right:
            single.append(item)
    return tuple((both + single)[:max_items])


def judge_pairs(
    variables: Sequence[str], evidence: Sequence[ScoredChunk], *, descriptions: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]], summary: Mapping[str, Any], relation_context: RelationContext,
    client: ModelClient, model_parameters: ModelParameters, prompt_builder: Callable[[PairPromptInput], str],
    max_pair_evidence: int, evidence_preview_chars: int
) -> tuple[PairJudgment, ...]:
    """Generate one response per unordered pair using only its supplied context.

    A caller may provide a model transport that batches requests externally. This
    entry point neither fabricates scores nor adds an implicit knowledge channel.
    """
    if len(variables) < 2 or len(set(variables)) != len(variables) or any(not v for v in variables):
        raise ValueError("Provide at least two distinct variable names.")
    if max_pair_evidence <= 0 or evidence_preview_chars <= 0:
        raise ValueError("Evidence limits must be positive.")
    output: list[PairJudgment] = []
    for pair in combinations(variables, 2):
        selected = select_pair_evidence(pair, evidence, aliases=aliases, max_items=max_pair_evidence)
        labeled = tuple(LabeledEvidence(f"E{index + 1}", hit.chunk.chunk_id, hit.chunk.text[:evidence_preview_chars])
                        for index, hit in enumerate(selected))
        prompt_input = PairPromptInput(pair, {v: descriptions.get(v, "") for v in pair}, labeled, summary, relation_context)
        response = client.complete(prompt_builder(prompt_input), parameters=model_parameters)
        output.extend(parse_pair_judgments(response, expected_pairs=(pair,), aliases=aliases,
                                          evidence_map={item.label: item.chunk_id for item in labeled}))
    return tuple(output)


def directional_scores(judgments: Sequence[PairJudgment]) -> dict[Edge, float]:
    scores: dict[Edge, float] = {}
    for judgment in judgments:
        forward = (judgment.var_a, judgment.var_b)
        reverse = (judgment.var_b, judgment.var_a)
        if forward in scores or reverse in scores:
            raise ValueError("Each unordered variable pair must occur only once.")
        scores[forward] = judgment.score_a_to_b
        scores[reverse] = judgment.score_b_to_a
    return scores
