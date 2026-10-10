"""Evidence batching, injected summary generation, and deterministic summary merging."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

from .model_client import ModelClient, ModelParameters
from .response_parser import parse_json_object
from .retrieval import ScoredChunk


@dataclass(frozen=True)
class SummaryParameters:
    context_token_limit: int
    prompt_overhead_tokens: int
    variable_mention_boost: float
    mechanism_max_chars: int
    context_max_chars: int

    def __post_init__(self) -> None:
        if self.context_token_limit <= 0 or self.prompt_overhead_tokens < 0:
            raise ValueError("Supply a positive context limit and non-negative prompt overhead.")
        if not isfinite(self.variable_mention_boost) or self.variable_mention_boost < 0:
            raise ValueError("variable_mention_boost must be finite and non-negative.")
        if self.mechanism_max_chars <= 0 or self.context_max_chars <= 0:
            raise ValueError("Summary merge character limits must be positive.")


@dataclass(frozen=True)
class SummaryBatch:
    query: str
    variables: tuple[str, ...]
    evidence: tuple[ScoredChunk, ...]


def normalize_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not any(key in value for key in ("variables", "relationships", "domain_context")):
        raise ValueError("Summary response lacks the requested structured fields.")
    variables = value.get("variables", [])
    relationships = value.get("relationships", [])
    context = value.get("domain_context", "")
    if not isinstance(variables, list) or not all(isinstance(item, dict) for item in variables):
        raise ValueError("Summary variables must be a list of objects.")
    if not isinstance(relationships, list) or not all(isinstance(item, dict) for item in relationships):
        raise ValueError("Summary relationships must be a list of objects.")
    if isinstance(context, list):
        context = "\n".join(str(item) for item in context)
    return {"variables": variables, "relationships": relationships, "domain_context": str(context)}


def merge_summaries(
    summaries: Sequence[Mapping[str, Any]], *, mechanism_max_chars: int, context_max_chars: int
) -> dict[str, Any]:
    if mechanism_max_chars <= 0 or context_max_chars <= 0:
        raise ValueError("Summary merge limits must be positive.")
    variables: dict[str, dict[str, Any]] = {}
    relationships: dict[tuple[str, str, str], dict[str, Any]] = {}
    contexts: list[str] = []
    for raw in summaries:
        summary = normalize_summary(raw)
        for variable in summary["variables"]:
            name = str(variable.get("name", "")).lower()
            if not name:
                continue
            description = str(variable.get("description", ""))
            if name not in variables or len(description) > len(str(variables[name].get("description", ""))):
                variables[name] = dict(variable)
        for relationship in summary["relationships"]:
            key = (str(relationship.get("cause", "")).lower(),
                   str(relationship.get("effect", "")).lower(), str(relationship.get("type", "")))
            if key not in relationships:
                relationships[key] = dict(relationship)
            else:
                existing = str(relationships[key].get("mechanism", ""))
                incoming = str(relationship.get("mechanism", ""))
                combined = f"{existing}; {incoming}"
                if incoming and incoming != existing and len(combined) < mechanism_max_chars:
                    relationships[key]["mechanism"] = combined
        if summary["domain_context"]:
            contexts.append(summary["domain_context"])
    context = " ".join(contexts)
    if len(context) > context_max_chars:
        context = context[:context_max_chars] + "..."
    return {"variables": list(variables.values()), "relationships": list(relationships.values()), "domain_context": context}


def batch_evidence(
    evidence: Sequence[ScoredChunk], *, variables: Sequence[str], available_tokens: int,
    mention_boost: float, count_tokens: Callable[[str], int]
) -> tuple[tuple[ScoredChunk, ...], ...]:
    if available_tokens <= 0:
        raise ValueError("The supplied model budget leaves no room for evidence.")
    variable_terms = {name.lower() for name in variables}
    ordered = sorted(evidence, key=lambda hit: hit.score + mention_boost * sum(
        term in hit.chunk.text.lower() for term in variable_terms), reverse=True)
    batches: list[tuple[ScoredChunk, ...]] = []
    current: list[ScoredChunk] = []
    current_tokens = 0
    for item in ordered:
        text = item.chunk.text
        tokens = count_tokens(text)
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("The token counter must return a positive integer for non-empty evidence.")
        if current and current_tokens + tokens > available_tokens:
            batches.append(tuple(current))
            current, current_tokens = [], 0
        if tokens > available_tokens:
            # Source proportional truncation, followed by an exact budget check.
            while tokens > available_tokens:
                next_length = min(len(text) - 1, int(len(text) * available_tokens / tokens))
                if next_length <= 0:
                    raise ValueError("No evidence text fits the supplied token budget.")
                text = text[:next_length]
                tokens = count_tokens(text)
                if not isinstance(tokens, int) or tokens <= 0:
                    raise ValueError("Invalid token count while truncating evidence.")
            item = replace(item, chunk=replace(item.chunk, text=text))
        current.append(item)
        current_tokens += tokens
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def summarize_evidence(
    evidence: Sequence[ScoredChunk], *, variables: Sequence[str], query: str,
    client: ModelClient, model_parameters: ModelParameters, parameters: SummaryParameters,
    prompt_builder: Callable[[SummaryBatch], str], count_tokens: Callable[[str], int]
) -> dict[str, Any]:
    """Use supplied prompts; failed batches raise instead of creating empty summaries."""
    if not evidence:
        raise ValueError("Supply retrieved evidence before generating a summary.")
    available = parameters.context_token_limit - parameters.prompt_overhead_tokens - model_parameters.max_output_tokens
    batches = batch_evidence(evidence, variables=variables, available_tokens=available,
                             mention_boost=parameters.variable_mention_boost, count_tokens=count_tokens)
    summaries = []
    for batch in batches:
        prompt = prompt_builder(SummaryBatch(query, tuple(variables), batch))
        if count_tokens(prompt) + model_parameters.max_output_tokens > parameters.context_token_limit:
            raise ValueError("The supplied summary prompt exceeds the declared context budget.")
        response = client.complete(prompt, parameters=model_parameters)
        summaries.append(normalize_summary(parse_json_object(response)))
    return merge_summaries(summaries, mechanism_max_chars=parameters.mechanism_max_chars,
                           context_max_chars=parameters.context_max_chars)
