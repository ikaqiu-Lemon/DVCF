"""Pattern matches and nearest example selection from caller-supplied material."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Sequence

from .retrieval import Embedder, cosine_similarity


@dataclass(frozen=True)
class RelationExample:
    sentence: str
    cause: str
    effect: str


@dataclass(frozen=True)
class RelationContext:
    patterns: tuple[str, ...]
    examples: tuple[RelationExample, ...]


def partial_ratio(pattern: str, text: str) -> float:
    """Lazy adapter for the source implementation's fuzzy partial-string match."""
    from fuzzywuzzy import fuzz

    return float(fuzz.partial_ratio(pattern, text))


def enhance_relation_context(
    text: str, *, patterns: Sequence[str], examples: Sequence[RelationExample],
    embedder: Embedder, pattern_similarity: Callable[[str, str], float],
    pattern_threshold: float, example_top_k: int
) -> RelationContext:
    if not isfinite(pattern_threshold) or not 0 <= pattern_threshold <= 100:
        raise ValueError("pattern_threshold must be between zero and one hundred.")
    if example_top_k < 0:
        raise ValueError("example_top_k must be non-negative.")
    matches = tuple(pattern for pattern in patterns if pattern_similarity(pattern, text) > pattern_threshold)
    if not examples or example_top_k == 0:
        return RelationContext(matches, ())
    vectors = tuple(tuple(float(x) for x in row) for row in embedder([e.sentence for e in examples]))
    query_vectors = tuple(tuple(float(x) for x in row) for row in embedder([text]))
    if len(vectors) != len(examples) or len(query_vectors) != 1:
        raise ValueError("The embedder must return one vector per supplied text.")
    ranked = sorted(
        ((cosine_similarity(query_vectors[0], vector), index) for index, vector in enumerate(vectors)),
        reverse=True,
    )[:example_top_k]
    return RelationContext(matches, tuple(examples[index] for _, index in ranked))
