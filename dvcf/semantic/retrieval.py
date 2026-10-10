"""Cosine retrieval, BM25, reciprocal-rank fusion, and injected reranking."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite, sqrt
from typing import Callable, Mapping, Sequence

from .corpus import Chunk
from .model_client import require_filled

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
Reranker = Callable[[str, Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) == 0 or len(left) != len(right):
        raise ValueError("Embedding vectors must be non-empty and have matching dimensions.")
    if not all(isfinite(float(x)) for x in (*left, *right)):
        raise ValueError("Embedding vectors must be finite.")
    denominator = sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    if denominator == 0:
        raise ValueError("Cosine retrieval requires non-zero embedding vectors.")
    return sum(a * b for a, b in zip(left, right)) / denominator


class BM25Retriever:
    """In-memory adapter for the same BM25s implementation used by the source."""

    def __init__(
        self, chunks: Sequence[Chunk], *, method: str, k1: float, b: float,
        delta: float, stopwords: str | Sequence[str] | None
    ) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk.")
        require_filled(method, "BM25 method")
        if not all(isfinite(x) for x in (k1, b, delta)) or k1 <= 0 or not 0 <= b <= 1 or delta < 0:
            raise ValueError("Invalid BM25 parameters.")
        import bm25s

        self._chunks = tuple(chunks)
        self._stopwords = stopwords
        self._tokenize = bm25s.tokenize
        self._index = bm25s.BM25(method=method, k1=k1, b=b, delta=delta)
        tokens = self._tokenize([chunk.text for chunk in chunks], stopwords=stopwords)
        self._index.index(tokens, show_progress=False)

    def __call__(self, query: str, top_k: int) -> tuple[ScoredChunk, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        tokens = self._tokenize([query], stopwords=self._stopwords)
        indices, scores = self._index.retrieve(
            tokens, k=min(top_k, len(self._chunks)), show_progress=False
        )
        return tuple(
            ScoredChunk(self._chunks[int(index)], float(score))
            for index, score in zip(indices[0], scores[0]) if score > 0
        )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]], *, rank_constant: float
) -> tuple[ScoredChunk, ...]:
    """Sum 1 / (rank_constant + rank + 1), retaining stable input-order ties."""
    if not isfinite(rank_constant) or rank_constant < 0:
        raise ValueError("rank_constant must be finite and non-negative.")
    scores: dict[str, float] = {}
    items: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, item in enumerate(ranked):
            identity = item.chunk.chunk_id
            if identity in seen:
                raise ValueError("Each ranking must contain distinct chunk IDs.")
            seen.add(identity)
            if identity in items and items[identity] != item.chunk:
                raise ValueError("The same chunk ID refers to inconsistent content.")
            items.setdefault(identity, item.chunk)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rank_constant + rank + 1)
    return tuple(ScoredChunk(items[key], scores[key]) for key in sorted(scores, key=scores.get, reverse=True))


class HybridRetriever:
    def __init__(
        self, chunks: Sequence[Chunk], *, embedder: Embedder,
        sparse_search: Callable[[str, int], Sequence[ScoredChunk]], reranker: Reranker,
        retrieval_top_k: int, rerank_top_k: int, rank_constant: float,
        expand_queries: Callable[[str], Sequence[str]], max_rerank_candidates: int
    ) -> None:
        if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Provide non-empty chunks with distinct IDs.")
        if retrieval_top_k <= 0 or rerank_top_k <= 0 or max_rerank_candidates <= 0:
            raise ValueError("Retrieval and rerank limits must be positive.")
        if not isfinite(rank_constant) or rank_constant < 0:
            raise ValueError("rank_constant must be finite and non-negative.")
        self._chunks = tuple(chunks)
        self._chunk_lookup = {c.chunk_id: c for c in chunks}
        self._embedder = embedder
        self._sparse_search = sparse_search
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._rank_constant = rank_constant
        self._expand_queries = expand_queries
        self._max_rerank_candidates = max_rerank_candidates
        self._vectors = tuple(tuple(float(v) for v in row) for row in embedder([c.text for c in chunks]))
        if len(self._vectors) != len(chunks):
            raise ValueError("The embedder must return one vector per chunk.")
        for vector in self._vectors:
            cosine_similarity(vector, self._vectors[0])

    def _hybrid_retrieve(self, query: str) -> tuple[ScoredChunk, ...]:
        require_filled(query, "retrieval query")
        vectors = tuple(tuple(float(v) for v in row) for row in self._embedder([query]))
        if len(vectors) != 1:
            raise ValueError("The embedder must return one vector for the query.")
        dense = sorted(
            (ScoredChunk(chunk, cosine_similarity(vectors[0], vector))
             for chunk, vector in zip(self._chunks, self._vectors)),
            key=lambda hit: hit.score, reverse=True,
        )[:self._retrieval_top_k]
        sparse = tuple(self._sparse_search(query, self._retrieval_top_k))
        if any(self._chunk_lookup.get(hit.chunk.chunk_id) != hit.chunk for hit in sparse):
            raise ValueError("Sparse results must refer to the supplied corpus.")
        return reciprocal_rank_fusion((dense, sparse), rank_constant=self._rank_constant)[:self._retrieval_top_k]

    def retrieve(self, query: str) -> tuple[ScoredChunk, ...]:
        """Expand → hybrid retrieval → maximum-score deduplication → rerank.

        The required expansion callback returns the complete query sequence, including
        the original query when desired. Supply ``lambda query: (query,)`` explicitly
        to disable expansion; no model prompt or expansion count is embedded here.
        """
        require_filled(query, "retrieval query")
        expansions = self._expand_queries(query)
        if isinstance(expansions, str):
            raise ValueError("Query expansion must return a sequence, not a string.")
        expansions = tuple(expansions)
        if not expansions:
            raise ValueError("Query expansion must return a non-empty sequence of queries.")
        collected: dict[str, ScoredChunk] = {}
        for expanded_query in expansions:
            require_filled(expanded_query, "expanded query")
            for hit in self._hybrid_retrieve(expanded_query):
                identity = hit.chunk.chunk_id
                if identity not in collected or hit.score > collected[identity].score:
                    collected[identity] = hit
        candidates = sorted(collected.values(), key=lambda hit: hit.score, reverse=True)[:self._max_rerank_candidates]
        scores = tuple(float(v) for v in self._reranker(query, [hit.chunk.text for hit in candidates]))
        if len(scores) != len(candidates) or not all(isfinite(v) for v in scores):
            raise ValueError("The reranker must return one finite score per candidate.")
        reranked = [ScoredChunk(hit.chunk, score) for hit, score in zip(candidates, scores)]
        return tuple(sorted(reranked, key=lambda hit: hit.score, reverse=True)[:self._rerank_top_k])


@dataclass(frozen=True)
class VariableMetadata:
    name: str
    description: str
    full_name: str
    synonyms: tuple[str, ...]


def metadata_queries(
    variables: Sequence[VariableMetadata], *, variable_templates: Sequence[str],
    synonym_templates: Sequence[str], pair_templates: Sequence[str], pair_variable_limit: int
) -> tuple[str, ...]:
    """Instantiate caller-supplied templates using variable descriptions and synonyms.

    Variable fields: {name}, {description}, {full_name}, {term}.
    Pair fields: {name_a}, {name_b}, {description_a}, {description_b}.
    """
    if not variables or len({v.name for v in variables}) != len(variables):
        raise ValueError("Provide distinct variable metadata.")
    if pair_variable_limit < 0 or not variable_templates:
        raise ValueError("Supply variable templates and a non-negative pair limit.")
    queries: list[str] = []
    for template in (*variable_templates, *synonym_templates, *pair_templates):
        require_filled(template, "query template")
    for variable in variables:
        require_filled(variable.name, "variable name")
        description = variable.description or variable.full_name or variable.name
        fields = dict(name=variable.name, description=description, full_name=variable.full_name, term=description)
        queries.extend(template.format(**fields) for template in variable_templates)
        for synonym in variable.synonyms:
            if synonym.lower() not in {variable.name.lower(), description.lower()}:
                queries.extend(template.format(**{**fields, "term": synonym}) for template in synonym_templates)
    if len(variables) <= pair_variable_limit:
        for left, right in combinations(variables, 2):
            fields = dict(name_a=left.name, name_b=right.name,
                          description_a=left.description or left.full_name or left.name,
                          description_b=right.description or right.full_name or right.name)
            queries.extend(template.format(**fields) for template in pair_templates)
    return tuple(dict.fromkeys(queries))


def retrieve_evidence(retriever: HybridRetriever, queries: Sequence[str]) -> tuple[ScoredChunk, ...]:
    """Accumulate distinct chunks in query order, retaining their first retrieved score."""
    evidence: dict[str, ScoredChunk] = {}
    for query in queries:
        for item in retriever.retrieve(query):
            evidence.setdefault(item.chunk.chunk_id, item)
    return tuple(evidence.values())
