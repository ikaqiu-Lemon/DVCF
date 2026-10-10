"""In-memory document splitting and stable evidence identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Sequence


@dataclass(frozen=True)
class Document:
    paper_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str


def recursive_character_splitter(
    *, chunk_size: int, chunk_overlap: int, separators: Sequence[str]
) -> Callable[[str], Sequence[str]]:
    """Create the source implementation's character splitter lazily.

    Supply all splitting settings; no corpus, file reader, or index is bundled.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0.")
    if not separators or separators[-1] != "":
        raise ValueError("Separators must end with an empty-string character fallback.")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
        length_function=len,
    )
    return splitter.split_text


def split_documents(
    documents: Sequence[Document], *, split_text: Callable[[str], Sequence[str]]
) -> tuple[Chunk, ...]:
    """Split caller-provided texts and assign paper/index/content-based IDs."""
    chunks: list[Chunk] = []
    seen_papers: set[str] = set()
    for document in documents:
        if not document.paper_id.strip() or document.paper_id in seen_papers:
            raise ValueError("Each document requires a distinct non-empty paper_id.")
        seen_papers.add(document.paper_id)
        if not document.text.strip():
            raise ValueError("Document text must be non-empty.")
        pieces = split_text(document.text)
        if isinstance(pieces, str):
            raise TypeError("split_text must return a sequence of text chunks.")
        for index, text in enumerate(pieces):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("The splitter returned an empty or invalid chunk.")
            text_hash = sha256(text.strip().encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{document.paper_id}::chunk_{index:04d}::{text_hash[:8]}"
            chunks.append(Chunk(chunk_id, document.paper_id, index, text))
    return tuple(chunks)
