from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "as",
    "is",
    "are",
    "be",
    "by",
    "this",
    "that",
    "it",
    "from",
    "at",
    "into",
    "within",
    "after",
    "before",
    "if",
    "when",
    "any",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in _STOPWORDS and len(t) > 2]
    return toks


def _chunk_text(text: str, *, chunk_chars: int = 900, overlap: int = 150) -> list[str]:
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be > overlap")
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
        if j == n:
            break
    return chunks


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    source: str
    chunk: str


class LocalRagIndex:
    """Tiny offline retriever.

    This is intentionally simple for hackathon use:
    - Splits markdown files into character chunks
    - Computes a token overlap score between query and chunk

    No LLM calls. No guessing. Only retrieval + rules/templates in UI.
    """

    def __init__(self, chunks: list[tuple[str, str]]):
        # (source, chunk)
        self._chunks = chunks
        self._chunk_tokens = [set(_tokenize(c)) for _, c in chunks]

    @classmethod
    def from_markdown_dir(
        cls,
        kb_dir: Path,
        *,
        glob: str = "**/*.md",
        chunk_chars: int = 900,
        overlap: int = 150,
    ) -> "LocalRagIndex":
        kb_dir = Path(kb_dir)
        files = sorted(kb_dir.glob(glob))
        chunks: list[tuple[str, str]] = []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            for chunk in _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap):
                chunks.append((str(fp), chunk))
        return cls(chunks)

    def retrieve(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        qtokens = set(_tokenize(query))
        if not qtokens:
            return []

        scored: list[RetrievedChunk] = []
        for (src, chunk), ctoks in zip(self._chunks, self._chunk_tokens):
            inter = len(qtokens & ctoks)
            if inter == 0:
                continue
            # overlap normalized (cheap cosine-ish)
            score = inter / math.sqrt(len(qtokens) * max(1, len(ctoks)))
            scored.append(RetrievedChunk(score=score, source=src, chunk=chunk))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def build_query(
    *,
    pain_level: float,
    pain_class: str,
    patient_type: str,
    context: str,
    detected_features: Iterable[str],
    trend_note: str | None = None,
) -> str:
    feats = ", ".join([f for f in detected_features if f]) or "none"
    parts = [
        f"pain_level={pain_level:.1f}/10",
        f"pain_class={pain_class}",
        f"patient_type={patient_type}",
        f"context={context}",
        f"detected_features={feats}",
    ]
    if trend_note:
        parts.append(f"trend={trend_note}")
    return " | ".join(parts)
