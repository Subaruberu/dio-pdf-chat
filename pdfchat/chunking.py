"""Chunking: divide páginas em trechos com sobreposição, preservando parágrafos."""

from __future__ import annotations

import hashlib
import re

from .models import Page, Passage
from .textutils import split_sentences

_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _sentence_tail(text: str, limit: int) -> str:
    """Última parte do texto (até `limit` chars) começando numa fronteira de sentença."""
    if limit <= 0 or not text:
        return ""
    sentences = split_sentences(text)
    tail: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        if size + len(sentence) > limit:
            break
        tail.insert(0, sentence)
        size += len(sentence) + 1
    if tail:
        return " ".join(tail)
    return text[-limit:]


def _units(text: str, size: int) -> list[str]:
    """Parágrafos -> sentenças -> cortes duros, garantindo unidades <= size."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph, max_chars=size):
            if len(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(sentence[i : i + size].strip() for i in range(0, len(sentence), size))
    return [u for u in units if u.strip()]


def chunk_text(text: str, *, size: int = 900, overlap: int = 150) -> list[str]:
    """Agrupa unidades de texto em chunks de até `size` caracteres, com overlap."""
    units = _units(text, size)
    if not units:
        return []

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = _sentence_tail(current, overlap)
        else:
            tail = ""

        with_tail = f"{tail}\n{unit}".strip() if tail else unit
        current = with_tail if len(with_tail) <= size else unit

    if current.strip():
        chunks.append(current.strip())
    return chunks


def passage_id(source: str, page: int | None, chunk: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{chunk}|{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def chunk_pages(
    pages: list[Page], *, chunk_size: int = 900, chunk_overlap: int = 150
) -> list[Passage]:
    """Converte páginas extraídas em passagens indexáveis."""
    passages: list[Passage] = []
    for page in pages:
        for index, chunk in enumerate(chunk_text(page.text, size=chunk_size, overlap=chunk_overlap)):
            passages.append(
                Passage(
                    id=passage_id(page.source, page.page, index, chunk),
                    text=chunk,
                    source=page.source,
                    page=page.page,
                    chunk=index,
                )
            )
    return passages
