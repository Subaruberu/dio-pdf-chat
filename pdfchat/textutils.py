"""Utilidades de texto: normalização, tokenização e segmentação em sentenças."""

from __future__ import annotations

import re
import unicodedata

# Stopwords do português (lista enxuta, focada em busca de conteúdo).
STOPWORDS: frozenset[str] = frozenset(
    """
    a ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela delas
    dele deles depois do dos e ela elas ele eles em entre era eram essa essas esse esses
    esta estas este estes eu foi fomos for foram fosse fui há isso isto já lhe lhes mais
    mas me mesmo meu meus minha minhas muito na nas nem no nos nós o os ou para pela
    pelas pelo pelos por qual quando que quem se sem seu seus sua suas são só também te
    tem temos tenho ter teu teus tua tuas um uma umas uns você vocês vos à às
    the a an and are as at be by for from has have in is it its of on or that to was were
    will with this these those
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9\"'(\[])")


def fold(text: str) -> str:
    """Remove acentos: 'informação' -> 'informacao'."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokeniza em minúsculas, sem acento, descartando tokens de 1 caractere."""
    tokens = _TOKEN_RE.findall(fold(text.lower()))
    tokens = [t for t in tokens if len(t) > 1]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str, *, max_chars: int = 1200) -> list[str]:
    """Segmenta em sentenças; sentenças gigantes são cortadas em pedaços."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    result: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            result.append(part)
            continue
        # Quebra por pontuação mais fraca, depois por tamanho.
        pieces = [p.strip() for p in re.split(r"(?<=[;:])\s+", part) if p.strip()]
        for piece in pieces:
            while len(piece) > max_chars:
                cut = piece.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                result.append(piece[:cut].strip())
                piece = piece[cut:].strip()
            if piece:
                result.append(piece)
    return result


def shorten(text: str, limit: int = 240) -> str:
    """Resumo de uma linha, com reticências."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip(" ,.;:") + "…"
