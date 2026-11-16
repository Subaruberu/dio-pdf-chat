"""Recuperação híbrida: similaridade vetorial + BM25 fundidos por Reciprocal Rank Fusion."""

from __future__ import annotations

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder
from .models import Retrieved
from .store import VectorStore


class HybridRetriever:
    """Busca densa (cosseno) + lexical (BM25) com fusão por rank."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        mode: str = "hybrid",
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.mode = mode if mode in {"hybrid", "dense", "lexical"} else "hybrid"
        self.rrf_k = rrf_k
        self._bm25: BM25 | None = None
        self._bm25_version = -1
        self._bm25_size = -1

    # ------------------------------------------------------------------ BM25
    @property
    def bm25(self) -> BM25 | None:
        if self.store.is_empty:
            return None
        if self._bm25 is None or self._bm25_version != self.store.version or self._bm25_size != len(self.store):
            self._bm25 = BM25([p.text for p in self.store.passages])
            self._bm25_version = self.store.version
            self._bm25_size = len(self.store)
        return self._bm25

    def invalidate(self) -> None:
        self._bm25 = None
        self._bm25_version = -1

    # ---------------------------------------------------------------- search
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_pool: int = 40,
        sources: list[str] | None = None,
    ) -> list[Retrieved]:
        if self.store.is_empty or not query.strip():
            return []

        pool = max(top_k * 4, candidate_pool)
        dense_scores = np.zeros(len(self.store), dtype=np.float32)
        lexical_scores = np.zeros(len(self.store), dtype=np.float32)
        dense_ranks: dict[int, int] = {}
        lexical_ranks: dict[int, int] = {}

        if self.mode in {"hybrid", "dense"}:
            vector = self.embedder.encode_one(query)
            if vector.size and vector.shape[0] == self.store.dim:
                for rank, (index, score) in enumerate(self.store.search(vector, pool), start=1):
                    dense_scores[index] = score
                    dense_ranks[index] = rank

        if self.mode in {"hybrid", "lexical"} and self.bm25 is not None:
            scores = self.bm25.scores(query)
            order = np.argsort(-scores)[:pool]
            for rank, index in enumerate(order, start=1):
                if scores[index] <= 0:
                    continue
                lexical_scores[index] = float(scores[index])
                lexical_ranks[index] = rank

        candidates = set(dense_ranks) | set(lexical_ranks)
        if not candidates:
            return []

        fused: dict[int, float] = {}
        for index in candidates:
            score = 0.0
            if index in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[index])
            if index in lexical_ranks:
                score += 1.0 / (self.rrf_k + lexical_ranks[index])
            fused[index] = score

        best = max(fused.values()) or 1.0
        results: list[Retrieved] = []
        for index, raw in sorted(fused.items(), key=lambda item: -item[1]):
            passage = self.store.passages[index]
            if sources and passage.source not in sources:
                continue
            if self.mode == "dense":
                display = float(max(0.0, dense_scores[index]))
            elif self.mode == "lexical":
                display = float(lexical_scores[index] / (lexical_scores.max() or 1.0))
            else:
                display = float(raw / best)
            results.append(
                Retrieved(
                    passage=passage,
                    score=display,
                    dense=float(dense_scores[index]),
                    lexical=float(lexical_scores[index]),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break
        return results
