"""BM25: busca lexical que complementa os embeddings (nomes, siglas, números)."""

from __future__ import annotations

import math

import numpy as np

from .textutils import tokenize


class BM25:
    """Implementação clássica de BM25 (k1, b) sobre uma lista de documentos."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens: list[list[str]] = [tokenize(doc) for doc in documents]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0

        self.term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for tokens in self.tokens:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            for token in tf:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        total = max(len(self.tokens), 1)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        result = np.zeros(len(self.tokens), dtype=np.float32)
        if not query_terms or not self.tokens:
            return result

        avg_len = self.avg_len or 1.0
        for term in set(query_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in enumerate(self.term_freqs):
                freq = tf.get(term)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / avg_len)
                result[index] += idf * (freq * (self.k1 + 1)) / (freq + norm)
        return result
