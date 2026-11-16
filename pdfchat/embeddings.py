"""Embeddings: OpenAI quando há chave, ou hashing local (offline, sem download)."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod

import numpy as np

from .config import Settings
from .textutils import tokenize


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    """Contrato mínimo: um nome estável (para detectar incompatibilidade) e um vetor por texto."""

    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Retorna matriz float32 (n, dim) com linhas normalizadas."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder(Embedder):
    """Embedding local determinístico via hashing trick (word + bigramas + char n-grams).

    Não baixa modelos e não usa rede. Serve como fallback em qualquer máquina e,
    combinado com BM25, entrega recuperação sólida para perguntas factuais.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"local-hashing-{dim}d"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        for word in tokens:
            if len(word) >= 5:
                for i in range(len(word) - 3):
                    features.append((f"#{word[i : i + 4]}", 0.35))
        for first, second in zip(tokens, tokens[1:]):
            features.append((f"{first}_{second}", 0.7))
        return features

    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feature, weight in self._features(text):
                bucket, sign = self._bucket(feature)
                counts[bucket] = counts.get(bucket, 0.0) + sign * weight
            for bucket, value in counts.items():
                matrix[row, bucket] = np.sign(value) * np.log1p(abs(value))
        return _l2_normalize(matrix)


class OpenAIEmbedder(Embedder):
    """Embeddings da OpenAI (text-embedding-3-small por padrão)."""

    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependência opcional na prática
            raise RuntimeError("Instale o pacote 'openai' para usar este backend.") from exc

        self.model = settings.embed_model
        self.name = f"openai-{self.model}"
        self.dim = 0  # descoberto no primeiro encode
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._request(batch))
        matrix = np.array(vectors, dtype=np.float32)
        self.dim = int(matrix.shape[1])
        return _l2_normalize(matrix)

    def _request(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
                return [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001 - retry com backoff
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Falha ao gerar embeddings na OpenAI: {last_error}") from last_error


def build_embedder(settings: Settings, backend: str | None = None) -> Embedder:
    backend = backend or settings.resolve_embed_backend()
    if backend == "openai":
        return OpenAIEmbedder(settings)
    return HashingEmbedder(dim=settings.embedding_dim)
