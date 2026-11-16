"""Índice vetorial persistido em disco (numpy + JSON, sem banco externo)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import Passage

MANIFEST_FILE = "manifest.json"
PASSAGES_FILE = "passages.json"
VECTORS_FILE = "vectors.npy"


class EmbedderMismatch(RuntimeError):
    """O índice foi criado com outro modelo de embedding."""


@dataclass
class VectorStore:
    passages: list[Passage] = field(default_factory=list)
    vectors: np.ndarray | None = None
    manifest: dict = field(default_factory=dict)
    embedder: str = ""
    version: int = 0

    def __post_init__(self) -> None:
        if self.vectors is None:
            self.vectors = np.zeros((0, 0), dtype=np.float32)

    # ---------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.passages)

    @property
    def is_empty(self) -> bool:
        return not self.passages

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    @property
    def embedder_name(self) -> str:
        return self.embedder or str(self.manifest.get("embedder", {}).get("name", ""))

    def sources(self) -> dict[str, dict]:
        """Metadados por documento: números de páginas e de trechos."""
        info: dict[str, dict] = {}
        for passage in self.passages:
            entry = info.setdefault(
                passage.source, {"chunks": 0, "pages": set()}
            )
            entry["chunks"] += 1
            if passage.page is not None:
                entry["pages"].add(passage.page)
        return {
            source: {
                "chunks": data["chunks"],
                "pages": len(data["pages"]) if data["pages"] else None,
            }
            for source, data in sorted(info.items())
        }

    # --------------------------------------------------------------- mutation
    def add(self, passages: list[Passage], vectors: np.ndarray) -> None:
        if len(passages) != len(vectors):
            raise ValueError("Quantidade de passagens e vetores não confere.")
        if not passages:
            return
        if self.vectors.size and vectors.shape[1] != self.dim:
            raise ValueError(
                f"Dimensão dos vetores ({vectors.shape[1]}) difere do índice ({self.dim})."
            )
        existing = {p.id for p in self.passages}
        new_passages = [p for p, _ in zip(passages, vectors) if p.id not in existing]
        if len(new_passages) != len(passages):
            keep = np.array([i for i, p in enumerate(passages) if p.id not in existing], dtype=int)
            vectors = vectors[keep] if keep.size else np.zeros((0, vectors.shape[1]), dtype=np.float32)

        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)]) if self.vectors.size else vectors.astype(np.float32)
        self.passages.extend(new_passages)
        self.version += 1

    def remove_sources(self, sources: set[str] | list[str]) -> int:
        sources = set(sources)
        if not sources:
            return 0
        keep = np.array([i for i, p in enumerate(self.passages) if p.source not in sources], dtype=int)
        removed = len(self.passages) - len(keep)
        self.passages = [self.passages[i] for i in keep]
        self.vectors = self.vectors[keep] if keep.size else np.zeros((0, self.dim), dtype=np.float32)
        for source in sources:
            self.manifest.get("files", {}).pop(source, None)
        self.version += 1
        return removed

    def clear(self) -> None:
        self.manifest = {}
        self.passages = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.version += 1

    def reembed(self, embedder) -> None:
        """Recalcula todos os vetores quando o backend de embedding muda."""
        self.embedder = embedder.name
        if not self.passages:
            self.manifest["embedder"] = {"name": embedder.name, "dim": 0}
            return
        self.vectors = embedder.encode([p.text for p in self.passages])
        self.manifest["embedder"] = {"name": embedder.name, "dim": int(self.vectors.shape[1])}
        self.version += 1

    # --------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, top_k: int = 40) -> list[tuple[int, float]]:
        if self.is_empty or not query_vector.size:
            return []
        scores = self.vectors @ query_vector.astype(np.float32)
        limit = min(top_k, len(scores))
        best = np.argpartition(-scores, limit - 1)[:limit]
        best = best[np.argsort(-scores[best])]
        return [(int(i), float(scores[i])) for i in best]

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path, *, embedder_name: str | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if embedder_name:
            self.embedder = embedder_name
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.manifest.setdefault("created_at", now)
        self.manifest["updated_at"] = now
        self.manifest["passages"] = len(self.passages)
        self.manifest["embedder"] = {
            "name": self.embedder_name,
            "dim": self.dim,
        }
        _atomic_write(directory / PASSAGES_FILE, json.dumps([p.to_dict() for p in self.passages], ensure_ascii=False, indent=2))
        np.save(directory / VECTORS_FILE, self.vectors)
        _atomic_write(directory / MANIFEST_FILE, json.dumps(self.manifest, ensure_ascii=False, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            return cls()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        passages_path = directory / PASSAGES_FILE
        passages = []
        if passages_path.exists():
            passages = [Passage.from_dict(d) for d in json.loads(passages_path.read_text(encoding="utf-8"))]
        vectors_path = directory / VECTORS_FILE
        vectors = np.load(vectors_path).astype(np.float32) if vectors_path.exists() else np.zeros((len(passages), 0), dtype=np.float32)
        embedder = str(manifest.get("embedder", {}).get("name", ""))
        return cls(passages=passages, vectors=vectors, manifest=manifest, embedder=embedder)

    def assert_compatible(self, embedder_name: str, dim: int) -> None:
        if self.is_empty:
            return
        if self.embedder_name and self.embedder_name != embedder_name:
            raise EmbedderMismatch(
                f"O índice foi criado com '{self.embedder_name}', mas o backend atual é "
                f"'{embedder_name}'. Reindexe os documentos (ingest --rebuild)."
            )
        if dim and self.dim and dim != self.dim:
            raise EmbedderMismatch(
                f"Dimensão do índice ({self.dim}) difere do embedder atual ({dim}). "
                "Reindexe os documentos (ingest --rebuild)."
            )

    def delete_disk(self, directory: str | Path) -> None:
        directory = Path(directory)
        if directory.exists():
            shutil.rmtree(directory)


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
