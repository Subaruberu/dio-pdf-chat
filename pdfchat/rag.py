"""Orquestração: ingestão de documentos, persistência e perguntas com citações."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder, build_embedder
from .llm import NO_CONTEXT_REPLY, ExtractiveLLM, build_llm
from .loaders import ExtractionError, collect_files, load_file
from .models import Answer, Passage, Retrieved
from .retriever import HybridRetriever
from .store import VectorStore


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def total_files(self) -> int:
        return len(self.added) + len(self.skipped)

    def summary(self) -> str:
        parts = [f"{len(self.added)} arquivo(s) indexado(s)"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} sem alterações")
        parts.append(f"{self.pages} página(s)")
        parts.append(f"{self.chunks} trecho(s)")
        return " · ".join(parts)


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _labels(paths: list[Path]) -> dict[Path, str]:
    """Evita colisão quando dois arquivos têm o mesmo nome em pastas diferentes."""
    counts: dict[str, int] = {}
    for path in paths:
        counts[path.name] = counts.get(path.name, 0) + 1
    labels: dict[Path, str] = {}
    for path in paths:
        labels[path] = path.name if counts[path.name] == 1 else f"{path.parent.name}/{path.name}"
    return labels


class ChatEngine:
    """Fachada única usada pelo Streamlit e pela CLI."""

    def __init__(self, settings: Settings, store: VectorStore | None = None):
        self.settings = settings
        self.store = store if store is not None else VectorStore()
        self.embedder: Embedder = build_embedder(settings)
        self._ensure_embedding_compat()
        self.retriever = HybridRetriever(
            self.store, self.embedder, mode=settings.retrieval, rrf_k=settings.rrf_k
        )
        self._llm_backend = settings.resolve_llm_backend()
        self.llm = build_llm(settings, self._llm_backend)

    # ------------------------------------------------------------------ load
    @classmethod
    def open(cls, settings: Settings | None = None) -> "ChatEngine":
        settings = settings or Settings.load()
        store = VectorStore.load(settings.index_dir)
        return cls(settings, store)

    def _ensure_embedding_compat(self) -> None:
        """Se o índice veio de outro modelo, recalcula os vetores (não re-lê os PDFs)."""
        if self.store.is_empty:
            return
        if self.store.embedder_name and self.store.embedder_name != self.embedder.name:
            previous = self.store.embedder_name
            self.store.reembed(self.embedder)
            self.store.save(self.settings.index_dir, embedder_name=self.embedder.name)
            self.settings.warnings.append(
                f"Embeddings recalculados: índice usava '{previous}', "
                f"agora usa '{self.embedder.name}'."
            )
        elif self.store.dim and self.embedder.dim and self.store.dim != self.embedder.dim:
            self.store.reembed(self.embedder)
            self.store.save(self.settings.index_dir, embedder_name=self.embedder.name)

    # ---------------------------------------------------------------- ingest
    def ingest(self, paths: list[str | Path], *, rebuild: bool = False) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()

        try:
            files = collect_files(paths)
        except (FileNotFoundError, ExtractionError) as exc:
            report.error = str(exc)
            return report
        if not files:
            report.error = "Nenhum arquivo .pdf, .txt ou .md encontrado nos caminhos informados."
            return report

        if rebuild:
            self.store.clear()
            self.store.manifest["files"] = {}
        self.store.manifest.setdefault("files", {})

        labels = _labels(files)
        new_passages: list[Passage] = []
        new_pages = 0

        for path in files:
            label = labels[path]
            try:
                digest = _sha1(path.read_bytes())
            except OSError as exc:
                report.warnings.append(f"{path.name}: não foi possível ler ({exc}).")
                continue

            previous = self.store.manifest["files"].get(label, {})
            if not rebuild and previous.get("sha1") == digest and self._source_exists(label):
                report.skipped.append(label)
                continue

            try:
                pages = load_file(path)
            except ExtractionError as exc:
                report.warnings.append(str(exc))
                continue
            if not pages:
                report.warnings.append(
                    f"{label}: nenhum texto extraído (PDF escaneado? aplique OCR antes de indexar)."
                )
                continue

            if self._source_exists(label):
                self.store.remove_sources({label})

            pages = [replace(page, source=label) for page in pages]
            passages = chunk_pages(
                pages,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            new_passages.extend(passages)
            new_pages += len(pages)
            self.store.manifest["files"][label] = {
                "sha1": digest,
                "pages": len(pages),
                "chunks": len(passages),
                "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            report.added.append(label)

        if new_passages:
            vectors = self.embedder.encode([p.text for p in new_passages])
            self.store.add(new_passages, vectors)
            self.store.embedder = self.embedder.name
            self.store.manifest["embedder"] = {
                "name": self.embedder.name,
                "dim": int(self.store.dim),
            }
            self.retriever.invalidate()

        self.store.save(self.settings.index_dir, embedder_name=self.embedder.name)
        report.pages = new_pages
        report.chunks = len(new_passages)
        report.elapsed = time.perf_counter() - started
        self.settings.warnings.extend(w for w in report.warnings if w not in self.settings.warnings)
        return report

    def _source_exists(self, label: str) -> bool:
        return any(p.source == label for p in self.store.passages)

    # ------------------------------------------------------------------- ask
    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(question="", text="Escreva uma pergunta.", backend="local", model="")

        if self.store.is_empty:
            return Answer(
                question=question,
                text=(
                    "Nenhum documento foi indexado ainda. Envie um PDF na barra lateral "
                    "(ou rode `python ingest.py inputs`) e faça a pergunta novamente."
                ),
                backend=self._llm_backend,
                model=self.settings.llm_model if self._llm_backend == "openai" else "extrativo-local",
            )

        sources = self.retriever.search(
            question,
            top_k=self.settings.top_k,
            candidate_pool=self.settings.candidate_pool,
        )
        if not sources:
            return Answer(
                question=question,
                text=NO_CONTEXT_REPLY,
                backend=self._llm_backend,
                model=self.settings.llm_model if self._llm_backend == "openai" else "extrativo-local",
            )

        trimmed_history = (history or [])[-self.settings.max_history_messages :]
        warning = None
        backend = self._llm_backend
        model = self.settings.llm_model if backend == "openai" else "extrativo-local"

        try:
            text, meta = self.llm.answer(question, sources, trimmed_history)
        except Exception as exc:  # noqa: BLE001 - resiliência: nunca derrubar o chat
            if not self.settings.allow_fallback:
                raise
            warning = f"Falha na API da OpenAI ({exc}). Resposta gerada em modo local."
            backend, model = "local", "extrativo-local"
            text, meta = ExtractiveLLM().answer(question, sources, trimmed_history)

        used = _used_citations(text, len(sources))
        if backend == "openai" and not used:
            warning = warning or "A resposta não citou fontes; confira os trechos recuperados abaixo."

        return Answer(
            question=question,
            text=text,
            sources=sources,
            backend=backend,
            model=model,
            elapsed=meta.get("elapsed", 0.0),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            warning=warning,
        )

    # ----------------------------------------------------------------- admin
    def remove_document(self, source: str) -> int:
        removed = self.store.remove_sources({source})
        self.store.save(self.settings.index_dir, embedder_name=self.embedder.name)
        self.retriever.invalidate()
        return removed

    def clear_index(self) -> None:
        self.store.clear()
        self.store.manifest = {"files": {}}
        self.store.save(self.settings.index_dir, embedder_name=self.embedder.name)

    def stats(self) -> dict:
        sources = self.store.sources()
        return {
            "documents": len(sources),
            "pages": sum(s["pages"] or 0 for s in sources.values()),
            "chunks": len(self.store),
            "embedder": self.embedder.name,
            "llm": "OpenAI · " + self.settings.llm_model if self._llm_backend == "openai" else "Local · extrativo",
            "retrieval": self.settings.retrieval,
            "updated_at": self.store.manifest.get("updated_at"),
        }


def _used_citations(text: str, total: int) -> set[int]:
    import re

    found = {int(match) for match in re.findall(r"\[(\d{1,2})\]", text)}
    return {index for index in found if 1 <= index <= total}
