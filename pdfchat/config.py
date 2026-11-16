"""Configuração central do projeto, carregada de variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = ROOT / "data" / "index"
DEFAULT_INPUTS_DIR = ROOT / "inputs"

_BACKENDS = {"auto", "openai", "local"}
_RETRIEVAL_MODES = {"hybrid", "dense", "lexical"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim", "on"}


@dataclass
class Settings:
    """Parâmetros do pipeline. Todos podem ser sobrescritos por variável de ambiente."""

    # Backends: "auto" usa OpenAI quando há chave; senão roda 100% local.
    backend: str = "auto"
    llm_backend: str = "auto"
    embed_backend: str = "auto"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.2

    # Recuperação
    retrieval: str = "hybrid"
    top_k: int = 5
    candidate_pool: int = 40
    rrf_k: int = 60

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Embedding local (hashing trick)
    embedding_dim: int = 1024

    # Limites de contexto
    max_context_chars: int = 7000
    max_history_messages: int = 6

    # Fallback resiliente: se a API falhar, responde em modo extrativo
    allow_fallback: bool = True

    # Caminhos
    index_dir: Path = DEFAULT_INDEX_DIR
    inputs_dir: Path = DEFAULT_INPUTS_DIR

    # Preenchido em tempo de execução
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "Settings":
        load_dotenv(dotenv_path=env_file or ROOT / ".env")

        raw_index = _env_str("PDFCHAT_INDEX_DIR", str(DEFAULT_INDEX_DIR))
        raw_inputs = _env_str("PDFCHAT_INPUTS_DIR", str(DEFAULT_INPUTS_DIR))

        settings = cls(
            backend=_env_str("PDFCHAT_BACKEND", "auto").lower(),
            llm_backend=_env_str("PDFCHAT_LLM_BACKEND", "auto").lower(),
            embed_backend=_env_str("PDFCHAT_EMBED_BACKEND", "auto").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            llm_model=_env_str("PDFCHAT_LLM_MODEL", "gpt-4o-mini"),
            embed_model=_env_str("PDFCHAT_EMBED_MODEL", "text-embedding-3-small"),
            temperature=_env_float("PDFCHAT_TEMPERATURE", 0.2),
            retrieval=_env_str("PDFCHAT_RETRIEVAL", "hybrid").lower(),
            top_k=_env_int("PDFCHAT_TOP_K", 5),
            candidate_pool=_env_int("PDFCHAT_CANDIDATE_POOL", 40),
            rrf_k=_env_int("PDFCHAT_RRF_K", 60),
            chunk_size=_env_int("PDFCHAT_CHUNK_SIZE", 900),
            chunk_overlap=_env_int("PDFCHAT_CHUNK_OVERLAP", 150),
            embedding_dim=_env_int("PDFCHAT_EMBEDDING_DIM", 1024),
            max_context_chars=_env_int("PDFCHAT_MAX_CONTEXT_CHARS", 7000),
            allow_fallback=_env_bool("PDFCHAT_FALLBACK", True),
            index_dir=Path(raw_index),
            inputs_dir=Path(raw_inputs),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.backend not in _BACKENDS:
            self.warnings.append(f"PDFCHAT_BACKEND inválido ({self.backend}); usando 'auto'.")
            self.backend = "auto"
        for name in ("llm_backend", "embed_backend"):
            value = getattr(self, name)
            if value not in _BACKENDS:
                self.warnings.append(f"{name} inválido ({value}); usando 'auto'.")
                setattr(self, name, "auto")
        if self.retrieval not in _RETRIEVAL_MODES:
            self.warnings.append(f"PDFCHAT_RETRIEVAL inválido ({self.retrieval}); usando 'hybrid'.")
            self.retrieval = "hybrid"
        self.top_k = max(1, min(self.top_k, 20))
        self.chunk_size = max(200, self.chunk_size)
        self.chunk_overlap = max(0, min(self.chunk_overlap, self.chunk_size // 2))
        self.embedding_dim = max(64, self.embedding_dim)

    # -------------------------------------------------------------- backends
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    def resolve_llm_backend(self) -> str:
        """'openai' ou 'local' (extrativo), nunca 'auto'."""
        if self.backend == "local":
            return "local"
        if self.llm_backend == "local":
            return "local"
        if self.llm_backend == "openai":
            if not self.has_openai_key:
                raise RuntimeError(
                    "PDFCHAT_LLM_BACKEND=openai exige OPENAI_API_KEY. "
                    "Defina a chave no .env ou use PDFCHAT_LLM_BACKEND=local."
                )
            return "openai"
        return "openai" if self.has_openai_key else "local"

    def resolve_embed_backend(self) -> str:
        if self.backend == "local":
            return "local"
        if self.embed_backend == "local":
            return "local"
        if self.embed_backend == "openai":
            if not self.has_openai_key:
                raise RuntimeError(
                    "PDFCHAT_EMBED_BACKEND=openai exige OPENAI_API_KEY. "
                    "Defina a chave no .env ou use PDFCHAT_EMBED_BACKEND=local."
                )
            return "openai"
        return "openai" if self.has_openai_key else "local"

    def describe(self) -> dict[str, str]:
        llm = self.resolve_llm_backend()
        embed = self.resolve_embed_backend()
        return {
            "llm": f"OpenAI · {self.llm_model}" if llm == "openai" else "Local · extrativo",
            "embeddings": (
                f"OpenAI · {self.embed_model}" if embed == "openai" else f"Local · hashing ({self.embedding_dim}d)"
            ),
            "retrieval": self.retrieval,
            "index_dir": str(self.index_dir),
        }
