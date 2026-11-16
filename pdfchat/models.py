"""Estruturas de dados compartilhadas pelo pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Uma página (ou um arquivo .txt/.md inteiro) já limpa."""

    source: str
    page: int | None
    text: str


@dataclass(frozen=True)
class Passage:
    """Um trecho indexável, com metadados suficientes para citar a fonte."""

    id: str
    text: str
    source: str
    page: int | None
    chunk: int

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"{self.source}, p. {self.page}"
        return self.source

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk": self.chunk,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Passage":
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            page=data.get("page"),
            chunk=int(data.get("chunk", 0)),
        )


@dataclass
class Retrieved:
    """Trecho recuperado + pontuações para transparência do ranking."""

    passage: Passage
    score: float
    dense: float = 0.0
    lexical: float = 0.0
    rank: int = 0


@dataclass
class Answer:
    """Resposta final do chat."""

    question: str
    text: str
    sources: list[Retrieved] = field(default_factory=list)
    backend: str = "local"
    model: str = ""
    elapsed: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    warning: str | None = None
