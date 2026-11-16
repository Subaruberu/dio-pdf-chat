"""Fixtures compartilhadas: configuração 100% local e um corpus de teste."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfchat.config import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        backend="local",
        index_dir=tmp_path / "index",
        inputs_dir=tmp_path / "inputs",
        chunk_size=300,
        chunk_overlap=60,
        top_k=3,
        embedding_dim=256,
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "rag.txt").write_text(
        "Recuperação Aumentada por Geração combina busca vetorial com modelos de linguagem. "
        "Cada página do PDF é dividida em trechos de 900 caracteres com sobreposição de 150 caracteres. "
        "O índice usa similaridade de cosseno entre embeddings da pergunta e dos trechos.\n\n"
        "A citação obrigatória da fonte reduz alucinações. O sistema cita o arquivo e a página de cada trecho.",
        encoding="utf-8",
    )
    (folder / "bm25.txt").write_text(
        "O BM25 é uma função de ranking lexical usada em recuperação de informação. "
        "Ela considera frequência do termo e comprimento do documento, valorizando termos raros. "
        "A fusão por Reciprocal Rank Fusion combina BM25 e busca densa em um único ranking.",
        encoding="utf-8",
    )
    return folder
