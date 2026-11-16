"""Testes de limpeza de texto e chunking."""

from __future__ import annotations

from pdfchat.chunking import chunk_pages, chunk_text
from pdfchat.loaders import clean_text
from pdfchat.models import Page
from pdfchat.textutils import shorten, split_sentences, tokenize


def test_clean_text_remove_hifenizacao_e_espacos():
    raw = "A engenha-\nria de software   é  multidisciplinar.\n\n\n\nNovo par\u00e1grafo."
    assert clean_text(raw) == "A engenharia de software é multidisciplinar.\n\nNovo parágrafo."


def test_tokenize_sem_acento_e_stopwords():
    assert tokenize("A Recuperação Aumentada por Geração!") == ["recuperacao", "aumentada", "geracao"]


def test_split_sentences_respeita_pontuacao():
    parts = split_sentences("Primeira frase. Segunda frase? Terceira frase!")
    assert parts == ["Primeira frase.", "Segunda frase?", "Terceira frase!"]


def test_chunk_text_respeita_tamanho_e_sobreposicao():
    texto = " ".join(f"Esta é a sentença número {i} sobre engenharia de software." for i in range(40))
    chunks = chunk_text(texto, size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert any("sentença número" in chunk for chunk in chunks)


def test_chunk_pages_preserva_fonte_e_pagina():
    pages = [Page(source="a.pdf", page=3, text="Conteúdo curto sobre embeddings.")]
    passages = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert len(passages) == 1
    assert passages[0].source == "a.pdf"
    assert passages[0].page == 3
    assert passages[0].citation == "a.pdf, p. 3"


def test_shorten_adiciona_reticencias():
    assert shorten("palavra " * 100, limit=40).endswith("…")
    assert shorten("curto", limit=40) == "curto"
