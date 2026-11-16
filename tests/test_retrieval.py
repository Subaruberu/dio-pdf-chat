"""Testes de embeddings, BM25, índice e recuperação híbrida."""

from __future__ import annotations

import numpy as np

from pdfchat.bm25 import BM25
from pdfchat.embeddings import HashingEmbedder
from pdfchat.models import Passage
from pdfchat.retriever import HybridRetriever
from pdfchat.store import VectorStore


def _passage(index: int, text: str) -> Passage:
    return Passage(id=f"p{index}", text=text, source="doc.txt", page=None, chunk=index)


def test_embedding_local_e_deterministico_e_normalizado():
    embedder = HashingEmbedder(dim=256)
    matrix = embedder.encode(["embeddings vetoriais para busca", "embeddings vetoriais para busca"])
    assert matrix.shape == (2, 256)
    assert np.allclose(matrix[0], matrix[1])
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0, atol=1e-5)


def test_embedding_local_captura_similaridade():
    embedder = HashingEmbedder(dim=1024)
    vectors = embedder.encode(
        [
            "busca vetorial com embeddings e similaridade de cosseno",
            "recuperação de documentos usando vetores e cosseno",
            "receita de bolo de chocolate com morangos frescos",
        ]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_bm25_prioriza_termo_raro():
    docs = [
        "documento sobre engenharia de software e testes unitários",
        "o algoritmo BM25 calcula ranking lexical com idf e frequência de termo",
        "outro texto qualquer sobre gestão de projetos",
    ]
    scores = BM25(docs).scores("BM25 ranking lexical")
    assert int(np.argmax(scores)) == 1
    assert scores[1] > scores[0]


def test_store_roundtrip(tmp_path):
    embedder = HashingEmbedder(dim=128)
    store = VectorStore()
    passages = [_passage(0, "texto sobre RAG e embeddings"), _passage(1, "texto sobre BM25")]
    store.add(passages, embedder.encode([p.text for p in passages]))
    assert len(store) == 2

    directory = store.save(tmp_path / "index", embedder_name=embedder.name)
    loaded = VectorStore.load(directory)
    assert len(loaded) == 2
    assert loaded.passages[1].text == "texto sobre BM25"
    assert loaded.dim == 128
    assert loaded.embedder_name == "local-hashing-128d"

    results = loaded.search(embedder.encode_one("RAG e embeddings"), top_k=1)
    assert results[0][0] == 0


def test_store_remove_fonte(tmp_path):
    embedder = HashingEmbedder(dim=128)
    store = VectorStore()
    store.add(
        [
            Passage("a", "texto a", "a.pdf", 1, 0),
            Passage("b", "texto b", "b.pdf", 1, 0),
        ],
        embedder.encode(["texto a", "texto b"]),
    )
    assert store.remove_sources({"a.pdf"}) == 1
    assert len(store) == 1
    assert store.passages[0].source == "b.pdf"


def test_recuperacao_hibrida_prioriza_trecho_certo():
    embedder = HashingEmbedder(dim=512)
    passages = [
        _passage(0, "A citação da fonte e da página reduz alucinações do modelo de linguagem."),
        _passage(1, "O BM25 usa frequência de termo e comprimento do documento no ranking lexical."),
        _passage(2, "O chunking divide o PDF em trechos de 900 caracteres com sobreposição."),
    ]
    store = VectorStore()
    store.add(passages, embedder.encode([p.text for p in passages]))
    retriever = HybridRetriever(store, embedder, mode="hybrid")
    results = retriever.search("qual o tamanho dos trechos do chunking?", top_k=2)
    assert results
    assert results[0].passage.id == "p2"
    assert 0.0 <= results[0].score <= 1.0
