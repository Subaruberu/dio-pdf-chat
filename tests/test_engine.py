"""Testes de integração do ChatEngine (ingestão, persistência e perguntas)."""

from __future__ import annotations

from pathlib import Path

from pdfchat.rag import ChatEngine


def test_ingest_e_pergunta_com_fonte(settings, corpus: Path):
    engine = ChatEngine.open(settings)
    report = engine.ingest([corpus])

    assert report.error is None
    assert sorted(report.added) == ["bm25.txt", "rag.txt"]
    assert report.pages == 2
    assert report.chunks >= 2
    assert engine.stats()["documents"] == 2

    answer = engine.ask("Qual é o tamanho dos trechos e a sobreposição?")
    assert answer.sources
    assert "[1]" in answer.text
    assert "900" in " ".join(s.passage.text for s in answer.sources)
    assert answer.backend == "local"


def test_reingest_ignora_arquivos_inalterados(settings, corpus: Path):
    engine = ChatEngine.open(settings)
    engine.ingest([corpus])

    again = engine.ingest([corpus])
    assert again.added == []
    assert sorted(again.skipped) == ["bm25.txt", "rag.txt"]

    rebuild = engine.ingest([corpus], rebuild=True)
    assert sorted(rebuild.added) == ["bm25.txt", "rag.txt"]
    assert engine.stats()["documents"] == 2


def test_indice_persiste_entre_sessoes(settings, corpus: Path):
    ChatEngine.open(settings).ingest([corpus])
    reopened = ChatEngine.open(settings)
    assert reopened.stats()["documents"] == 2
    assert reopened.ask("O que é BM25?").sources


def test_pergunta_sem_documentos(settings):
    engine = ChatEngine.open(settings)
    answer = engine.ask("Qual o tema?")
    assert "Nenhum documento" in answer.text
    assert answer.sources == []


def test_pergunta_fora_do_corpus(settings, corpus: Path):
    engine = ChatEngine.open(settings)
    engine.ingest([corpus])
    answer = engine.ask("Qual é a capital da França?")
    assert answer.text
    assert "não encontrei" in answer.text.lower()


def test_remover_documento(settings, corpus: Path):
    engine = ChatEngine.open(settings)
    engine.ingest([corpus])
    removed = engine.remove_document("rag.txt")
    assert removed > 0
    assert engine.stats()["documents"] == 1
    assert all(p.source != "rag.txt" for p in engine.store.passages)


def test_pdf_com_paginas(settings, tmp_path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "artigo.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf.drawString(72, 780, "Este artigo descreve a tecnica de Recuperacao Aumentada por Geracao.")
    pdf.drawString(72, 760, "O corpus possui 42 artigos e o chunk tem 900 caracteres.")
    pdf.drawString(72, 40, "Rodape institucional repetido · p. 1")
    pdf.showPage()
    pdf.drawString(72, 780, "Segunda pagina: o BM25 complementa a busca vetorial no ranking hibrido.")
    pdf.drawString(72, 40, "Rodape institucional repetido · p. 2")
    pdf.save()

    engine = ChatEngine.open(settings)
    report = engine.ingest([pdf_path])
    assert report.error is None
    assert report.pages == 2
    assert all("Rodape institucional" not in p.text for p in engine.store.passages)

    answer = engine.ask("Quantos artigos o corpus possui?")
    pages = {s.passage.page for s in answer.sources}
    assert 1 in pages
    assert any("42" in s.passage.text for s in answer.sources)
