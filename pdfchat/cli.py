"""Interface de terminal do PDF Chat.

Exemplos:
    python -m pdfchat.cli                     # conversa com o índice existente
    python -m pdfchat.cli inputs --rebuild    # reindexa a pasta inputs
    python -m pdfchat.cli --ask "O que é RAG?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .rag import ChatEngine

BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[0m",
)

HELP = """Comandos:
  /ajuda            mostra esta ajuda
  /docs             lista os documentos indexados
  /fontes           mostra o índice e os embeddings em uso
  /top N            mostra N trechos recuperados (1-20)
  /modo MODO        hybrid | dense | lexical
  /limpar           apaga o índice
  /sair             encerra a conversa
"""


def banner(engine: ChatEngine) -> None:
    settings = engine.settings
    info = settings.describe()
    print(f"{BOLD}🍦 PDF Chat{RESET} {DIM}— perguntas fundamentadas nos seus PDFs{RESET}")
    print(f"  LLM:        {info['llm']}")
    print(f"  Embeddings: {info['embeddings']}")
    print(f"  Recuperação:{' ' * 1} {engine.retriever.mode} · top_k={engine.settings.top_k}")
    stats = engine.stats()
    print(f"  Índice:     {stats['documents']} doc(s) · {stats['chunks']} trecho(s) · {stats['pages']} pág.")
    if stats["documents"] == 0:
        print(f"  {YELLOW}Dica:{RESET} rode `python ingest.py inputs` para indexar os arquivos de exemplo.")
    for warning in settings.warnings:
        print(f"  {YELLOW}⚠ {warning}{RESET}")
    print()


def print_answer(engine: ChatEngine, question: str, show_sources: bool = True) -> None:
    answer = engine.ask(question)
    label = "OpenAI" if answer.backend == "openai" else "local"
    print(f"\n{BOLD}Assistente{RESET} {DIM}({label} · {answer.elapsed:.1f}s){RESET}")
    print(answer.text)
    if answer.warning:
        print(f"{YELLOW}⚠ {answer.warning}{RESET}")
    if show_sources and answer.sources:
        print(f"\n{DIM}Fontes usadas:{RESET}")
        for item in answer.sources:
            print(f"  {CYAN}[{item.rank}]{RESET} {item.passage.citation} {DIM}(score {item.score:.2f}){RESET}")
            snippet = " ".join(item.passage.text.split())[:180]
            print(f"      {DIM}{snippet}…{RESET}")
    print()


def list_documents(engine: ChatEngine) -> None:
    sources = engine.store.sources()
    if not sources:
        print("Nenhum documento indexado.")
        return
    print(f"{BOLD}Documentos indexados{RESET}")
    for name, info in sources.items():
        pages = f"{info['pages']} pág." if info["pages"] else "texto"
        print(f"  • {name} {DIM}— {pages}, {info['chunks']} trecho(s){RESET}")


def run_chat(engine: ChatEngine, top_k: int) -> None:
    settings = engine.settings
    settings.top_k = top_k
    while True:
        try:
            question = input(f"{GREEN}Você:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais! 👋")
            return
        if not question:
            continue
        command = question.lower()
        if command in {"/sair", "/exit", "/quit"}:
            print("Até mais! 👋")
            return
        if command == "/ajuda":
            print(HELP)
            continue
        if command == "/docs":
            list_documents(engine)
            continue
        if command == "/fontes":
            stats = engine.stats()
            print(f"{BOLD}Índice{RESET}: {settings.index_dir}")
            print(f"{BOLD}Embeddings{RESET}: {stats['embedder']}")
            print(f"{BOLD}LLM{RESET}: {stats['llm']}")
            print(f"{BOLD}Documentos{RESET}: {stats['documents']} · trechos: {stats['chunks']}")
            continue
        if command.startswith("/top"):
            try:
                value = int(command.split()[1])
            except (IndexError, ValueError):
                print("Uso: /top 8")
                continue
            settings.top_k = max(1, min(value, 20))
            print(f"top_k = {settings.top_k}")
            continue
        if command.startswith("/modo"):
            parts = command.split()
            mode = parts[1] if len(parts) > 1 else ""
            if mode not in {"hybrid", "dense", "lexical"}:
                print("Uso: /modo hybrid | dense | lexical")
                continue
            engine.retriever.mode = mode
            print(f"modo de recuperação = {mode}")
            continue
        if command == "/limpar":
            confirm = input("Isso apaga o índice. Confirmar? [s/N] ").strip().lower()
            if confirm == "s":
                engine.clear_index()
                print("Índice apagado.")
            continue
        print_answer(engine, question)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(
        prog="pdfchat",
        description="Chat com IA que responde com base nos seus PDFs (RAG).",
    )
    parser.add_argument("paths", nargs="*", help="PDFs, .txt/.md ou pastas a indexar")
    parser.add_argument("--rebuild", action="store_true", help="recria o índice do zero")
    parser.add_argument("--ask", metavar="PERGUNTA", help="faz uma pergunta e encerra")
    parser.add_argument("--stats", action="store_true", help="mostra o estado do índice")
    parser.add_argument("--list", action="store_true", help="lista os documentos indexados")
    parser.add_argument("--clear", action="store_true", help="apaga o índice")
    parser.add_argument("--top-k", type=int, default=None, help="trechos recuperados por pergunta")
    parser.add_argument("--no-chat", action="store_true", help="não abre o modo interativo")
    args = parser.parse_args(argv)

    settings = Settings.load()
    if args.top_k:
        settings.top_k = max(1, min(args.top_k, 20))
    engine = ChatEngine.open(settings)
    banner(engine)

    if args.clear:
        engine.clear_index()
        print("Índice apagado.")
        engine = ChatEngine.open(settings)

    if args.paths:
        report = engine.ingest(args.paths, rebuild=args.rebuild)
        if report.error:
            print(f"{RED}Erro:{RESET} {report.error}")
            return 1
        print(f"{GREEN}✓{RESET} {report.summary()} {DIM}({report.elapsed:.1f}s){RESET}")
        for warning in report.warnings:
            print(f"  {YELLOW}⚠ {warning}{RESET}")
        carregados = engine.store.sources()
        for name in report.added:
            info = carregados.get(name)
            if info:
                print(f"  + {name} {DIM}({info['chunks']} trechos){RESET}")
        print()
        if args.no_chat and not args.ask:
            return 0

    if args.stats:
        stats = engine.stats()
        print(f"{BOLD}Índice{RESET}: {stats['updated_at'] or 'vazio'}")
        print(f"{BOLD}Documentos{RESET}: {stats['documents']} · páginas: {stats['pages']} · trechos: {stats['chunks']}")
        print(f"{BOLD}Embeddings{RESET}: {stats['embedder']}")
        print(f"{BOLD}LLM{RESET}: {stats['llm']}")
        print()

    if args.list:
        list_documents(engine)
        print()

    if args.ask:
        print_answer(engine, args.ask)
        return 0

    if args.no_chat:
        return 0

    if not sys.stdin.isatty():
        print(f"{YELLOW}Terminal não interativo.{RESET} Use --ask \"sua pergunta\" ou --no-chat.")
        return 0

    print(f"{DIM}Digite sua pergunta ou /ajuda. Enter vazio ignora.{RESET}\n")
    run_chat(engine, settings.top_k)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
