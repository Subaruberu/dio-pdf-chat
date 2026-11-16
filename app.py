"""Interface web do PDF Chat (Streamlit).

Rodar com:  streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from pdfchat import ChatEngine, Settings

st.set_page_config(
    page_title="PDF Chat · RAG com citações",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 1100px; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        background: linear-gradient(120deg, #fff4e6 0%, #ffe8cc 45%, #ffd8a8 100%);
        border: 1px solid #ffd8a8;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 1.9rem; color: #5f3b13; letter-spacing: -0.5px; }
    .hero p { margin: .4rem 0 0; color: #7a5224; font-size: .98rem; }
    .chip {
        display: inline-block; padding: .18rem .6rem; margin: .15rem .25rem .15rem 0;
        border-radius: 999px; font-size: .74rem; font-weight: 600;
        border: 1px solid transparent; white-space: nowrap;
    }
    .chip.ok   { background: #e6fcf5; color: #0b7285; border-color: #b2f2bb; }
    .chip.warn { background: #fff9db; color: #8a5a00; border-color: #ffe066; }
    .chip.info { background: #e7f5ff; color: #1864ab; border-color: #a5d8ff; }

    .docrow { font-size: .86rem; padding: .3rem 0; border-bottom: 1px dashed #eee; }
    .docrow small { color: #888; }

    .stChatMessage { border-radius: 14px; }
    .src {
        border-left: 3px solid #ffb366; padding: .35rem .7rem; margin: .5rem 0;
        background: #fffaf5; border-radius: 0 8px 8px 0; font-size: .85rem;
    }
    .src b { color: #995c00; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------- estado
def get_engine() -> ChatEngine:
    if "engine" not in st.session_state:
        settings = Settings.load()
        st.session_state.settings = settings
        st.session_state.engine = ChatEngine.open(settings)
        st.session_state.messages = []
        st.session_state.pending = None
    return st.session_state.engine


def chip(text: str, tone: str = "info") -> str:
    return f'<span class="chip {tone}">{html.escape(text)}</span>'


def ingest(paths: list[str | Path], rebuild: bool = False) -> None:
    engine = get_engine()
    with st.spinner("Extraindo texto, gerando embeddings e atualizando o índice…"):
        report = engine.ingest(paths, rebuild=rebuild)
    if report.error:
        st.session_state.flash = ("error", report.error)
        st.rerun()
    if report.added:
        message = f"✅ {report.summary()} em {report.elapsed:.1f}s"
        st.session_state.flash = ("success", message)
    elif report.skipped:
        st.session_state.flash = ("info", "Nada novo: os documentos enviados já estão indexados.")
    else:
        st.session_state.flash = ("warning", "Nenhum texto foi extraído dos arquivos enviados.")
    if report.warnings:
        st.session_state.flash = ("warning", " · ".join(report.warnings))
    st.rerun()


def save_uploads(files) -> list[Path]:
    settings = st.session_state.settings
    folder = Path(settings.index_dir).parent / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for upload in files:
        target = folder / Path(upload.name).name
        target.write_bytes(upload.getbuffer())
        saved.append(target)
    return saved


def snapshot_sources(answer) -> list[dict]:
    return [
        {
            "rank": item.rank,
            "citation": item.passage.citation,
            "source": item.passage.source,
            "page": item.passage.page,
            "score": item.score,
            "dense": item.dense,
            "lexical": item.lexical,
            "text": item.passage.text,
        }
        for item in answer.sources
    ]


def ask(question: str) -> None:
    engine = get_engine()
    messages = st.session_state.messages
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages.append({"role": "user", "content": question})
    with st.spinner("Consultando os documentos…"):
        answer = engine.ask(question, history=history)
    messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": snapshot_sources(answer),
            "backend": answer.backend,
            "model": answer.model,
            "elapsed": answer.elapsed,
            "tokens": (answer.prompt_tokens, answer.completion_tokens),
            "warning": answer.warning,
        }
    )


def conversation_markdown() -> str:
    lines = ["# Conversa — PDF Chat", ""]
    for message in st.session_state.get("messages", []):
        if message["role"] == "user":
            lines += [f"## 🧑 Pergunta", message["content"], ""]
        else:
            lines += ["## 🍦 Resposta", message["content"], ""]
            for source in message.get("sources", []):
                lines.append(f"- Fonte [{source['rank']}]: {source['citation']} (score {source['score']:.2f})")
            lines.append("")
    return "\n".join(lines)


engine = get_engine()
settings: Settings = st.session_state.settings
stats = engine.stats()
messages: list[dict] = st.session_state.messages

# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🍦 PDF Chat")
    st.caption("Pergunte aos seus PDFs. Cada resposta cita arquivo e página.")

    left, right = st.columns(2)
    left.metric("Documentos", stats["documents"])
    right.metric("Trechos", stats["chunks"])

    llm_badge = chip(stats["llm"], "ok" if stats["llm"].startswith("OpenAI") else "warn")
    embed_badge = chip(
        "Embeddings OpenAI" if stats["embedder"].startswith("openai") else "Embeddings locais",
        "ok" if stats["embedder"].startswith("openai") else "info",
    )
    st.markdown(llm_badge + embed_badge, unsafe_allow_html=True)
    if not settings.has_openai_key:
        st.caption("Sem `OPENAI_API_KEY`: modo local extrativo (sem custo, 100% offline).")

    st.divider()
    st.markdown("#### 📤 Enviar documentos")
    uploads = st.file_uploader(
        "PDF, TXT ou MD",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Indexar enviados", type="primary", use_container_width=True, disabled=not uploads):
        ingest(save_uploads(uploads))

    sample_dir = Path(settings.inputs_dir) / "pdfs"
    sample_target = sample_dir if sample_dir.exists() else Path(settings.inputs_dir)
    if st.button("✨ Indexar exemplos de /inputs", use_container_width=True):
        ingest([sample_target])

    st.divider()
    st.markdown("#### ⚙️ Ajustes")
    top_k = st.slider("Trechos por resposta", 2, 12, value=settings.top_k)
    settings.top_k = top_k
    mode = st.radio(
        "Recuperação",
        options=["hybrid", "dense", "lexical"],
        index=["hybrid", "dense", "lexical"].index(engine.retriever.mode),
        horizontal=True,
        help="hybrid = vetorial + BM25 (recomendado) · dense = só embeddings · lexical = só palavras-chave",
    )
    engine.retriever.mode = mode

    st.divider()
    st.markdown("#### 📚 Documentos")
    sources = engine.store.sources()
    if not sources:
        st.caption("Nenhum documento indexado ainda.")
    for name, info in sources.items():
        col_name, col_del = st.columns([5, 1])
        pages = f"{info['pages']} pág · " if info["pages"] else ""
        col_name.markdown(
            f'<div class="docrow">{html.escape(name)}<br><small>{pages}{info["chunks"]} trechos</small></div>',
            unsafe_allow_html=True,
        )
        if col_del.button("🗑", key=f"del-{name}", help=f"Remover {name}"):
            engine.remove_document(name)
            st.rerun()

    if sources:
        with st.expander("🗑 Apagar todo o índice"):
            confirm = st.checkbox("Confirmo que quero apagar tudo", key="confirm-clear")
            if st.button("Apagar índice", disabled=not confirm, use_container_width=True):
                engine.clear_index()
                st.session_state.messages = []
                st.rerun()

    if messages:
        st.divider()
        st.download_button(
            "⬇️ Baixar conversa (.md)",
            data=conversation_markdown(),
            file_name="conversa-pdfchat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("ℹ️ Como funciona"):
        st.markdown(
            """
1. **Extração** — `pypdf` lê cada página e limpa hifenização/espaços.
2. **Chunking** — trechos de ~900 caracteres com sobreposição de 150.
3. **Embeddings** — OpenAI (`text-embedding-3-small`) ou hashing local offline.
4. **Índice** — vetores em `.npy` + metadados em JSON (sem banco externo).
5. **Busca híbrida** — cosseno + BM25 fundidos por *Reciprocal Rank Fusion*.
6. **Resposta** — LLM recebe os trechos numerados e é obrigado a citar `[n]`.
            """
        )

# ----------------------------------------------------------------------- main
st.markdown(
    """
<div class="hero">
  <h1>🍦 Converse com seus PDFs</h1>
  <p>RAG com busca vetorial híbrida: respostas fundamentadas no seu material, com arquivo e página em cada citação.</p>
</div>
""",
    unsafe_allow_html=True,
)

for warning in settings.warnings:
    st.info(warning)

flash = st.session_state.pop("flash", None)
if flash:
    kind, message = flash
    if kind == "error":
        st.error(message)
    elif kind == "info":
        st.info(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.success(message)

if not messages:
    st.markdown("#### Comece por aqui")
    step1, step2, step3 = st.columns(3)
    step1.markdown("**1 · Envie**\n\nPDFs de artigos, apostilas ou seu TCC na barra lateral.")
    step2.markdown("**2 · Indexe**\n\nO texto vira embeddings e entra no índice vetorial.")
    step3.markdown("**3 · Pergunte**\n\nAs respostas citam o trecho, o arquivo e a página.")

    if stats["documents"] == 0:
        st.info("Nenhum documento no índice. Use **✨ Indexar exemplos de /inputs** na barra lateral para testar em segundos.")
    else:
        st.markdown("##### Experimente perguntar")
        suggestions = [
            "Qual foi a redução no tempo de triagem?",
            "Quantos estudantes participaram do estudo?",
            "Quais são as limitações apontadas?",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state.pending = suggestion
                st.rerun()
    st.divider()

for message in messages:
    role = message["role"]
    avatar = "🧑" if role == "user" else "🍦"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if message.get("warning"):
            st.warning(message["warning"])
        sources = message.get("sources") or []
        if sources:
            with st.expander(f"📚 {len(sources)} trecho(s) recuperado(s) — citação e pontuação", expanded=False):
                for source in sources:
                    st.markdown(
                        f'<div class="src"><b>[{source["rank"]}] {html.escape(source["citation"])}</b> · '
                        f'score <code>{source["score"]:.2f}</code> · '
                        f'denso <code>{source["dense"]:.2f}</code> · lexical <code>{source["lexical"]:.1f}</code>'
                        f'<br>{html.escape(source["text"][:420])}…</div>',
                        unsafe_allow_html=True,
                    )
        if role == "assistant":
            backend = "OpenAI" if message.get("backend") == "openai" else "local"
            tokens = message.get("tokens") or (None, None)
            token_note = f" · {tokens[0]}→{tokens[1]} tokens" if tokens[0] else ""
            st.caption(f"{backend} · {message.get('model', '')} · {message.get('elapsed', 0):.1f}s{token_note}")

if st.session_state.get("pending"):
    question, st.session_state.pending = st.session_state.pending, None
    ask(question)
    st.rerun()

if prompt := st.chat_input("Pergunte algo sobre os documentos…"):
    ask(prompt)
    st.rerun()

st.caption(
    f"Índice em `{stats['updated_at'] or '—'}` · "
    f"{stats['documents']} documento(s) · {stats['chunks']} trecho(s) · "
    f"recuperação `{engine.retriever.mode}`"
)
