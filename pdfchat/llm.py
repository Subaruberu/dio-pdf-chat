"""Geração da resposta: LLM via OpenAI ou modo extrativo local (offline)."""

from __future__ import annotations

import re
import time

from .config import Settings
from .models import Retrieved
from .textutils import split_sentences, tokenize

SYSTEM_PROMPT = """Você é o PDF Chat, um assistente de estudos que responde com base nos documentos do usuário.

Regras obrigatórias:
1. Use SOMENTE as informações do CONTEXTO fornecido. Não use conhecimento externo.
2. Cite a fonte de cada afirmação com o número do trecho entre colchetes, por exemplo [1] ou [2][3].
3. Se o contexto não contiver informação suficiente, diga isso claramente e sugira o que procurar.
4. Responda em português do Brasil, de forma direta e organizada (use marcadores quando ajudar).
5. Nunca invente números, nomes, citações ou referências bibliográficas.
6. Ao comparar documentos, deixe explícito de qual arquivo/página vem cada ponto."""

NO_CONTEXT_REPLY = (
    "Não encontrei essa informação nos documentos indexados. "
    "Tente reformular a pergunta, enviar o PDF que contém esse conteúdo ou aumentar a quantidade de trechos recuperados."
)


def build_context(sources: list[Retrieved], max_chars: int = 7000) -> str:
    """Monta o contexto numerado [1]..[n] que o LLM deve citar."""
    blocks: list[str] = []
    used = 0
    for index, item in enumerate(sources, start=1):
        page = f", p. {item.passage.page}" if item.passage.page is not None else ""
        header = f"[{index}] {item.passage.source}{page}"
        body = item.passage.text
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            remaining = max_chars - used - len(header) - 4
            if remaining < 200:
                break
            block = f"{header}\n{body[:remaining].rstrip()}…"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def build_messages(question: str, sources: list[Retrieved], history: list[dict] | None, max_chars: int) -> list[dict]:
    context = build_context(sources, max_chars=max_chars)
    user_content = (
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA: {question}\n\n"
        "Responda usando apenas o contexto acima e cite as fontes com [n]."
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in history or []:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


class OpenAILLM:
    """Cliente de chat da OpenAI (compatível com base_url customizada)."""

    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instale o pacote 'openai' para usar este backend.") from exc

        self.settings = settings
        self.model = settings.llm_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=90.0,
            max_retries=2,
        )

    def answer(self, question: str, sources: list[Retrieved], history=None) -> tuple[str, dict]:
        messages = build_messages(
            question, sources, history, max_chars=self.settings.max_context_chars
        )
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=900,
        )
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        meta = {
            "elapsed": time.perf_counter() - started,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        return text, meta


class ExtractiveLLM:
    """Responde sem LLM: seleciona as sentenças mais aderentes à pergunta.

    É o modo padrão quando não há OPENAI_API_KEY. Continua sendo RAG: a
    resposta é fundamentada e cada trecho vem com a fonte e a página.
    """

    model = "extrativo-local"

    def answer(self, question: str, sources: list[Retrieved], history=None) -> tuple[str, dict]:
        started = time.perf_counter()
        if not sources:
            return NO_CONTEXT_REPLY, {"elapsed": time.perf_counter() - started}

        query_terms = set(tokenize(question))
        chosen: list[tuple[str, int]] = []  # (sentença, índice da fonte 1-based)
        used_sentences: set[str] = set()

        for index, item in enumerate(sources, start=1):
            best_sentence = ""
            best_score = 0.0
            for sentence in split_sentences(item.passage.text):
                clean = sentence.strip()
                if len(clean) < 40 or clean in used_sentences:
                    continue
                terms = tokenize(clean)
                if not terms:
                    continue
                overlap = len(query_terms & set(terms))
                if not overlap:
                    continue
                score = overlap / (len(terms) ** 0.35)
                if score > best_score:
                    best_score, best_sentence = score, clean
            if best_sentence:
                used_sentences.add(best_sentence)
                chosen.append((best_sentence, index))

        if not chosen:
            return NO_CONTEXT_REPLY, {"elapsed": time.perf_counter() - started}

        chosen = chosen[:4]
        lines = ["**Resposta (modo local — extrativa)**", "", "Trechos mais relevantes dos documentos:"]
        for position, (sentence, source_index) in enumerate(chosen, start=1):
            item = sources[source_index - 1]
            lines.append(f"{position}. “{ _trim(sentence, 420) }” — **[{source_index}]** `{item.passage.citation}`")
        lines += [
            "",
            "> **Modo offline:** a resposta foi montada selecionando sentenças dos trechos recuperados, "
            "sem gerar texto livre. Defina `OPENAI_API_KEY` no `.env` para obter respostas sintetizadas por IA.",
        ]
        return "\n".join(lines), {"elapsed": time.perf_counter() - started, "chosen": len(chosen)}


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip(" ,.;:") + "…"


def build_llm(settings: Settings, backend: str | None = None):
    backend = backend or settings.resolve_llm_backend()
    return OpenAILLM(settings) if backend == "openai" else ExtractiveLLM()
