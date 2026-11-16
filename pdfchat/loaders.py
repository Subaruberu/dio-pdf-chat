"""Leitura de PDFs e arquivos de texto, com limpeza para melhorar a extração."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .models import Page

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_MANY_BREAKS = re.compile(r"\n{3,}")


class ExtractionError(RuntimeError):
    """Falha ao ler um documento."""


def clean_text(text: str) -> str:
    """Normaliza quebras, hifenização de fim de linha e espaços repetidos."""
    if not text:
        return ""
    text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    # "engenha-\nria" -> "engenharia" (hifenização de quebra de linha comum em PDFs)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = _WHITESPACE.sub(" ", text)
    text = _MANY_BREAKS.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _normalize_running_line(line: str) -> str:
    """Normaliza números para detectar cabeçalhos/rodapés repetidos com número de página."""
    return re.sub(r"\d+", "#", " ".join(line.split()).lower())


def _repeated_lines(raw_pages: list[str]) -> set[str]:
    """Linhas curtas que aparecem em pelo menos metade das páginas (cabeçalho/rodapé)."""
    counts: Counter[str] = Counter()
    for raw in raw_pages:
        seen = {
            normalized
            for line in raw.splitlines()
            if 3 <= len(normalized := _normalize_running_line(line)) <= 120
        }
        counts.update(seen)
    threshold = max(2, math.ceil(len(raw_pages) * 0.5))
    return {line for line, count in counts.items() if count >= threshold}


def _strip_running_lines(raw: str, repeated: set[str]) -> str:
    if not repeated:
        return raw
    kept = [line for line in raw.splitlines() if _normalize_running_line(line) not in repeated]
    return "\n".join(kept)


def load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - queremos reportar a causa original
        raise ExtractionError(f"Não foi possível abrir {path.name}: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"{path.name} está protegido por senha.") from exc

    pages: list[Page] = []
    raw_pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - uma página ruim não derruba o arquivo
            raw = ""
        raw_pages.append((number, raw))

    repeated = _repeated_lines([raw for _, raw in raw_pages])
    for number, raw in raw_pages:
        text = clean_text(_strip_running_lines(raw, repeated))
        if len(text.split()) >= 5:  # ignora páginas em branco / capa sem texto
            pages.append(Page(source=path.name, page=number, text=text))
    return pages


def load_text_file(path: Path) -> list[Page]:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 sempre decodifica
        raise ExtractionError(f"Não foi possível decodificar {path.name}.")
    text = clean_text(raw)
    return [Page(source=path.name, page=None, text=text)] if text else []


def load_file(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in SUPPORTED_EXTENSIONS:
        return load_text_file(path)
    raise ExtractionError(f"Formato não suportado: {path.name} ({suffix}).")


def collect_files(paths: list[str | Path]) -> list[Path]:
    """Expande diretórios recursivamente e filtra extensões suportadas."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(
                p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
        elif path.exists():
            raise ExtractionError(f"Formato não suportado: {path.name}.")
        else:
            raise FileNotFoundError(f"Caminho não encontrado: {path}")
    # deduplica preservando a ordem
    seen: set[Path] = set()
    unique = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique
