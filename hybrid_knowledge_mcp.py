#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_knowledge_mcp.py — локальный MCP-сервер гибридного поиска по Go-коду
и документации (выгрузки Confluence в .md/.txt).

Архитектура
-----------
* Эмбеддинги  : sentence-transformers/all-MiniLM-L6-v2 (CPU, PyTorch), 384-dim,
                нормализованные векторы.
* Векторный BD: ChromaDB PersistentClient (cosine space), embedding_function=None —
                векторы считаем сами и передаём явно.
* Лексический : rank_bm25.BM25Okapi, состояние сохраняется через pickle.
* Слияние     : Reciprocal Rank Fusion, score = sum(1 / (k + rank)), k = 60.
* Протокол    : официальный Python SDK `mcp` (low-level Server + stdio transport).

Инструменты MCP
---------------
* index_directory(path)             — инкрементальная индексация каталога.
* search_knowledge_base(query, top_k) — гибридный поиск (векторы + BM25 + RRF).
* clear_index()                     — полная очистка индексов, кэша хэшей и БД.

Раскладка на диске (по умолчанию ~/.local/share/go-confluence-mcp):
    hybrid_knowledge_mcp.py   — этот файл
    venv/                     — виртуальное окружение
    models/                   — HF_HOME, сюда заранее скачивается MiniLM
    data/chroma/              — persistent-хранилище ChromaDB
    data/bm25.pkl             — pickle со состоянием BM25
    data/manifest.json        — кэш хэшей файлов (MD5 + SHA256) и id чанков

ВАЖНО про stdio: stdout зарезервирован под JSON-RPC. Любой посторонний вывод
в fd 1 ломает протокол, поэтому при старте fd 1 перенаправляется в stderr,
а «настоящий» stdout сохраняется отдельно и отдаётся только транспорту MCP.
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------- #
# 0. Защита stdio-канала и переменные окружения — ДО импорта тяжёлых библиотек. #
# --------------------------------------------------------------------------- #

#: Дескриптор «настоящего» stdout, отданный транспорту MCP и никому больше.
_PROTOCOL_STDOUT_FD: int | None = None

try:
    _PROTOCOL_STDOUT_FD = os.dup(1)  # сохраняем реальный stdout
    os.dup2(2, 1)  # всё, что пишется в fd 1 (в т.ч. из C-расширений), уходит в stderr
except OSError:  # pragma: no cover — на всякий случай, если stdio нестандартный
    _PROTOCOL_STDOUT_FD = None

# print() и прочий python-level вывод тоже уводим в stderr.
sys.stdout = sys.stderr

# Тишина и отсутствие телеметрии: прогресс-бары HF и posthog у Chroma
# любят писать в stdout/сеть — нам не нужно ни то, ни другое.
for _key, _value in {
    "ANONYMIZED_TELEMETRY": "False",
    "CHROMA_ANALYTICS": "False",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}.items():
    os.environ.setdefault(_key, _value)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import heapq  # noqa: E402
import importlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable, Iterator, Sequence  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. Конфигурация                                                              #
# --------------------------------------------------------------------------- #

SERVER_NAME = "go-confluence-kb"
SERVER_VERSION = "1.0.0"

DEFAULT_BASE_DIR = Path(
    os.environ.get(
        "GO_CONFLUENCE_MCP_HOME",
        Path.home() / ".local" / "share" / "go-confluence-mcp",
    )
).expanduser()

EMBEDDING_MODEL = os.environ.get(
    "GO_CONFLUENCE_MCP_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
COLLECTION_NAME = "hybrid_knowledge"
MANIFEST_VERSION = 1
BM25_STATE_VERSION = 1

#: k из формулы Reciprocal Rank Fusion.
RRF_K = 60

#: Сколько кандидатов тянем из каждой подсистемы перед слиянием.
CANDIDATE_MULTIPLIER = 6
MIN_CANDIDATES = 40
MAX_CANDIDATES = 300

#: Расширения → тип источника.
CODE_EXTENSIONS = {".go"}
DOC_EXTENSIONS = {".md", ".markdown", ".txt"}
INDEXABLE_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS

SOURCE_CODE = "Code"
SOURCE_CONFLUENCE = "Confluence"

#: Каталоги, которые никогда не индексируем.
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".vs",
    "node_modules", "vendor", "venv", ".venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "target", "bin", "obj",
    ".terraform", ".gradle", ".next", ".cache", ".tox",
}

#: Файлы больше этого размера пропускаем (сгенерированный код, дампы).
MAX_FILE_BYTES = 2 * 1024 * 1024

#: Параметры нарезки.
GO_MAX_CHUNK_CHARS = 2400
GO_MIN_CHUNK_CHARS = 300
GO_OVERLAP_LINES = 3
DOC_CHUNK_CHARS = 1200
DOC_CHUNK_OVERLAP = 160

#: Размер батча при кодировании и записи в Chroma.
EMBED_BATCH = 32
CHROMA_BATCH = 1000

#: Ограничения на выдачу.
MAX_TOP_K = 50
SNIPPET_CHARS = 900

# --------------------------------------------------------------------------- #
# 2. Логирование — строго в sys.stderr                                         #
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.environ.get("GO_CONFLUENCE_MCP_LOG_LEVEL", "INFO").upper()

_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
)
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), handlers=[_handler])

# Болтливые зависимости приглушаем.
for _noisy in (
    "chromadb", "chromadb.telemetry", "chromadb.segment",
    "sentence_transformers", "sentence_transformers.SentenceTransformer",
    "transformers", "urllib3", "httpx", "filelock", "torch",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("go-confluence-kb")


# --------------------------------------------------------------------------- #
# 3. Хэши и токенизация                                                        #
# --------------------------------------------------------------------------- #

def file_digests(path: Path) -> tuple[str, str, int]:
    """Считает MD5 и SHA256 файла за один проход. Возвращает (md5, sha256, size)."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
            size += len(block)
    return md5.hexdigest(), sha.hexdigest(), size


def path_key(path: Path) -> str:
    """Стабильный короткий префикс id чанков, производный от абсолютного пути."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Токенайзер для BM25, заточенный под смесь кода и прозы.

    `ParseUserConfig` → ['parseuserconfig', 'parse', 'user', 'config'],
    `http_client`     → ['http_client', 'http', 'client'].
    Такое расширение резко повышает точное попадание по именам символов.
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        tokens.append(lowered)
        parts = [p.lower() for p in _CAMEL_RE.findall(word) if len(p) > 1]
        if len(parts) > 1 or (parts and parts[0] != lowered):
            tokens.extend(parts)
    return tokens


# --------------------------------------------------------------------------- #
# 4. Нарезка Go: package / func / method / struct / interface                  #
# --------------------------------------------------------------------------- #

_GO_DECL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("package", re.compile(r"^package\s+([A-Za-z_]\w*)")),
    ("import", re.compile(r"^import\b")),
    # Метод обязан проверяться раньше обычной функции.
    ("method", re.compile(
        r"^func\s*\(\s*(?:[A-Za-z_]\w*\s+)?(\*?)([A-Za-z_]\w*)[^)]*\)\s*([A-Za-z_]\w*)"
    )),
    ("func", re.compile(r"^func\s+([A-Za-z_]\w*)")),
    ("struct", re.compile(r"^type\s+([A-Za-z_]\w*)(?:\[[^\]]*\])?\s+struct\b")),
    ("interface", re.compile(r"^type\s+([A-Za-z_]\w*)(?:\[[^\]]*\])?\s+interface\b")),
    ("typeblock", re.compile(r"^type\s*\(")),
    ("type", re.compile(r"^type\s+([A-Za-z_]\w*)")),
    ("const", re.compile(r"^const\b")),
    ("var", re.compile(r"^var\b")),
)

#: Виды объявлений, которые допустимо склеивать между собой в один чанк.
#: func / method / struct / interface сюда сознательно не входят.
_GO_MERGEABLE_KINDS = frozenset({"header", "package", "import", "const", "var", "type"})


def _go_scan_line(line: str, state: dict[str, bool]) -> tuple[int, int]:
    """
    Считает баланс `{}` и `()` в строке Go, игнорируя комментарии, строковые
    и рунные литералы. `state` переносит между строками контекст блочного
    комментария и raw-строки (backticks).
    """
    brace = 0
    paren = 0
    i = 0
    n = len(line)
    while i < n:
        if state["block_comment"]:
            end = line.find("*/", i)
            if end == -1:
                return brace, paren
            state["block_comment"] = False
            i = end + 2
            continue
        if state["raw_string"]:
            end = line.find("`", i)
            if end == -1:
                return brace, paren
            state["raw_string"] = False
            i = end + 1
            continue

        ch = line[i]
        if ch == "/" and i + 1 < n:
            nxt = line[i + 1]
            if nxt == "/":
                return brace, paren  # построчный комментарий — остаток строки не код
            if nxt == "*":
                state["block_comment"] = True
                i += 2
                continue
        if ch == "`":
            state["raw_string"] = True
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        i += 1
    return brace, paren


def _match_go_decl(line: str) -> tuple[str | None, str]:
    """По строке нулевого отступа определяет вид объявления и человекочитаемое имя."""
    for kind, pattern in _GO_DECL_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue
        if kind == "package":
            return kind, f"package {match.group(1)}"
        if kind == "import":
            return kind, "imports"
        if kind == "method":
            star, receiver, name = match.group(1), match.group(2), match.group(3)
            return kind, f"({star}{receiver}) {name}"
        if kind == "func":
            return kind, f"func {match.group(1)}"
        if kind in ("struct", "interface", "type"):
            return kind, f"{kind} {match.group(1)}"
        if kind == "typeblock":
            return "type", "type block"
        return kind, f"{kind} block"
    return None, ""


def _split_code_block(text: str, max_chars: int, overlap_lines: int) -> list[str]:
    """Режет слишком длинное объявление по строкам, сохраняя перекрытие."""
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines()
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in lines:
        line_size = len(line) + 1
        if buffer and size + line_size > max_chars:
            chunks.append("\n".join(buffer))
            tail = buffer[-overlap_lines:] if overlap_lines else []
            buffer = list(tail)
            size = sum(len(item) + 1 for item in buffer)
        buffer.append(line)
        size += line_size
    if buffer:
        chunks.append("\n".join(buffer))
    return [chunk for chunk in chunks if chunk.strip()]


def split_go(text: str, rel_path: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Специализированный сплиттер Go: один чанк ≈ одно объявление верхнего уровня
    вместе с предшествующим doc-комментарием.

    Возвращает список пар (текст чанка, дополнительные метаданные).
    """
    lines = text.splitlines()
    if not lines:
        return []

    state = {"block_comment": False, "raw_string": False}
    depth_brace = 0
    depth_paren = 0
    package_name = ""
    decls: list[tuple[int, str, str]] = []  # (индекс первой строки, вид, имя)
    comment_run: int | None = None

    for idx, line in enumerate(lines):
        at_top = (
            depth_brace <= 0
            and depth_paren <= 0
            and not state["block_comment"]
            and not state["raw_string"]
        )
        if at_top and line and not line[0].isspace():
            if line.startswith("//"):
                if comment_run is None:
                    comment_run = idx
            else:
                kind, symbol = _match_go_decl(line)
                if kind:
                    start = comment_run if comment_run is not None else idx
                    decls.append((start, kind, symbol))
                    if kind == "package" and not package_name:
                        package_name = symbol.split(" ", 1)[-1]
                comment_run = None
        else:
            comment_run = None

        delta_brace, delta_paren = _go_scan_line(line, state)
        depth_brace = max(0, depth_brace + delta_brace)
        depth_paren = max(0, depth_paren + delta_paren)

    if not decls:
        # Не распознали структуру (например, файл только из комментариев) —
        # откатываемся на построчную нарезку, чтобы контент не потерялся.
        segments = [(text, "file", "file")]
    else:
        if decls[0][0] > 0:
            decls.insert(0, (0, "header", "file header"))
        segments = []
        for position, (start, kind, symbol) in enumerate(decls):
            end = decls[position + 1][0] if position + 1 < len(decls) else len(lines)
            body = "\n".join(lines[start:end]).strip("\n")
            if body.strip():
                segments.append((body, symbol, kind))

    # Склеиваем только «шапочные» объявления (package, import, одиночные
    # const/var/type), иначе индекс распухнет от чанков в две строки.
    # func, method, struct и interface всегда остаются отдельными чанками —
    # именно на них рассчитан поиск по символам.
    merged: list[tuple[str, str, str]] = []
    for body, symbol, kind in segments:
        if merged and kind in _GO_MERGEABLE_KINDS:
            prev_body, prev_symbol, prev_kind = merged[-1]
            if (
                prev_kind in _GO_MERGEABLE_KINDS
                and len(prev_body) < GO_MIN_CHUNK_CHARS
                and len(prev_body) + len(body) <= GO_MAX_CHUNK_CHARS
            ):
                merged[-1] = (
                    f"{prev_body}\n\n{body}", f"{prev_symbol}, {symbol}", prev_kind
                )
                continue
        merged.append((body, symbol, kind))

    chunks: list[tuple[str, dict[str, Any]]] = []
    for body, symbol, _kind in merged:
        for part in _split_code_block(body, GO_MAX_CHUNK_CHARS, GO_OVERLAP_LINES):
            short_symbol = symbol if len(symbol) <= 120 else symbol[:117] + "..."
            header = f"// file: {rel_path}\n"
            if package_name:
                header += f"// package: {package_name}\n"
            header += f"// symbol: {short_symbol}\n\n"
            chunks.append((header + part, {"symbol": short_symbol, "package": package_name}))
    return chunks


# --------------------------------------------------------------------------- #
# 5. Recursive character splitting для .md/.txt                                #
# --------------------------------------------------------------------------- #

_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", "")


def recursive_split(
    text: str,
    chunk_size: int = DOC_CHUNK_CHARS,
    overlap: int = DOC_CHUNK_OVERLAP,
    separators: Sequence[str] = _DEFAULT_SEPARATORS,
) -> list[str]:
    """
    Классический recursive character splitting: пытаемся резать по самому
    «крупному» разделителю, который встречается в тексте, и спускаемся ниже,
    пока куски не влезут в chunk_size. В конце добавляем перекрытие.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    def _split(fragment: str, seps: Sequence[str]) -> list[str]:
        if len(fragment) <= chunk_size:
            return [fragment] if fragment.strip() else []

        separator = ""
        rest: Sequence[str] = ()
        for position, candidate in enumerate(seps):
            if candidate == "":
                break
            if candidate in fragment:
                separator = candidate
                rest = seps[position + 1:]
                break

        if not separator:
            step = max(1, chunk_size - overlap)
            return [
                fragment[start:start + chunk_size]
                for start in range(0, len(fragment), step)
                if fragment[start:start + chunk_size].strip()
            ]

        pieces = fragment.split(separator)
        out: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = piece if not buffer else buffer + separator + piece
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue
            if buffer:
                out.append(buffer)
                buffer = ""
            if len(piece) > chunk_size:
                out.extend(_split(piece, rest))
            else:
                buffer = piece
        if buffer:
            out.append(buffer)
        return [item for item in out if item.strip()]

    chunks = _split(text, list(separators))
    return _apply_overlap(chunks, overlap)


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Дописывает в начало каждого чанка хвост предыдущего — контекст на стыках."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    result = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous[-overlap:].lstrip()
        result.append(f"{tail}\n{current}" if tail else current)
    return result


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_markdown(text: str, rel_path: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Markdown/Confluence: сначала бьём документ по заголовкам, затем каждую
    секцию — recursive character splitting. К чанку приклеиваем «хлебные крошки»
    заголовков — для выгрузок Confluence это заметно поднимает релевантность.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[tuple[int, str]] = []
    current: list[str] = []
    current_trail = ""
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in current):
            sections.append((current_trail, list(current)))
        current.clear()

    for line in lines:
        if _MD_FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue
        heading = None if in_fence else _MD_HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_trail = " > ".join(item[1] for item in heading_stack)
            current.append(line)
        else:
            current.append(line)
    flush()

    if not sections:
        sections = [("", lines)]

    chunks: list[tuple[str, dict[str, Any]]] = []
    for trail, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for part in recursive_split(body):
            header = f"# file: {rel_path}\n"
            if trail:
                header += f"# section: {trail}\n"
            header += "\n"
            chunks.append((header + part, {"symbol": trail, "package": ""}))
    return chunks


def split_confluence_markdown(
    content: str, title: str, ancestors: Sequence[str]
) -> list[tuple[str, dict[str, Any]]]:
    """Split one Confluence page and retain its human-readable location."""
    if not content.strip():
        return []

    breadcrumb = " > ".join(item.strip() for item in ancestors if item.strip())
    chunks = split_markdown(content, title)
    result: list[tuple[str, dict[str, Any]]] = []
    for chunk, metadata in chunks:
        header = f"Confluence: {title}"
        if breadcrumb:
            header += f"\nРаздел: {breadcrumb}"
        item = dict(metadata)
        item["title"] = title
        item["ancestor_path"] = breadcrumb
        result.append((f"{header}\n\n{chunk}", item))
    return result


def split_plain_text(text: str, rel_path: str) -> list[tuple[str, dict[str, Any]]]:
    """Обычный .txt — только recursive character splitting."""
    return [
        (f"# file: {rel_path}\n\n{part}", {"symbol": "", "package": ""})
        for part in recursive_split(text)
    ]


def chunk_file(path: Path, rel_path: str, text: str) -> list[tuple[str, dict[str, Any]]]:
    """Выбирает сплиттер по расширению файла."""
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return split_go(text, rel_path)
    if suffix in (".md", ".markdown"):
        return split_markdown(text, rel_path)
    return split_plain_text(text, rel_path)


def classify(path: Path) -> tuple[str, str]:
    """Возвращает (source_type, language) для файла."""
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return SOURCE_CODE, "go"
    if suffix in (".md", ".markdown"):
        return SOURCE_CONFLUENCE, "markdown"
    return SOURCE_CONFLUENCE, "text"


# --------------------------------------------------------------------------- #
# 6. Эмбеддинги                                                                #
# --------------------------------------------------------------------------- #

def _embedding_dimension(model: Any) -> int:
    """В sentence-transformers 6.x метод переименован — поддерживаем оба имени."""
    for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attribute, None)
        if callable(getter):
            return int(getter())
    return 0


class Embedder:
    """Ленивая обёртка над SentenceTransformer. Модель грузится один раз."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return _embedding_dimension(self.model)

    @property
    def model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            started = time.perf_counter()
            log.info("Загружаю модель эмбеддингов %s (CPU)...", self.model_name)
            from sentence_transformers import SentenceTransformer  # тяжёлый импорт

            # Сначала пробуем строго из локального кэша: установщик уже скачал
            # модель, так что это и быстрее, и работает без сети (VPN, офлайн).
            # Если кэша нет — идём в Hugging Face.
            try:
                self._model = SentenceTransformer(
                    self.model_name, device="cpu", local_files_only=True
                )
            except Exception as error:  # noqa: BLE001 — нет кэша или старый API
                log.info("Локального кэша модели нет (%s), скачиваю с Hugging Face...",
                         type(error).__name__)
                self._model = SentenceTransformer(self.model_name, device="cpu")

            log.info(
                "Модель готова за %.1f c, размерность %d",
                time.perf_counter() - started,
                _embedding_dimension(self._model),
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Возвращает список нормализованных векторов (cosine == dot product)."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=EMBED_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]


# --------------------------------------------------------------------------- #
# 7. Хранилище: Chroma + BM25 + манифест хэшей                                 #
# --------------------------------------------------------------------------- #

@dataclass
class SearchHit:
    """Одна строка выдачи после слияния."""

    chunk_id: str
    document: str
    metadata: dict[str, Any]
    rrf_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None


@dataclass
class IndexReport:
    """Итоги прохода индексации — из этого собирается ответ инструмента."""

    root: str = ""
    scanned: int = 0
    code_files: int = 0
    doc_files: int = 0
    new_files: int = 0
    modified_files: int = 0
    unchanged_files: int = 0
    removed_files: int = 0
    skipped_files: list[str] = field(default_factory=list)
    chunks_added: int = 0
    chunks_deleted: int = 0
    total_chunks: int = 0
    elapsed: float = 0.0


@dataclass
class PageIndexReport:
    """Result of indexing one Confluence page."""

    status: str
    chunks_added: int = 0
    chunks_deleted: int = 0
    total_chunks: int = 0
    elapsed: float = 0.0


class KnowledgeBase:
    """
    Вся работа с диском и поиском. Методы блокирующие: MCP-слой вызывает их
    через asyncio.to_thread под общей блокировкой, чтобы не морозить event loop.
    """

    def __init__(self, base_dir: Path, model_name: str = EMBEDDING_MODEL) -> None:
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.chroma_dir = self.data_dir / "chroma"
        self.bm25_path = self.data_dir / "bm25.pkl"
        self.manifest_path = self.data_dir / "manifest.json"
        self.models_dir = base_dir / "models"
        self.model_name = model_name

        self._client: Any = None
        self._collection: Any = None
        self._embedder = Embedder(model_name, self.models_dir)

        # Состояние BM25: параллельные списки id и токенов + сам индекс.
        self._bm25: Any = None
        self._bm25_ids: list[str] = []
        self._bm25_tokens: list[list[str]] = []
        self._bm25_loaded = False

        self._manifest: dict[str, Any] | None = None

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Chroma -------------------------------- #

    @property
    def collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.config import Settings

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Разные ветки chromadb принимают конфиг HNSW по-разному —
        # пробуем от нового API к старому и в конце к «голому» варианту.
        attempts: list[dict[str, Any]] = [
            {"embedding_function": None, "configuration": {"hnsw": {"space": "cosine"}}},
            {"embedding_function": None, "metadata": {"hnsw:space": "cosine"}},
            {"embedding_function": None},
            {},
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME, **kwargs
                )
                break
            except Exception as error:  # noqa: BLE001 — перебираем варианты API
                last_error = error
        if self._collection is None:
            raise RuntimeError(f"Не удалось открыть коллекцию ChromaDB: {last_error}")
        return self._collection

    def _max_batch(self) -> int:
        getter = getattr(self._client, "get_max_batch_size", None)
        if callable(getter):
            try:
                return max(1, min(CHROMA_BATCH, int(getter())))
            except Exception:  # noqa: BLE001
                pass
        return CHROMA_BATCH

    # ---------------------------- Манифест ------------------------------- #

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path.exists():
            try:
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("files"), dict):
                    self._manifest = raw
                    self._manifest.setdefault("model", self.model_name)
                    self._manifest.setdefault("version", MANIFEST_VERSION)
                    return self._manifest
                log.warning("Манифест повреждён, начинаю с пустого")
            except (OSError, json.JSONDecodeError) as error:
                log.warning("Не смог прочитать манифест (%s), начинаю с пустого", error)
        self._manifest = {
            "version": MANIFEST_VERSION,
            "model": self.model_name,
            "files": {},
        }
        return self._manifest

    def _save_manifest(self) -> None:
        manifest = self.manifest
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write_text(
            self.manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2)
        )

    def _check_model_compatibility(self) -> None:
        """Смена модели меняет размерность векторов — молча смешивать нельзя."""
        recorded = self.manifest.get("model")
        if recorded and recorded != self.model_name and self.manifest["files"]:
            raise RuntimeError(
                f"Индекс собран моделью '{recorded}', а сервер запущен с "
                f"'{self.model_name}'. Выполните clear_index() и переиндексируйте."
            )
        self.manifest["model"] = self.model_name

    # ------------------------------ BM25 --------------------------------- #

    def _load_bm25(self) -> None:
        if self._bm25_loaded:
            return
        self._bm25_loaded = True
        if not self.bm25_path.exists():
            return
        try:
            with self.bm25_path.open("rb") as handle:
                state = pickle.load(handle)
        except Exception as error:  # noqa: BLE001 — битый pickle не должен ронять сервер
            log.warning("Не смог прочитать %s (%s) — BM25 будет пересобран",
                        self.bm25_path.name, error)
            return

        if not isinstance(state, dict) or state.get("version") != BM25_STATE_VERSION:
            log.warning("Версия состояния BM25 не совпала — пересоберу при индексации")
            return

        self._bm25_ids = list(state.get("ids", []))
        self._bm25_tokens = [list(item) for item in state.get("tokens", [])]
        self._bm25 = state.get("bm25")
        if self._bm25 is None and self._bm25_ids:
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        if not self._bm25_ids:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        corpus = [tokens if tokens else ["__empty__"] for tokens in self._bm25_tokens]
        self._bm25 = BM25Okapi(corpus)

    def _save_bm25(self) -> None:
        state = {
            "version": BM25_STATE_VERSION,
            "model": self.model_name,
            "ids": self._bm25_ids,
            "tokens": self._bm25_tokens,
            "bm25": self._bm25,
        }
        tmp_path = self.bm25_path.with_suffix(".pkl.tmp")
        try:
            with tmp_path.open("wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.bm25_path)
        except Exception:
            # BM25Okapi обычно пиклится без проблем, но если нет — сохраним
            # хотя бы токены, индекс пересоберётся при загрузке.
            state["bm25"] = None
            with tmp_path.open("wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.bm25_path)

    # --------------------------- Индексация ------------------------------ #

    def index_confluence_page(
        self,
        page_id: str,
        url: str,
        title: str,
        content: str,
        version: str | int | None = None,
        ancestors: Sequence[str] = (),
    ) -> PageIndexReport:
        """Persist one Confluence page, replacing only chunks with its id."""
        started = time.perf_counter()
        page_id, url, title = page_id.strip(), url.strip(), title.strip()
        if not page_id or not url or not title or not isinstance(content, str):
            raise ValueError("page_id, url, title и content обязательны")
        if not all(isinstance(item, str) and item.strip() for item in ancestors):
            raise ValueError("ancestors должен содержать непустые строки")

        self._check_model_compatibility()
        self._load_bm25()
        pages = self.manifest.setdefault("pages", {})
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        version_text = "" if version is None else str(version)
        known = pages.get(page_id)
        details = {
            "sha256": digest,
            "version": version_text,
            "url": url,
            "title": title,
            "ancestors": list(ancestors),
        }
        if known and all(known.get(key) == value for key, value in details.items()):
            return PageIndexReport("unchanged", total_chunks=self._count_chunks(), elapsed=time.perf_counter() - started)

        stale_ids = list(known.get("chunks", [])) if known else []
        if stale_ids:
            self._delete_chunks(stale_ids)
        pieces = split_confluence_markdown(content, title, ancestors)
        chunk_ids: list[str] = []
        if pieces:
            prefix = hashlib.sha1(page_id.encode("utf-8")).hexdigest()[:16]
            chunk_ids = [f"{prefix}-{index:05d}" for index in range(len(pieces))]
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._add_chunks(
                chunk_ids,
                [text for text, _extra in pieces],
                [
                    {
                        "page_id": page_id,
                        "url": url,
                        "title": title,
                        "ancestor_path": extra["ancestor_path"],
                        "heading_path": extra.get("symbol", ""),
                        "chunk_index": index,
                        "n_chunks": len(pieces),
                        "sha256": digest,
                        "indexed_at": timestamp,
                    }
                    for index, (_text, extra) in enumerate(pieces)
                ],
            )
        pages[page_id] = details | {"chunks": chunk_ids}
        self._rebuild_bm25()
        self._save_bm25()
        self._save_manifest()
        return PageIndexReport(
            "empty" if not pieces else "updated" if known else "indexed",
            chunks_added=len(chunk_ids),
            chunks_deleted=len(stale_ids),
            total_chunks=self._count_chunks(),
            elapsed=time.perf_counter() - started,
        )

    def iter_candidates(self, root: Path) -> Iterator[Path]:
        """Обходит дерево, отсекая служебные каталоги и слишком крупные файлы."""
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in SKIP_DIRS and not name.startswith(".")
            )
            for filename in sorted(filenames):
                candidate = Path(dirpath) / filename
                if candidate.suffix.lower() in INDEXABLE_EXTENSIONS:
                    yield candidate

    def index_directory(self, raw_path: str) -> IndexReport:
        started = time.perf_counter()
        root = Path(raw_path).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        else:
            root = root.resolve()

        if not root.exists():
            raise ValueError(f"Путь не существует: {root}")
        if not root.is_dir():
            raise ValueError(f"Путь не является каталогом: {root}")

        self._check_model_compatibility()
        self._load_bm25()

        report = IndexReport(root=str(root))
        files = self.manifest["files"]

        pending: list[tuple[Path, str, str, int]] = []  # (путь, md5, sha256, size)
        seen: set[str] = set()

        for path in self.iter_candidates(root):
            key = str(path)
            seen.add(key)
            report.scanned += 1
            source_type, _language = classify(path)
            if source_type == SOURCE_CODE:
                report.code_files += 1
            else:
                report.doc_files += 1

            try:
                stat = path.stat()
            except OSError as error:
                report.skipped_files.append(f"{path}: {error}")
                continue
            if stat.st_size > MAX_FILE_BYTES:
                report.skipped_files.append(f"{path}: больше {MAX_FILE_BYTES // 1024} КБ")
                continue

            try:
                md5, sha256, size = file_digests(path)
            except OSError as error:
                report.skipped_files.append(f"{path}: {error}")
                continue

            known = files.get(key)
            if known and known.get("sha256") == sha256 and known.get("md5") == md5:
                report.unchanged_files += 1
                continue
            if known:
                report.modified_files += 1
            else:
                report.new_files += 1
            pending.append((path, md5, sha256, size))

        # Файлы, исчезнувшие из дерева (или переставшие подходить по расширению).
        prefix = str(root) + os.sep
        removed = [
            key for key in list(files)
            if key.startswith(prefix) and key not in seen
        ]
        report.removed_files = len(removed)

        stale_ids: list[str] = []
        for key in removed:
            stale_ids.extend(files[key].get("chunks", []))
            files.pop(key, None)
        for path, _md5, _sha256, _size in pending:
            known = files.get(str(path))
            if known:
                stale_ids.extend(known.get("chunks", []))

        if stale_ids:
            self._delete_chunks(stale_ids)
            report.chunks_deleted = len(stale_ids)

        if not pending:
            if stale_ids:
                self._rebuild_bm25()
                self._save_bm25()
                self._save_manifest()
            report.total_chunks = self._count_chunks()
            report.elapsed = time.perf_counter() - started
            return report

        log.info("Индексирую %d файлов из %s", len(pending), root)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict[str, Any]] = []

        for number, (path, md5, sha256, size) in enumerate(pending, start=1):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as error:
                report.skipped_files.append(f"{path}: {error}")
                continue

            try:
                rel_path = str(path.relative_to(root))
            except ValueError:
                rel_path = path.name

            pieces = chunk_file(path, rel_path, text)
            if not pieces:
                files[str(path)] = {
                    "md5": md5, "sha256": sha256, "size": size,
                    "chunks": [], "source_type": classify(path)[0],
                    "indexed_at": timestamp,
                }
                continue

            source_type, language = classify(path)
            prefix_id = path_key(path)
            chunk_ids: list[str] = []
            for position, (chunk_text, extra) in enumerate(pieces):
                chunk_id = f"{prefix_id}-{position:05d}"
                chunk_ids.append(chunk_id)
                batch_ids.append(chunk_id)
                batch_docs.append(chunk_text)
                batch_metas.append({
                    "path": str(path),
                    "rel_path": rel_path,
                    "root": str(root),
                    "source_type": source_type,
                    "language": language,
                    "symbol": str(extra.get("symbol", ""))[:200],
                    "package": str(extra.get("package", ""))[:120],
                    "chunk_index": position,
                    "n_chunks": len(pieces),
                    "sha256": sha256,
                    "indexed_at": timestamp,
                })

            files[str(path)] = {
                "md5": md5,
                "sha256": sha256,
                "size": size,
                "chunks": chunk_ids,
                "source_type": source_type,
                "indexed_at": timestamp,
            }

            if number % 25 == 0 or number == len(pending):
                log.info("  обработано файлов: %d/%d", number, len(pending))

        if batch_ids:
            self._add_chunks(batch_ids, batch_docs, batch_metas)
            report.chunks_added = len(batch_ids)

        self._rebuild_bm25()
        self._save_bm25()
        self._save_manifest()

        report.total_chunks = self._count_chunks()
        report.elapsed = time.perf_counter() - started
        return report

    def _add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        log.info("Считаю эмбеддинги для %d чанков...", len(ids))
        embeddings = self._embedder.encode(documents)

        batch_size = self._max_batch()
        collection = self.collection
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            collection.upsert(
                ids=ids[start:stop],
                documents=documents[start:stop],
                metadatas=metadatas[start:stop],
                embeddings=embeddings[start:stop],
            )
            log.info("  записано в ChromaDB: %d/%d", min(stop, len(ids)), len(ids))

        # BM25: убираем возможные дубликаты id и дописываем новые записи.
        incoming = set(ids)
        if incoming & set(self._bm25_ids):
            keep = [
                position for position, existing in enumerate(self._bm25_ids)
                if existing not in incoming
            ]
            self._bm25_ids = [self._bm25_ids[position] for position in keep]
            self._bm25_tokens = [self._bm25_tokens[position] for position in keep]
        self._bm25_ids.extend(ids)
        self._bm25_tokens.extend(tokenize(document) for document in documents)

    def _delete_chunks(self, ids: Sequence[str]) -> None:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return
        collection = self.collection
        for start in range(0, len(unique), CHROMA_BATCH):
            try:
                collection.delete(ids=unique[start:start + CHROMA_BATCH])
            except Exception as error:  # noqa: BLE001
                log.warning("Не смог удалить часть чанков из ChromaDB: %s", error)

        drop = set(unique)
        keep = [
            position for position, existing in enumerate(self._bm25_ids)
            if existing not in drop
        ]
        self._bm25_ids = [self._bm25_ids[position] for position in keep]
        self._bm25_tokens = [self._bm25_tokens[position] for position in keep]

    def _count_chunks(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:  # noqa: BLE001
            return len(self._bm25_ids)

    # ------------------------------ Поиск -------------------------------- #

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Пустой запрос")
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        self._load_bm25()
        if self._count_chunks() == 0:
            return []

        candidates = max(MIN_CANDIDATES, min(top_k * CANDIDATE_MULTIPLIER, MAX_CANDIDATES))

        vector_ids, vector_scores = self._vector_search(query, candidates)
        bm25_ids, bm25_scores = self._bm25_search(query, candidates)

        # Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank(d))) по всем спискам.
        fused: dict[str, SearchHit] = {}
        for rank, chunk_id in enumerate(vector_ids, start=1):
            hit = fused.setdefault(chunk_id, SearchHit(chunk_id, "", {}))
            hit.rrf_score += 1.0 / (RRF_K + rank)
            hit.vector_rank = rank
            hit.vector_score = vector_scores.get(chunk_id)
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            hit = fused.setdefault(chunk_id, SearchHit(chunk_id, "", {}))
            hit.rrf_score += 1.0 / (RRF_K + rank)
            hit.bm25_rank = rank
            hit.bm25_score = bm25_scores.get(chunk_id)

        ordered = sorted(
            fused.values(),
            key=lambda item: (-item.rrf_score, item.vector_rank or 10**6),
        )[:top_k]
        if not ordered:
            return []

        payload = self._fetch_documents([hit.chunk_id for hit in ordered])
        result: list[SearchHit] = []
        for hit in ordered:
            document, metadata = payload.get(hit.chunk_id, ("", {}))
            if not document and not metadata:
                continue  # чанк исчез из Chroma — пропускаем «висячую» ссылку
            hit.document = document
            hit.metadata = metadata
            result.append(hit)
        return result

    def _vector_search(self, query: str, limit: int) -> tuple[list[str], dict[str, float]]:
        vector = self._embedder.encode([query])[0]
        try:
            response = self.collection.query(
                query_embeddings=[vector],
                n_results=limit,
                include=["distances"],
            )
        except Exception as error:  # noqa: BLE001
            log.warning("Векторный поиск не удался: %s", error)
            return [], {}

        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        scores: dict[str, float] = {}
        for position, chunk_id in enumerate(ids):
            if position < len(distances) and distances[position] is not None:
                # cosine distance → similarity
                scores[chunk_id] = 1.0 - float(distances[position])
        return list(ids), scores

    def _bm25_search(self, query: str, limit: int) -> tuple[list[str], dict[str, float]]:
        if self._bm25 is None or not self._bm25_ids:
            return [], {}
        tokens = tokenize(query)
        if not tokens:
            return [], {}
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as error:  # noqa: BLE001
            log.warning("BM25-поиск не удался: %s", error)
            return [], {}

        best = heapq.nlargest(
            min(limit, len(self._bm25_ids)),
            range(len(self._bm25_ids)),
            key=lambda position: float(scores[position]),
        )
        ids: list[str] = []
        mapping: dict[str, float] = {}
        for position in best:
            value = float(scores[position])
            if value <= 0.0:
                continue  # нулевой BM25 = совпадений нет, в фьюжн такое не пускаем
            chunk_id = self._bm25_ids[position]
            ids.append(chunk_id)
            mapping[chunk_id] = value
        return ids, mapping

    def _fetch_documents(
        self, ids: Sequence[str]
    ) -> dict[str, tuple[str, dict[str, Any]]]:
        if not ids:
            return {}
        try:
            response = self.collection.get(
                ids=list(ids), include=["documents", "metadatas"]
            )
        except Exception as error:  # noqa: BLE001
            log.warning("Не смог получить документы из ChromaDB: %s", error)
            return {}

        found_ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        payload: dict[str, tuple[str, dict[str, Any]]] = {}
        for position, chunk_id in enumerate(found_ids):
            document = documents[position] if position < len(documents) else ""
            metadata = metadatas[position] if position < len(metadatas) else {}
            payload[chunk_id] = (document or "", dict(metadata or {}))
        return payload

    # ------------------------------ Очистка ------------------------------ #

    def _release_chroma(self) -> None:
        """
        Полностью отпускает ChromaDB перед удалением файлов с диска.

        Chroma кэширует «систему» глобально по пути к каталогу. Если просто
        удалить каталог, следующий PersistentClient вернёт закэшированный
        объект, держащий уже несуществующий sqlite, и первая же запись упадёт
        с «attempt to write a readonly database». Поэтому: сначала штатный
        reset(), затем сброс глобального кэша систем.
        """
        if self._client is not None:
            try:
                self._client.reset()
            except Exception as error:  # noqa: BLE001 — reset может быть запрещён
                log.debug("chroma reset: %s", error)

        self._collection = None
        self._client = None

        # Путь к SharedSystemClient переезжал между версиями chromadb.
        for module_path in ("chromadb.api.shared_system_client", "chromadb.api.client"):
            try:
                module = importlib.import_module(module_path)
            except Exception:  # noqa: BLE001
                continue
            shared = getattr(module, "SharedSystemClient", None)
            clear_cache = getattr(shared, "clear_system_cache", None)
            if callable(clear_cache):
                try:
                    clear_cache()
                    return
                except Exception as error:  # noqa: BLE001
                    log.debug("clear_system_cache: %s", error)
        log.warning(
            "Не нашёл SharedSystemClient.clear_system_cache — если после очистки "
            "индексация не заработает, перезапустите сервер."
        )

    def clear(self) -> dict[str, Any]:
        before = 0
        try:
            before = self._count_chunks()
        except Exception:  # noqa: BLE001
            pass
        files_before = len(self.manifest.get("files", {}))

        self._release_chroma()

        self._bm25 = None
        self._bm25_ids = []
        self._bm25_tokens = []
        self._bm25_loaded = True
        self._manifest = {"version": MANIFEST_VERSION, "model": self.model_name, "files": {}}

        # Защита от катастрофы: удаляем только внутри своего base_dir.
        resolved = self.data_dir.resolve()
        if resolved.parent != self.base_dir.resolve():
            raise RuntimeError(f"Отказ удалять каталог вне base_dir: {resolved}")

        removed_paths: list[str] = []
        for target in (self.chroma_dir, self.bm25_path, self.manifest_path):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed_paths.append(str(target))
            elif target.exists():
                target.unlink(missing_ok=True)
                removed_paths.append(str(target))

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._save_manifest()

        return {
            "chunks_removed": before,
            "files_forgotten": files_before,
            "paths": removed_paths,
        }

    # ------------------------------ Статус ------------------------------- #

    def stats(self) -> dict[str, Any]:
        self._load_bm25()
        files = self.manifest.get("files", {})
        code = sum(1 for item in files.values() if item.get("source_type") == SOURCE_CODE)
        return {
            "chunks": self._count_chunks(),
            "files": len(files),
            "code_files": code,
            "doc_files": len(files) - code,
            "bm25_docs": len(self._bm25_ids),
            "model": self.model_name,
            "base_dir": str(self.base_dir),
        }


def _atomic_write_text(path: Path, content: str) -> None:
    """Пишем через временный файл + os.replace — не оставляем полуфайлов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------- #
# 8. Форматирование ответов                                                    #
# --------------------------------------------------------------------------- #

def format_index_report(report: IndexReport) -> str:
    lines = [
        f"Индексация завершена: {report.root}",
        "",
        f"Просмотрено файлов : {report.scanned} "
        f"(Go: {report.code_files}, документация: {report.doc_files})",
        f"Новых              : {report.new_files}",
        f"Изменённых         : {report.modified_files}",
        f"Без изменений      : {report.unchanged_files} (пропущены по хэшу MD5+SHA256)",
        f"Удалённых из индекса: {report.removed_files}",
        "",
        f"Добавлено чанков   : {report.chunks_added}",
        f"Удалено чанков     : {report.chunks_deleted}",
        f"Всего в индексе    : {report.total_chunks}",
        f"Время              : {report.elapsed:.1f} c",
    ]
    if report.skipped_files:
        lines.append("")
        lines.append(f"Пропущено файлов: {len(report.skipped_files)}")
        for item in report.skipped_files[:10]:
            lines.append(f"  - {item}")
        if len(report.skipped_files) > 10:
            lines.append(f"  ... и ещё {len(report.skipped_files) - 10}")
    return "\n".join(lines)


def _snippet(document: str, limit: int = SNIPPET_CHARS) -> str:
    text = document.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... (обрезано)"


def format_search_results(query: str, hits: Sequence[SearchHit], top_k: int) -> str:
    if not hits:
        return (
            f"По запросу «{query}» ничего не найдено.\n\n"
            "Проверьте, что каталог проиндексирован: index_directory(path=\"/путь/к/проекту\")."
        )

    blocks = [
        f"Гибридный поиск (векторы + BM25, слияние RRF k={RRF_K}) по запросу: «{query}»",
        f"Показано {len(hits)} из запрошенных {top_k}.",
        "",
    ]
    for position, hit in enumerate(hits, start=1):
        metadata = hit.metadata
        title = metadata.get("title") or "?"
        url = metadata.get("url") or "?"
        symbol = metadata.get("heading_path") or metadata.get("symbol") or ""

        signals = []
        if hit.vector_rank:
            score = f" sim={hit.vector_score:.3f}" if hit.vector_score is not None else ""
            signals.append(f"вектор #{hit.vector_rank}{score}")
        if hit.bm25_rank:
            score = f" bm25={hit.bm25_score:.2f}" if hit.bm25_score is not None else ""
            signals.append(f"BM25 #{hit.bm25_rank}{score}")

        header = f"## {position}. {title}"
        if symbol:
            header += f" — {symbol}"
        blocks.append(header)
        blocks.append(f"Страница: {url}")
        blocks.append(f"RRF: {hit.rrf_score:.5f} | " + " | ".join(signals or ["—"]))
        blocks.append("```markdown")
        blocks.append(_snippet(hit.document))
        blocks.append("```")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def format_clear_result(result: dict[str, Any], stats: dict[str, Any]) -> str:
    lines = [
        "Индекс полностью очищен.",
        "",
        f"Удалено чанков       : {result['chunks_removed']}",
        f"Забыто файлов (хэши) : {result['files_forgotten']}",
        "",
        "Удалено с диска:",
    ]
    lines.extend(f"  - {item}" for item in result["paths"] or ["(нечего было удалять)"])
    lines.append("")
    lines.append(f"Каталог данных пересоздан: {stats['base_dir']}/data")
    lines.append("Запустите index_directory(path=...) заново, чтобы наполнить базу.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 9. MCP-сервер                                                                #
# --------------------------------------------------------------------------- #

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "index_confluence_page",
        "description": "Индексировать одну страницу, уже полученную агентом через Confluence MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"}, "url": {"type": "string"},
                "title": {"type": "string"}, "content": {"type": "string"},
                "version": {"type": ["string", "integer"]},
                "ancestors": {"type": "array", "items": {"type": "string"}, "default": []},
            },
            "required": ["page_id", "url", "title", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_confluence",
        "description": "Гибридный поиск по проиндексированным страницам Confluence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Поисковый запрос: вопрос на естественном языке, имя функции "
                        "или типа, фрагмент сообщения об ошибке."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Сколько фрагментов вернуть (1..50).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "clear_index",
        "description": (
            "Полностью удалить индекс: коллекцию ChromaDB, состояние BM25 и кэш "
            "хэшей файлов. Каталог данных на диске очищается и пересоздаётся. "
            "Нужен при смене модели эмбеддингов или для сборки базы с нуля."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


SERVER_INSTRUCTIONS = (
    "Локальная база знаний по командному Confluence. Для поиска используйте "
    "search_confluence. Страницы сначала получите через Confluence MCP, затем "
    "передайте их в index_confluence_page. clear_index вызывайте только по "
    "однозначной просьбе пользователя удалить индекс."
)


def build_server(knowledge_base: KnowledgeBase) -> Any:
    """Собирает low-level MCP Server поверх переданного хранилища."""
    import mcp.types as types
    from mcp.server.lowlevel import Server

    try:
        server = Server(
            SERVER_NAME, version=SERVER_VERSION, instructions=SERVER_INSTRUCTIONS
        )
    except TypeError:  # очень старая ветка SDK без этих параметров
        server = Server(SERVER_NAME)
    # Chroma, BM25 и модель не рассчитаны на конкурентную запись —
    # сериализуем все обращения к хранилищу.
    gate = asyncio.Lock()

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["inputSchema"],
            )
            for item in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        arguments = arguments or {}
        log.info("Вызов инструмента %s(%s)", name, ", ".join(sorted(arguments)))

        if name == "index_confluence_page":
            page_id = arguments.get("page_id")
            url = arguments.get("url")
            title = arguments.get("title")
            content = arguments.get("content")
            ancestors = arguments.get("ancestors", [])
            if not all(isinstance(value, str) and value.strip() for value in (page_id, url, title)):
                raise ValueError("page_id, url и title обязательны")
            if not isinstance(content, str) or not isinstance(ancestors, list):
                raise ValueError("content должен быть строкой, ancestors — массивом")
            async with gate:
                report = await asyncio.to_thread(
                    knowledge_base.index_confluence_page,
                    page_id, url, title, content, arguments.get("version"), ancestors,
                )
            return [types.TextContent(type="text", text=f"Страница {report.status}: +{report.chunks_added} чанков, всего {report.total_chunks}.")]

        if name == "search_confluence":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Параметр 'query' обязателен и должен быть строкой")
            raw_top_k = arguments.get("top_k", 5)
            try:
                top_k = int(raw_top_k)
            except (TypeError, ValueError):
                raise ValueError(f"'top_k' должен быть целым числом, получено: {raw_top_k!r}")
            async with gate:
                hits = await asyncio.to_thread(knowledge_base.search, query, top_k)
            return [
                types.TextContent(
                    type="text", text=format_search_results(query, hits, top_k)
                )
            ]

        if name == "clear_index":
            async with gate:
                result = await asyncio.to_thread(knowledge_base.clear)
                stats = await asyncio.to_thread(knowledge_base.stats)
            return [types.TextContent(type="text", text=format_clear_result(result, stats))]

        raise ValueError(f"Неизвестный инструмент: {name}")

    return server


def _protocol_stdout() -> Any:
    """
    Возвращает текстовый поток поверх сохранённого «настоящего» stdout.
    Именно он и только он отдаётся stdio-транспорту MCP.
    """
    if _PROTOCOL_STDOUT_FD is None:
        return None
    raw = os.fdopen(_PROTOCOL_STDOUT_FD, "wb", buffering=0, closefd=False)
    return io.TextIOWrapper(raw, encoding="utf-8", write_through=True, newline="")


async def run_server(base_dir: Path) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    knowledge_base = KnowledgeBase(base_dir)
    server = build_server(knowledge_base)

    log.info("%s v%s — база: %s", SERVER_NAME, SERVER_VERSION, base_dir)
    log.info("Модель: %s | HF_HOME: %s", EMBEDDING_MODEL, os.environ.get("HF_HOME", "-"))

    stream = _protocol_stdout()
    context = None
    if stream is not None:
        try:
            context = stdio_server(stdout=anyio.wrap_file(stream))
        except TypeError:
            # Древняя/новая ветка SDK без kwargs — отдаём поток штатным путём.
            sys.stdout = stream
            context = None
    if context is None:
        context = stdio_server()

    async with context as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# --------------------------------------------------------------------------- #
# 10. Самопроверка (используется установщиком)                                 #
# --------------------------------------------------------------------------- #

SELFTEST_GO = '''package billing

import "fmt"

// Invoice описывает выставленный счёт клиенту.
type Invoice struct {
	ID     string
	Amount int64
}

// Repository хранит счета.
type Repository interface {
	Save(invoice Invoice) error
}

// CalculateVAT считает НДС по ставке 20 процентов.
func CalculateVAT(amount int64) int64 {
	return amount * 20 / 100
}

func (i *Invoice) String() string {
	return fmt.Sprintf("invoice %s for %d", i.ID, i.Amount)
}
'''

SELFTEST_MD = """# Биллинг

## Расчёт НДС

Ставка НДС в сервисе биллинга фиксирована и равна 20 процентам.
Расчёт выполняется на стороне Go-сервиса перед сохранением счёта.

## Хранение счетов

Счета сохраняются через интерфейс Repository.
"""


def selftest() -> int:
    """Прогон индексации → поиска → очистки на временных данных."""
    print("=== Самопроверка hybrid_knowledge_mcp ===", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="gckb-home-") as home, \
            tempfile.TemporaryDirectory(prefix="gckb-src-") as source:
        source_dir = Path(source)
        (source_dir / "internal").mkdir()
        (source_dir / "internal" / "billing.go").write_text(SELFTEST_GO, encoding="utf-8")
        (source_dir / "docs").mkdir()
        (source_dir / "docs" / "billing.md").write_text(SELFTEST_MD, encoding="utf-8")

        knowledge_base = KnowledgeBase(Path(home))

        report = knowledge_base.index_directory(str(source_dir))
        print(format_index_report(report), file=sys.stderr)
        assert report.chunks_added > 0, "индексация не создала ни одного чанка"
        assert report.new_files == 2, f"ожидалось 2 новых файла, получено {report.new_files}"

        again = knowledge_base.index_directory(str(source_dir))
        assert again.unchanged_files == 2, "повторная индексация не сработала по хэшам"
        assert again.chunks_added == 0, "повторная индексация добавила лишние чанки"
        print("Дедупликация по хэшам: OK", file=sys.stderr)

        hits = knowledge_base.search("как считается НДС", top_k=3)
        assert hits, "гибридный поиск ничего не вернул"
        print(format_search_results("как считается НДС", hits, 3), file=sys.stderr)

        sources = {hit.metadata.get("source_type") for hit in hits}
        symbol_hits = knowledge_base.search("CalculateVAT", top_k=3)
        assert symbol_hits, "поиск по имени функции ничего не вернул"
        assert any(
            "CalculateVAT" in hit.document for hit in symbol_hits
        ), "BM25 не нашёл точное совпадение по имени функции"
        print(f"Найденные типы источников: {sorted(s for s in sources if s)}", file=sys.stderr)

        result = knowledge_base.clear()
        print(format_clear_result(result, knowledge_base.stats()), file=sys.stderr)
        assert knowledge_base.stats()["chunks"] == 0, "после clear() индекс не пуст"

        # Регресс: chromadb кэширует систему по пути, и после физического
        # удаления каталога запись падала с «readonly database».
        after_clear = knowledge_base.index_directory(str(source_dir))
        assert after_clear.chunks_added == report.chunks_added, (
            f"после clear() переиндексация дала {after_clear.chunks_added} чанков "
            f"вместо {report.chunks_added}"
        )
        assert knowledge_base.search("CalculateVAT", top_k=3), (
            "после clear() и переиндексации поиск ничего не находит"
        )
        print("Цикл очистка → переиндексация → поиск: OK", file=sys.stderr)

        # Эмуляция перезапуска сервера: новый объект на том же каталоге.
        reopened = KnowledgeBase(Path(home))
        assert reopened.stats()["chunks"] == after_clear.total_chunks, (
            "после перезапуска число чанков в индексе не совпало"
        )
        assert reopened.search("Repository", top_k=3), (
            "после перезапуска индекс не читается с диска"
        )
        print("Переживание перезапуска сервера: OK", file=sys.stderr)

    print("=== Самопроверка пройдена ===", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# 11. Точка входа                                                              #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid_knowledge_mcp",
        description="MCP-сервер гибридного поиска по Go-коду и документации Confluence",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="каталог с индексами (по умолчанию ~/.local/share/go-confluence-mcp)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="прогнать индексацию/поиск/очистку на временных данных и выйти",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="скачать и прогреть модель эмбеддингов, затем выйти",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(base_dir / "models"))

    if args.selftest:
        return selftest()

    if args.warmup:
        embedder = Embedder(EMBEDDING_MODEL, base_dir / "models")
        vector = embedder.encode(["warmup"])[0]
        print(f"Модель прогрета, размерность вектора: {len(vector)}", file=sys.stderr)
        return 0

    try:
        asyncio.run(run_server(base_dir))
    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    return 0


if __name__ == "__main__":
    sys.exit(main())
