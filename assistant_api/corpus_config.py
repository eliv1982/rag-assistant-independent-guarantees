"""
Конфигурация корпуса для RAG по независимым гарантиям.
Пути задаются относительно каталога assistant_api.

Отпечаток корпуса (CORPUS_ID) считается автоматически по содержимому файлов, метаданным
источников и параметрам индексации. Он входит в имя коллекции Chroma и в ключ кеша, поэтому
после изменения корпуса индекс пересобирается, а старые ответы из кеша не отдаются.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ASSISTANT_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSISTANT_DIR / "data"

# Увеличивайте при изменении логики нарезки или метаданных (chunking.py, chunk_splitter.py,
# vector_store.py), чтобы существующие индексы не использовались повторно.
# 2: нарезка с учётом структуры источника (статья/часть/пункт/позиция), узкие метаданные.
INDEX_SCHEMA_VERSION = 2

COLLECTION_PREFIX = "guarantees"
_CORPUS_ID_LEN = 12


def default_corpus_entries() -> List[Dict[str, Any]]:
    """
    Семь источников: ГК РФ (ст. 368–379), 44-ФЗ (ст. 45), 223-ФЗ (ст. 3.4),
    постановления Правительства РФ № 1005 и № 1397, два обзора практики ВС РФ.
    doc_type: statute (нормативный акт) или overview (обзор практики).
    chunker — стратегия нарезки по структуре источника (см. chunking.py), citation — как источник
    называется в узкой ссылке («44-ФЗ, ст. 45, ч. 6»). Подробности о составе корпуса: data/README.md.
    """
    return [
        {
            "path": DATA_DIR / "GK_368_379.txt",
            "source": "gk_rf_368_379",
            "source_display": "ГК РФ, ст. 368–379",
            "source_kind": "law",
            "doc_type": "statute",
            "chunker": "civil_code",
            "citation": "ГК РФ",
        },
        {
            "path": DATA_DIR / "44FZ_article_45.txt",
            "source": "fz_44_art_45",
            "source_display": "44-ФЗ, ст. 45",
            "source_kind": "procurement_law",
            "doc_type": "statute",
            "chunker": "federal_law",
            "citation": "44-ФЗ",
        },
        {
            "path": DATA_DIR / "223FZ_article_3_4_guarantees.txt",
            "source": "fz_223_art_3_4",
            "source_display": "223-ФЗ, ст. 3.4",
            "source_kind": "procurement_law",
            "doc_type": "statute",
            "chunker": "federal_law",
            "citation": "223-ФЗ",
        },
        {
            "path": DATA_DIR / "PP_1005.txt",
            "source": "pp_1005_2013",
            "source_display": "Постановление Правительства РФ № 1005",
            "source_kind": "government_resolution",
            "doc_type": "statute",
            "chunker": "government_resolution",
            "citation": "ПП РФ №1005",
        },
        {
            "path": DATA_DIR / "PP_1397.txt",
            "source": "pp_1397_2022",
            "source_display": "Постановление Правительства РФ № 1397",
            "source_kind": "government_resolution",
            "doc_type": "statute",
            "chunker": "government_resolution",
            "citation": "ПП РФ №1397",
        },
        {
            "path": DATA_DIR / "VS_independent_guarantee_2019.txt",
            "source": "vs_independent_guarantee_2019",
            "source_display": "Обзор практики ВС РФ по независимой гарантии (2019)",
            "source_kind": "case_law_summary",
            "doc_type": "overview",
            "chunker": "court_review",
            "citation": "Обзор ВС РФ от 05.06.2019",
        },
        {
            "path": DATA_DIR / "VS_contract_system_2017_guarantees.txt",
            "source": "vs_contract_system_2017",
            "source_display": "Обзор практики ВС РФ по контрактной системе (2017)",
            "source_kind": "case_law_summary",
            "doc_type": "overview",
            "chunker": "court_review",
            "citation": "Обзор ВС РФ от 28.06.2017",
        },
    ]


def single_file_entry(path: Path) -> Dict[str, Any]:
    """Корпус из одного произвольного файла (параметр data_file у RAGPipeline)."""
    return {
        "path": Path(path),
        "source": "single_file",
        "source_display": Path(path).name,
        "source_kind": "unknown",
        "doc_type": "overview",
    }


def index_settings() -> Dict[str, Any]:
    """Параметры индексации из окружения. От них зависит содержимое индекса."""
    return {
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        "min_chunk_len": int(os.getenv("RAG_MIN_CHUNK_LEN", "80")),
    }


def compute_corpus_id(
    entries: Optional[List[Dict[str, Any]]] = None,
    settings: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Path] = None,
) -> str:
    """
    Короткий детерминированный отпечаток корпуса.

    Входит: версия схемы индекса, RAG_CORPUS_VERSION (ручной «пересобрать всё»), параметры
    индексации, а для каждого источника — метаданные и содержимое файла. Пути в хеш не входят
    (только имя файла), переводы строк нормализуются: отпечаток не зависит от машины и ОС.
    """
    entries = default_corpus_entries() if entries is None else entries
    settings = index_settings() if settings is None else settings
    base = Path(base_dir) if base_dir is not None else ASSISTANT_DIR

    digest = hashlib.sha256()
    header = {
        "schema": INDEX_SCHEMA_VERSION,
        "corpus_version": os.getenv("RAG_CORPUS_VERSION", "1"),
        "settings": settings,
    }
    digest.update(json.dumps(header, sort_keys=True, ensure_ascii=False).encode("utf-8"))

    for entry in entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = (base / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Файл корпуса не найден: {path}")

        meta = {
            "file": path.name,
            "source": entry["source"],
            "source_display": entry["source_display"],
            "source_kind": entry.get("source_kind", "unknown"),
            "doc_type": entry.get("doc_type", "overview"),
            "chunker": entry.get("chunker", "generic"),
            "citation": entry.get("citation") or entry["source_display"],
        }
        digest.update(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))

    return digest.hexdigest()[:_CORPUS_ID_LEN]


def collection_name_for(corpus_id: str) -> str:
    """Имя коллекции Chroma для данного корпуса (допустимые символы: [a-z0-9_])."""
    return f"{COLLECTION_PREFIX}_{corpus_id}"
