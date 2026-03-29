"""
Конфигурация корпуса для RAG по независимым гарантиям.
Пути задаются относительно каталога assistant_api.
"""

from pathlib import Path
from typing import Any, Dict, List

ASSISTANT_DIR = Path(__file__).resolve().parent


def default_corpus_entries() -> List[Dict[str, Any]]:
    """
    Три источника: ГК РФ, URDG 2010, обзор практики ВС РФ.
    doc_type: statute — нарезка по статьям/параграфам; overview — смысловые чанки без статей.
    """
    return [
        {
            "path": ASSISTANT_DIR / "data" / "GK_clean.txt",
            "source": "gk_rf",
            "source_display": "ГК РФ",
            "source_kind": "law",
            "doc_type": "statute",
        },
        {
            "path": ASSISTANT_DIR / "data" / "URDG_clean.txt",
            "source": "urdg_2010",
            "source_display": "URDG 2010",
            "source_kind": "rules",
            "doc_type": "statute",
        },
        {
            "path": ASSISTANT_DIR / "data" / "Overview_clean.txt",
            "source": "vs_overview_2019",
            "source_display": "Обзор судебной практики ВС РФ",
            "source_kind": "case_law_summary",
            "doc_type": "overview",
        },
    ]
