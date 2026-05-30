"""
Утилиты для post-processing результатов retrieval.
"""

import re
from difflib import SequenceMatcher
from typing import Any, List, Optional

_NEAR_DUP_RATIO = 0.92
_MIN_NEAR_DUP_LEN = 40


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_doc_text(text: str) -> str:
    """Нормализация текста для сравнения дублей."""
    return _collapse_whitespace(text.lower())


def extract_doc_text(doc: Any) -> str:
    """Извлечение основного текста из документа разных форматов."""
    if doc is None:
        return ""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "content", "page_content", "chunk"):
            value = doc.get(key)
            if value:
                return str(value)
        return ""
    page_content = getattr(doc, "page_content", None)
    if page_content:
        return str(page_content)
    text = getattr(doc, "text", None)
    if text:
        return str(text)
    return ""


def _is_near_duplicate(normalized: str, kept_normalized: List[str]) -> bool:
    if len(normalized) < _MIN_NEAR_DUP_LEN:
        return False
    for seen in kept_normalized:
        if len(seen) < _MIN_NEAR_DUP_LEN:
            continue
        if SequenceMatcher(None, normalized, seen).ratio() >= _NEAR_DUP_RATIO:
            return True
    return False


def deduplicate_context_docs(
    docs: List[Any],
    max_docs: Optional[int] = None,
) -> List[Any]:
    """
    Удаление точных и почти точных дублей с сохранением порядка релевантности.
    """
    if not docs:
        return []

    result: List[Any] = []
    seen_exact: set[str] = set()
    kept_normalized: List[str] = []

    for doc in docs:
        normalized = normalize_doc_text(extract_doc_text(doc))

        if normalized:
            if normalized in seen_exact:
                continue
            if _is_near_duplicate(normalized, kept_normalized):
                continue
            seen_exact.add(normalized)
            kept_normalized.append(normalized)

        result.append(doc)
        if max_docs is not None and len(result) >= max_docs:
            break

    return result
