"""
Общие helpers для CLI и web-слоя поверх RAG pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import bleach
import markdown

try:
    from .rag_pipeline import RAGPipeline
except ImportError:
    from rag_pipeline import RAGPipeline

_BASE_DIR = Path(__file__).resolve().parent

_ALLOWED_TAGS = [
    "p",
    "br",
    "strong",
    "em",
    "ul",
    "ol",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "blockquote",
    "code",
    "pre",
    "a",
]
_ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "target", "rel"],
}


def render_markdown_safe(text: str) -> str:
    """Преобразование Markdown в безопасный HTML для отображения в web UI."""
    if not text:
        return ""

    html = markdown.markdown(
        text,
        extensions=["extra", "nl2br", "sane_lists"],
    )
    cleaned = bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        strip=True,
    )
    return bleach.linkify(cleaned)


def create_rag_pipeline() -> RAGPipeline:
    """Создание RAG pipeline с теми же параметрами, что и в CLI."""
    return RAGPipeline(
        collection_name="api_rag_collection",
        cache_db_path=str(_BASE_DIR / "api_rag_cache.db"),
        persist_directory=os.getenv("RAG_CHROMA_PATH", str(_BASE_DIR / "chroma_db")),
        model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
    )


def interaction_log_fields(result: Any, pipeline: RAGPipeline) -> Dict[str, Any]:
    """Извлечение полей для DatabaseLogger из результата pipeline."""
    if isinstance(result, dict):
        context_docs = result.get("context_docs") or []
        sources_count = len(context_docs) if isinstance(context_docs, list) else None
        return {
            "response": str(result.get("answer", "")),
            "from_cache": bool(result.get("from_cache", False)),
            "model": result.get("model") or pipeline.model,
            "top_k": pipeline.top_k,
            "sources_count": sources_count,
        }

    return {
        "response": str(result),
        "from_cache": False,
        "model": pipeline.model,
        "top_k": pipeline.top_k,
        "sources_count": None,
    }


def normalize_sources(context_docs: Any) -> list:
    """Приведение context_docs к списку dict для шаблонов."""
    if not context_docs or not isinstance(context_docs, list):
        return []

    sources = []
    for doc in context_docs:
        if isinstance(doc, dict):
            meta = doc.get("metadata") or {}
            sources.append(
                {
                    "text": doc.get("text", ""),
                    "label": meta.get("source_display", ""),
                    "heading": (meta.get("section_heading") or "").strip(),
                }
            )
        else:
            sources.append({"text": str(doc), "label": "", "heading": ""})
    return sources
