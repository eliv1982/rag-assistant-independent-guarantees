"""
Общие helpers для CLI и web-слоя поверх RAG pipeline.
"""

import os
from typing import Any, Dict

import markdown
import nh3

try:
    from .chunking import source_label
    from .rag_pipeline import RAGPipeline
except ImportError:
    from chunking import source_label
    from rag_pipeline import RAGPipeline

_ALLOWED_TAGS = {
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
}
# Только href и title: target/rel не пропускаются (nh3 сам ставит rel="noopener noreferrer").
_ALLOWED_ATTRIBUTES = {
    "a": {"href", "title"},
}
# Явный список вместо умолчаний nh3 (в них есть ftp, ssh, irc и др.). javascript:, data:, vbscript: не входят.
_ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


def render_markdown_safe(text: str) -> str:
    """
    Преобразование Markdown в безопасный HTML для отображения в web UI.

    Markdown пропускает сырой HTML из ответа модели как есть, поэтому результат всегда проходит
    через nh3 по белому списку тегов, атрибутов и схем ссылок. Содержимое script/style удаляется
    целиком, относительные ссылки (включая //host) и обработчики событий отбрасываются.
    """
    if not text:
        return ""

    html = markdown.markdown(
        text,
        extensions=["extra", "nl2br", "sane_lists"],
    )
    return nh3.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        url_schemes=_ALLOWED_URL_SCHEMES,
        url_relative="deny",
        link_rel="noopener noreferrer",
        strip_comments=True,
    )


def create_rag_pipeline() -> RAGPipeline:
    """
    Создание RAG pipeline для CLI и web.
    Пути (RAG_CHROMA_PATH, RAG_CACHE_DB_PATH) и имя коллекции определяет сам RAGPipeline.
    """
    return RAGPipeline(model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"))


def interaction_log_fields(result: Any, pipeline: RAGPipeline) -> Dict[str, Any]:
    """Извлечение полей для DatabaseLogger из результата pipeline."""
    if isinstance(result, dict):
        context_docs = result.get("context_docs") or []
        sources_count = len(context_docs) if isinstance(context_docs, list) else None
        return {
            "response": str(result.get("answer", "")),
            "from_cache": bool(result.get("from_cache", False)),
            # Без вызова чат-модели (недостаточно данных в корпусе) модель в журнал не записывается.
            "model": None if result.get("no_evidence") else (result.get("model") or pipeline.model),
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
                    "label": source_label(meta) if meta else "",
                    "heading": (meta.get("section_heading") or "").strip(),
                }
            )
        else:
            sources.append({"text": str(doc), "label": "", "heading": ""})
    return sources
