"""
Web UI для RAG-ассистента по независимым гарантиям (FastAPI + Jinja2).
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from .db_logger import DatabaseLogger, get_logs_db_path
    from .openai_client import api_key_configured
    from .rag_helpers import (
        create_rag_pipeline,
        interaction_log_fields,
        normalize_sources,
        render_markdown_safe,
    )
    from .rag_pipeline import RAGPipeline
except ImportError:
    from db_logger import DatabaseLogger, get_logs_db_path
    from openai_client import api_key_configured
    from rag_helpers import (
        create_rag_pipeline,
        interaction_log_fields,
        normalize_sources,
        render_markdown_safe,
    )
    from rag_pipeline import RAGPipeline

_HERE = Path(__file__).resolve().parent
_env = _HERE.parent / ".env"
if _env.exists():
    load_dotenv(_env)
else:
    load_dotenv()

log = logging.getLogger("assistant_api.web")

app = FastAPI(title="RAG Assistant: Independent Guarantees")
templates = Jinja2Templates(directory=str(_HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

_pipeline: Optional[RAGPipeline] = None
_pipeline_lock = threading.Lock()
_logger: Optional[DatabaseLogger] = None

_ERROR_GENERIC = "Не удалось получить ответ. Попробуйте повторить запрос позже."
_ERROR_UNAVAILABLE = "Ассистент временно недоступен: сервис не настроен."
_STATS_UNAVAILABLE = "Журнал запросов временно недоступен."

_EMPTY_STATS: Dict[str, Any] = {
    "total_interactions": 0,
    "successful_interactions": 0,
    "failed_interactions": 0,
    "cache_hits": 0,
    "cache_hit_rate": 0.0,
    "average_response_time_ms": None,
}


class AssistantUnavailableError(RuntimeError):
    """Ассистент не настроен (например, нет OPENAI_API_KEY): запрос не может быть выполнен."""


def get_logger() -> DatabaseLogger:
    """Ленивая инициализация DatabaseLogger."""
    global _logger
    if _logger is None:
        _logger = DatabaseLogger(db_path=get_logs_db_path())
    return _logger


def get_pipeline() -> RAGPipeline:
    """
    Ленивая инициализация RAG pipeline (требует OPENAI_API_KEY).
    Под блокировкой: одновременные первые запросы не строят индекс дважды.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            if not api_key_configured():
                raise AssistantUnavailableError(
                    "OPENAI_API_KEY не установлен. Настройте переменную окружения для работы ассистента."
                )
            _pipeline = create_rag_pipeline()
        return _pipeline


def _log_interaction_safe(**kwargs: Any) -> None:
    """Журнал вторичен: сбой записи не должен ломать ответ пользователю."""
    try:
        get_logger().log_interaction(**kwargs)
    except Exception:
        log.exception("Не удалось записать взаимодействие в журнал")


def _log_error_safe(**kwargs: Any) -> None:
    """Сбой журнала не должен скрывать исходную ошибку запроса."""
    try:
        get_logger().log_error(**kwargs)
    except Exception:
        log.exception("Не удалось записать ошибку в журнал")


def _truncate(text: Any, max_len: int = 80) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "…"


def _index_context(
    request: Request,
    *,
    question: str = "",
    answer: Optional[str] = None,
    answer_html: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sources: Optional[List[Dict[str, str]]] = None,
) -> dict:
    return {
        "request": request,
        "question": question,
        "answer": answer,
        "answer_html": answer_html,
        "error": error,
        "metadata": metadata or {},
        "sources": sources or [],
    }


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        _index_context(request),
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(default="")):
    # Обычная (не async) функция: FastAPI выполняет её в пуле потоков, и блокирующие
    # вызовы (OpenAI, Chroma, SQLite) не останавливают event loop.
    question = question.strip()

    if not question:
        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(request, error="Пожалуйста, введите вопрос."),
            status_code=400,
        )

    start = time.perf_counter()
    try:
        pipeline = get_pipeline()
        result = pipeline.query(question)

        fields = interaction_log_fields(result, pipeline)
        context_docs = result.get("context_docs") if isinstance(result, dict) else None
        sources = normalize_sources(context_docs)
        answer_html = render_markdown_safe(fields["response"])
    except Exception as exc:
        response_time_ms = int((time.perf_counter() - start) * 1000)
        log.exception("Ошибка обработки вопроса")

        # Только уже созданный pipeline: повторная инициализация здесь могла бы заново
        # индексировать корпус (платные эмбеддинги) ради двух полей журнала.
        failed_pipeline = _pipeline
        _log_error_safe(
            query=question,
            error_message=f"{type(exc).__name__}: {exc}",
            response_time_ms=response_time_ms,
            model=failed_pipeline.model if failed_pipeline else None,
            top_k=failed_pipeline.top_k if failed_pipeline else None,
            interface="web",
        )

        unavailable = isinstance(exc, AssistantUnavailableError)
        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(
                request,
                question=question,
                error=_ERROR_UNAVAILABLE if unavailable else _ERROR_GENERIC,
            ),
            status_code=503 if unavailable else 500,
        )

    response_time_ms = int((time.perf_counter() - start) * 1000)
    _log_interaction_safe(
        query=question,
        response=fields["response"],
        from_cache=fields["from_cache"],
        response_time_ms=response_time_ms,
        model=fields["model"],
        top_k=fields["top_k"],
        sources_count=fields["sources_count"],
        interface="web",
    )

    return templates.TemplateResponse(
        request,
        "index.html",
        _index_context(
            request,
            question=question,
            answer=fields["response"],
            answer_html=answer_html,
            metadata={
                "from_cache": fields["from_cache"],
                "response_time_ms": response_time_ms,
                "model": fields["model"],
                "sources_count": fields["sources_count"],
            },
            sources=sources,
        ),
    )


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    try:
        logger = get_logger()
        stats_data = logger.get_stats()
        recent_raw = logger.get_recent(limit=10)
    except Exception:
        log.exception("Статистика недоступна")
        return templates.TemplateResponse(
            request,
            "stats.html",
            {
                "request": request,
                "stats": _EMPTY_STATS,
                "recent": [],
                "error": _STATS_UNAVAILABLE,
            },
            status_code=503,
        )

    recent = [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "query": _truncate(row["query"], 100),
            "response": _truncate(row["response"], 80),
            "from_cache": bool(row["from_cache"]),
            "response_time_ms": row["response_time_ms"],
            "model": row["model"] or "—",
            "sources_count": row["sources_count"],
            "status": row["status"],
            "interface": row["interface"] or "—",
        }
        for row in recent_raw
    ]

    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "request": request,
            "stats": stats_data,
            "recent": recent,
            "error": None,
        },
    )
