"""
Web UI для RAG-ассистента по независимым гарантиям (FastAPI + Jinja2).
"""

import os
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
    from .rag_helpers import (
        create_rag_pipeline,
        interaction_log_fields,
        normalize_sources,
        render_markdown_safe,
    )
    from .rag_pipeline import RAGPipeline
except ImportError:
    from db_logger import DatabaseLogger, get_logs_db_path
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

app = FastAPI(title="RAG Assistant: Independent Guarantees")
templates = Jinja2Templates(directory=str(_HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

_pipeline: Optional[RAGPipeline] = None
_logger: Optional[DatabaseLogger] = None


def get_logger() -> DatabaseLogger:
    """Ленивая инициализация DatabaseLogger."""
    global _logger
    if _logger is None:
        _logger = DatabaseLogger(db_path=get_logs_db_path())
    return _logger


def get_pipeline() -> RAGPipeline:
    """Ленивая инициализация RAG pipeline (требует OPENAI_API_KEY)."""
    global _pipeline
    if _pipeline is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY не установлен. Настройте переменную окружения для работы ассистента."
            )
        _pipeline = create_rag_pipeline()
    return _pipeline


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
async def ask(request: Request, question: str = Form(default="")):
    question = question.strip()
    logger = get_logger()

    if not question:
        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(
                request,
                error="Пожалуйста, введите вопрос.",
            ),
            status_code=400,
        )

    start = time.perf_counter()
    try:
        pipeline = get_pipeline()
        result = pipeline.query(question)
        response_time_ms = int((time.perf_counter() - start) * 1000)

        fields = interaction_log_fields(result, pipeline)
        logger.log_interaction(
            query=question,
            response=fields["response"],
            from_cache=fields["from_cache"],
            response_time_ms=response_time_ms,
            model=fields["model"],
            top_k=fields["top_k"],
            sources_count=fields["sources_count"],
            interface="web",
        )

        context_docs = result.get("context_docs") if isinstance(result, dict) else None
        sources = normalize_sources(context_docs)
        answer_html = render_markdown_safe(fields["response"])

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

    except Exception as exc:
        response_time_ms = int((time.perf_counter() - start) * 1000)
        model = None
        top_k = None
        try:
            pipeline = get_pipeline()
            model = pipeline.model
            top_k = pipeline.top_k
        except Exception:
            pass

        logger.log_error(
            query=question,
            error_message=str(exc),
            response_time_ms=response_time_ms,
            model=model,
            top_k=top_k,
            interface="web",
        )

        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(
                request,
                question=question,
                error=f"Не удалось получить ответ: {exc}",
            ),
            status_code=500,
        )


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    logger = get_logger()
    stats_data = logger.get_stats()
    recent_raw = logger.get_recent(limit=10)

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
        },
    )
