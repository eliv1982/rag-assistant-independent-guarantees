"""
SQLite-логгер взаимодействий с RAG-ассистентом.
"""

import csv
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "logs.db"

# Ожидание снятия блокировки SQLite (секунды): web и CLI могут писать в один файл.
_BUSY_TIMEOUT_S = 10
_MAX_ERROR_MESSAGE_LEN = 1000
_SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-*.]{4,}")


def get_logs_db_path() -> str:
    """
    Путь к SQLite-логам: LOGS_DB_PATH или assistant_api/logs.db по умолчанию.

    Локально переменную обычно не задают. В Docker путь /app/runtime/logs.db
    задаёт docker-compose.yml (и Dockerfile), а не .env.
    """
    return (os.getenv("LOGS_DB_PATH") or "").strip() or str(_DEFAULT_DB_PATH)


def _sanitize_error_message(message: Optional[str]) -> Optional[str]:
    """Убрать из текста ошибки ключи доступа и ограничить длину (логи не должны хранить секреты)."""
    if message is None:
        return None
    text = str(message)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and len(api_key) >= 8:
        text = text.replace(api_key, "[REDACTED]")
    text = _SECRET_PATTERN.sub("[REDACTED]", text)
    if len(text) > _MAX_ERROR_MESSAGE_LEN:
        text = text[:_MAX_ERROR_MESSAGE_LEN] + "…"
    return text


_INTERACTION_COLUMNS = (
    "id",
    "created_at",
    "query",
    "response",
    "from_cache",
    "response_time_ms",
    "model",
    "top_k",
    "sources_count",
    "status",
    "error_message",
    "interface",
)


class DatabaseLogger:
    """Логгер взаимодействий пользователя с ассистентом в SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Инициализация логгера.

        Args:
            db_path: путь к файлу базы данных SQLite (по умолчанию assistant_api/logs.db);
                недостающие родительские каталоги создаются.
        """
        self.db_path = str(db_path) if db_path is not None else get_logs_db_path()
        self._init_db()

    @contextmanager
    def _connection(self, row_factory: Optional[Any] = None) -> Iterator[sqlite3.Connection]:
        """Соединение, которое всегда закрывается; commit при успехе, rollback при ошибке."""
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_S)
        try:
            if row_factory is not None:
                conn.row_factory = row_factory
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Создание каталога и таблицы interactions, если они не существуют."""
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    from_cache INTEGER NOT NULL DEFAULT 0,
                    response_time_ms INTEGER,
                    model TEXT,
                    top_k INTEGER,
                    sources_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'success',
                    error_message TEXT,
                    interface TEXT DEFAULT 'cli'
                )
                """
            )

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {column: row[column] for column in _INTERACTION_COLUMNS}

    def log_interaction(
        self,
        query: str,
        response: str,
        from_cache: bool = False,
        response_time_ms: Optional[int] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        sources_count: Optional[int] = None,
        interface: str = "cli",
    ) -> int:
        """
        Запись успешного взаимодействия.

        Returns:
            id созданной записи
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interactions (
                    created_at, query, response, from_cache,
                    response_time_ms, model, top_k, sources_count,
                    status, interface
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'success', ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    response,
                    int(from_cache),
                    response_time_ms,
                    model,
                    top_k,
                    sources_count,
                    interface,
                ),
            )
            return cursor.lastrowid

    def log_error(
        self,
        query: str,
        error_message: str,
        response_time_ms: Optional[int] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        interface: str = "cli",
    ) -> int:
        """
        Запись ошибки при обработке запроса.
        Текст ошибки очищается от ключей доступа и ограничивается по длине.

        Returns:
            id созданной записи
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interactions (
                    created_at, query, response, from_cache,
                    response_time_ms, model, top_k, sources_count,
                    status, error_message, interface
                )
                VALUES (?, ?, NULL, 0, ?, ?, ?, NULL, 'error', ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    response_time_ms,
                    model,
                    top_k,
                    _sanitize_error_message(error_message),
                    interface,
                ),
            )
            return cursor.lastrowid

    def get_stats(self) -> Dict[str, Any]:
        """Агрегированная статистика по всем взаимодействиям."""
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM interactions WHERE status = 'success'"
            )
            successful_interactions = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM interactions WHERE status != 'success'"
            )
            failed_interactions = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM interactions WHERE from_cache = 1"
            )
            cache_hits = cursor.fetchone()[0]

            cursor.execute(
                "SELECT AVG(response_time_ms) FROM interactions WHERE response_time_ms IS NOT NULL"
            )
            avg_row = cursor.fetchone()[0]

        cache_hit_rate = (
            cache_hits / total_interactions if total_interactions > 0 else 0.0
        )
        average_response_time_ms = (
            round(avg_row, 2) if avg_row is not None else None
        )

        return {
            "total_interactions": total_interactions,
            "successful_interactions": successful_interactions,
            "failed_interactions": failed_interactions,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "average_response_time_ms": average_response_time_ms,
        }

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Последние записи в порядке от новых к старым."""
        with self._connection(row_factory=sqlite3.Row) as conn:
            cursor = conn.execute(
                f"""
                SELECT {", ".join(_INTERACTION_COLUMNS)}
                FROM interactions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def export_csv(self, csv_path: str) -> None:
        """Экспорт всех interactions в CSV с заголовками колонок."""
        with self._connection(row_factory=sqlite3.Row) as conn:
            cursor = conn.execute(
                f"""
                SELECT {", ".join(_INTERACTION_COLUMNS)}
                FROM interactions
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(_INTERACTION_COLUMNS)
            for row in rows:
                writer.writerow([row[column] for column in _INTERACTION_COLUMNS])
