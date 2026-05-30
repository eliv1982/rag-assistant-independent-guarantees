"""
SQLite-логгер взаимодействий с RAG-ассистентом.
"""

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "logs.db"

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
            db_path: путь к файлу базы данных SQLite (по умолчанию assistant_api/logs.db)
        """
        self.db_path = str(db_path) if db_path is not None else str(_DEFAULT_DB_PATH)
        self._init_db()

    def _init_db(self) -> None:
        """Создание таблицы interactions, если она не существует."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
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

        conn.commit()
        conn.close()

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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
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

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

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

        Returns:
            id созданной записи
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
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
                error_message,
                interface,
            ),
        )

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def get_stats(self) -> Dict[str, Any]:
        """Агрегированная статистика по всем взаимодействиям."""
        conn = sqlite3.connect(self.db_path)
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

        conn.close()

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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {", ".join(_INTERACTION_COLUMNS)}
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = [self._row_to_dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def export_csv(self, csv_path: str) -> None:
        """Экспорт всех interactions в CSV с заголовками колонок."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {", ".join(_INTERACTION_COLUMNS)}
            FROM interactions
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()
        conn.close()

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(_INTERACTION_COLUMNS)
            for row in rows:
                writer.writerow([row[column] for column in _INTERACTION_COLUMNS])
