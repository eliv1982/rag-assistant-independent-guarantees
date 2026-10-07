"""
Тесты для DatabaseLogger.
"""

import csv
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from db_logger import DatabaseLogger


class TestDatabaseLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_logs.db")
        self.logger = DatabaseLogger(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_creates_db_and_table(self):
        self.assertTrue(os.path.exists(self.db_path))

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'"
        )
        table = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(table)

    def test_log_interaction_and_get_stats(self):
        self.logger.log_interaction(
            query="Что такое независимая гарантия?",
            response="Независимая гарантия — это обязательство банка...",
            from_cache=False,
            response_time_ms=1200,
            model="gpt-4o-mini",
            top_k=5,
            sources_count=3,
        )
        self.logger.log_interaction(
            query="Срок действия гарантии?",
            response="Срок определяется условиями договора.",
            from_cache=True,
            response_time_ms=50,
            model="gpt-4o-mini",
            top_k=5,
            sources_count=2,
        )

        stats = self.logger.get_stats()

        self.assertEqual(stats["total_interactions"], 2)
        self.assertEqual(stats["successful_interactions"], 2)
        self.assertEqual(stats["failed_interactions"], 0)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_hit_rate"], 0.5)
        self.assertEqual(stats["average_response_time_ms"], 625.0)

    def test_log_error(self):
        self.logger.log_error(
            query="Некорректный запрос",
            error_message="OpenAI API timeout",
            response_time_ms=5000,
            model="gpt-4o-mini",
            top_k=5,
        )

        stats = self.logger.get_stats()

        self.assertEqual(stats["total_interactions"], 1)
        self.assertEqual(stats["successful_interactions"], 0)
        self.assertEqual(stats["failed_interactions"], 1)
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_hit_rate"], 0.0)

        recent = self.logger.get_recent(limit=1)
        self.assertEqual(recent[0]["status"], "error")
        self.assertEqual(recent[0]["error_message"], "OpenAI API timeout")
        self.assertIsNone(recent[0]["response"])

    def test_creates_missing_parent_directories(self):
        nested = os.path.join(self.temp_dir.name, "runtime", "deeper", "logs.db")

        logger = DatabaseLogger(db_path=nested)
        logger.log_interaction(query="Q", response="A")

        self.assertTrue(os.path.exists(nested))
        self.assertEqual(logger.get_stats()["total_interactions"], 1)

    def test_log_error_redacts_secrets_and_truncates(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "real-key-value-12345"}):
            self.logger.log_error(
                query="Q",
                error_message=(
                    "401: Incorrect API key provided: real-key-value-12345 "
                    "(also sk-proj-AbCdEf123456) " + "x" * 5000
                ),
            )

        message = self.logger.get_recent(limit=1)[0]["error_message"]
        self.assertNotIn("real-key-value-12345", message)
        self.assertNotIn("sk-proj-AbCdEf123456", message)
        self.assertIn("[REDACTED]", message)
        self.assertLessEqual(len(message), 1001)

    def test_redaction_does_not_touch_ordinary_words(self):
        self.logger.log_error(query="Q", error_message="task-force failed; ask-me-later")
        message = self.logger.get_recent(limit=1)[0]["error_message"]
        self.assertEqual(message, "task-force failed; ask-me-later")

    def test_failed_write_does_not_block_later_writes(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.logger.log_interaction(query=None, response="A")  # query NOT NULL

        self.logger.log_interaction(query="После ошибки", response="A")

        self.assertEqual(self.logger.get_stats()["total_interactions"], 1)

    def test_connections_are_closed(self):
        closed = []
        real_connect = sqlite3.connect

        class TrackingConnection(sqlite3.Connection):
            def close(self):
                closed.append(True)
                super().close()

        def tracking_connect(*args, **kwargs):
            kwargs["factory"] = TrackingConnection
            return real_connect(*args, **kwargs)

        with mock.patch("db_logger.sqlite3.connect", side_effect=tracking_connect) as connect:
            self.logger.log_interaction(query="Q", response="A")
            self.logger.log_error(query="Q", error_message="e")
            self.logger.get_stats()
            self.logger.get_recent()
            with self.assertRaises(sqlite3.IntegrityError):
                self.logger.log_interaction(query=None, response="A")

        self.assertEqual(connect.call_count, len(closed))

    def test_get_recent_order(self):
        self.logger.log_interaction(query="Первый", response="Ответ 1")
        self.logger.log_interaction(query="Второй", response="Ответ 2")
        self.logger.log_interaction(query="Третий", response="Ответ 3")

        recent = self.logger.get_recent(limit=2)

        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["query"], "Третий")
        self.assertEqual(recent[1]["query"], "Второй")

    def test_export_csv(self):
        self.logger.log_interaction(
            query="Экспорт тест",
            response="Ответ для CSV",
            from_cache=False,
            response_time_ms=100,
            model="gpt-4o-mini",
            top_k=3,
            sources_count=1,
            interface="cli",
        )
        self.logger.log_error(
            query="Ошибка экспорта",
            error_message="test error",
            interface="cli",
        )

        csv_path = os.path.join(self.temp_dir.name, "export.csv")
        self.logger.export_csv(csv_path)

        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        expected_headers = [
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
        ]
        self.assertEqual(reader.fieldnames, expected_headers)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["query"], "Экспорт тест")
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[1]["query"], "Ошибка экспорта")
        self.assertEqual(rows[1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
