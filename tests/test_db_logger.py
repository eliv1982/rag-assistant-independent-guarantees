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

from db_logger import DatabaseLogger, env_flag, redact_secrets


class TestDatabaseLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_logs.db")
        # Эти тесты проверяют сам журнал, поэтому текст явно включён (по умолчанию он не хранится).
        self.logger = DatabaseLogger(db_path=self.db_path, store_text=True)

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


class TestRedaction(unittest.TestCase):
    def test_common_credential_shapes_are_redacted(self):
        cases = {
            "Incorrect API key provided: sk-proj-****abcd.": "sk-proj-****abcd",
            "Authorization: Bearer abcdefghijklmnop123": "abcdefghijklmnop123",
            "request failed, api_key=AbCd1234EfGh5678 retrying": "AbCd1234EfGh5678",
            "password: hunter2hunter2": "hunter2hunter2",
            "access_token=ya29.a0AfH6SMBx": "ya29.a0AfH6SMBx",
            "proxy http://user:pa55w0rd@proxy.internal:3128 refused": "pa55w0rd",
        }
        for text, secret in cases.items():
            with self.subTest(text=text):
                redacted = redact_secrets(text)
                self.assertNotIn(secret, redacted)
                self.assertIn("[REDACTED]", redacted)

    def test_proxy_host_stays_readable_for_debugging(self):
        redacted = redact_secrets("proxy http://user:pa55w0rd@proxy.internal:3128 refused")
        self.assertIn("proxy.internal:3128", redacted)

    def test_value_of_openai_api_key_is_redacted_wherever_it_appears(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "plain-key-without-prefix-9876"}):
            redacted = redact_secrets("bad key plain-key-without-prefix-9876 (twice: plain-key-without-prefix-9876)")
        self.assertNotIn("plain-key-without-prefix", redacted)

    def test_ordinary_text_is_untouched(self):
        for text in (
            "task-force failed; ask-me-later",
            "max tokens: 1500 exceeded",
            "RAG_MAX_DISTANCE='abc': ожидается число (косинусное расстояние, 0..2)",
            "unable to open database file",
        ):
            with self.subTest(text=text):
                self.assertEqual(redact_secrets(text), text)

    def test_none_stays_none(self):
        self.assertIsNone(redact_secrets(None))

    def test_stored_error_message_never_contains_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DatabaseLogger(db_path=os.path.join(tmp, "l.db"))
            logger.log_error(
                query="Q",
                error_message="AuthenticationError: Bearer abcdefghijklmnop123 / api_key=AbCd1234EfGh5678",
            )
            message = logger.get_recent(limit=1)[0]["error_message"]
        self.assertNotIn("abcdefghijklmnop123", message)
        self.assertNotIn("AbCd1234EfGh5678", message)
        self.assertIn("AuthenticationError", message)  # диагностика остаётся


class TestTextStorage(unittest.TestCase):
    """По умолчанию журнал хранит только метаданные: тексты вопросов и ответов - по явному LOGS_STORE_TEXT."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "test_logs.db")
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("LOGS_STORE_TEXT", None)

    def test_default_stores_no_question_or_answer_text(self):
        logger = DatabaseLogger(db_path=self.db_path)
        logger.log_interaction(query="Секретный вопрос", response="Секретный ответ", model="m", sources_count=2)
        logger.log_error(query="Другой секретный вопрос", error_message="boom", model="m")

        rows = logger.get_recent(limit=5)

        self.assertEqual({row["query"] for row in rows}, {""})
        self.assertIsNone(rows[0]["response"])
        self.assertIsNone(rows[1]["response"])
        # Метаданные для отладки сохраняются.
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error_message"], "boom")
        self.assertEqual(rows[1]["sources_count"], 2)
        self.assertEqual(logger.get_stats()["total_interactions"], 2)

    def test_default_leaves_no_trace_of_text_in_the_database_file(self):
        logger = DatabaseLogger(db_path=self.db_path)
        logger.log_interaction(query="УНИКАЛЬНЫЙ-ВОПРОС-123", response="УНИКАЛЬНЫЙ-ОТВЕТ-456")

        with open(self.db_path, "rb") as db_file:
            raw = db_file.read()

        self.assertNotIn("УНИКАЛЬНЫЙ-ВОПРОС-123".encode("utf-8"), raw)
        self.assertNotIn("УНИКАЛЬНЫЙ-ОТВЕТ-456".encode("utf-8"), raw)

    def test_env_flag_enables_text_storage_and_redacts_secrets_in_it(self):
        os.environ["LOGS_STORE_TEXT"] = "1"
        logger = DatabaseLogger(db_path=self.db_path)

        logger.log_interaction(
            query="Мой ключ sk-proj-AbCdEf123456, что делать?",
            response="Ответ про ключ sk-proj-AbCdEf123456",
        )

        row = logger.get_recent(limit=1)[0]
        self.assertIn("что делать?", row["query"])
        self.assertIn("Ответ про ключ", row["response"])
        self.assertNotIn("AbCdEf123456", row["query"])
        self.assertNotIn("AbCdEf123456", row["response"])

    def test_explicit_argument_overrides_the_environment(self):
        os.environ["LOGS_STORE_TEXT"] = "1"
        off = DatabaseLogger(db_path=self.db_path, store_text=False)
        off.log_interaction(query="Q", response="A")
        self.assertEqual(off.get_recent(limit=1)[0]["query"], "")

        os.environ.pop("LOGS_STORE_TEXT")
        on = DatabaseLogger(db_path=self.db_path, store_text=True)
        on.log_interaction(query="Q2", response="A2")
        self.assertEqual(on.get_recent(limit=1)[0]["query"], "Q2")

    def test_env_flag_parsing_requires_an_explicit_truthy_value(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"FLAG_UNDER_TEST": value}):
                    self.assertTrue(env_flag("FLAG_UNDER_TEST"))
        for value in ("", "0", "false", "no", "off", "2", "enabled"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"FLAG_UNDER_TEST": value}):
                    self.assertFalse(env_flag("FLAG_UNDER_TEST"))
        os.environ.pop("FLAG_UNDER_TEST", None)
        self.assertFalse(env_flag("FLAG_UNDER_TEST"))


class TestStatsAggregates(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.logger = DatabaseLogger(db_path=os.path.join(self.temp_dir.name, "l.db"), store_text=False)

    def test_model_usage_and_latency_aggregates(self):
        for _ in range(3):
            self.logger.log_interaction(query="Q", response="A", model="model-a", response_time_ms=100)
        self.logger.log_interaction(query="Q", response="A", model="model-b", response_time_ms=900)
        self.logger.log_interaction(query="Q", response="A", model=None, response_time_ms=10)  # без модели
        self.logger.log_error(query="Q", error_message="boom", model="model-a", response_time_ms=50)

        stats = self.logger.get_stats()

        self.assertEqual(
            stats["model_usage"],
            [
                {"model": "model-a", "count": 4},
                {"model": "model-b", "count": 1},
                {"model": None, "count": 1},
            ],
        )
        self.assertEqual(stats["max_response_time_ms"], 900)
        self.assertEqual(stats["total_interactions"], 6)
        self.assertEqual(stats["failed_interactions"], 1)

    def test_empty_database_aggregates(self):
        stats = self.logger.get_stats()
        self.assertEqual(stats["model_usage"], [])
        self.assertIsNone(stats["max_response_time_ms"])
        self.assertIsNone(stats["average_response_time_ms"])

    def test_stats_contain_only_aggregates_even_when_text_is_stored(self):
        logger = DatabaseLogger(db_path=os.path.join(self.temp_dir.name, "text.db"), store_text=True)
        logger.log_interaction(query="УНИКАЛЬНЫЙ-ВОПРОС", response="УНИКАЛЬНЫЙ-ОТВЕТ", model="m")

        self.assertEqual(logger.get_recent(limit=1)[0]["query"], "УНИКАЛЬНЫЙ-ВОПРОС")  # текст в БД есть
        self.assertNotIn("УНИКАЛЬНЫЙ", repr(logger.get_stats()))


if __name__ == "__main__":
    unittest.main()
