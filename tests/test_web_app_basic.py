"""
Тесты web UI без обращений к OpenAI: pipeline подменяется заглушками.
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from fastapi.testclient import TestClient

import web_app
from db_logger import DatabaseLogger


class FakePipeline:
    """Заглушка RAGPipeline: ответ без сети."""

    model = "fake-model"
    top_k = 5

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.questions = []

    def query(self, question):
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        return self._result or {
            "query": question,
            "answer": "**Ответ** по статье 368",
            "from_cache": False,
            "model": "fake-model",
            "context_docs": [
                {
                    "text": "Статья 368. Понятие и форма независимой гарантии",
                    "metadata": {
                        "source_display": "ГК РФ, ст. 368–379",
                        "section_heading": "Статья 368",
                    },
                }
            ],
        }


class BrokenLogger:
    """Журнал, который падает на любой операции."""

    def log_interaction(self, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    def log_error(self, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    def get_stats(self):
        raise sqlite3.OperationalError("unable to open database file")

    def get_recent(self, limit=10):
        raise sqlite3.OperationalError("unable to open database file")


class WebAppTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_logger = web_app._logger
        self.original_pipeline = web_app._pipeline
        web_app._pipeline = None
        web_app._logger = DatabaseLogger(
            db_path=os.path.join(self.temp_dir.name, "test_logs.db")
        )
        self.client = TestClient(web_app.app)

    def tearDown(self):
        web_app._logger = self.original_logger
        web_app._pipeline = self.original_pipeline
        self.temp_dir.cleanup()

    def rows(self):
        return web_app._logger.get_recent(limit=50)


class TestWebAppBasic(WebAppTestCase):
    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_index_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("RAG Assistant: Independent Guarantees", response.text)
        self.assertIn("Спросить ассистента", response.text)

    def test_stats_empty(self):
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Пока нет записей в логе", response.text)


class TestAskSuccess(WebAppTestCase):
    def test_answer_sources_and_logging(self):
        pipeline = FakePipeline()
        web_app._pipeline = pipeline

        response = self.client.post("/ask", data={"question": "  Что такое гарантия?  "})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(pipeline.questions, ["Что такое гарантия?"])
        self.assertIn("<strong>Ответ</strong>", response.text)
        self.assertIn("ГК РФ, ст. 368–379", response.text)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["interface"], "web")
        self.assertEqual(rows[0]["sources_count"], 1)
        self.assertEqual(rows[0]["model"], "fake-model")

    def test_logger_failure_does_not_break_successful_answer(self):
        web_app._pipeline = FakePipeline()
        web_app._logger = BrokenLogger()

        response = self.client.post("/ask", data={"question": "Вопрос"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("<strong>Ответ</strong>", response.text)

    def test_logger_init_failure_does_not_break_successful_answer(self):
        web_app._pipeline = FakePipeline()
        web_app._logger = None

        with mock.patch.object(
            web_app, "DatabaseLogger", side_effect=PermissionError("read-only fs")
        ):
            response = self.client.post("/ask", data={"question": "Вопрос"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("<strong>Ответ</strong>", response.text)


class TestAskValidation(WebAppTestCase):
    def test_empty_question_returns_400(self):
        response = self.client.post("/ask", data={"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Пожалуйста, введите вопрос", response.text)
        self.assertEqual(self.rows(), [])

    def test_missing_field_returns_400(self):
        response = self.client.post("/ask", data={})
        self.assertEqual(response.status_code, 400)


class TestAskFailures(WebAppTestCase):
    def test_missing_api_key_returns_503_and_logs_error(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("OPENAI_API_KEY", None)
            with mock.patch.object(web_app, "create_rag_pipeline") as factory:
                response = self.client.post("/ask", data={"question": "Вопрос"})

        factory.assert_not_called()
        self.assertEqual(response.status_code, 503)
        self.assertIn("временно недоступен", response.text)
        self.assertNotIn("OPENAI_API_KEY", response.text)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIsNone(rows[0]["model"])

    def test_placeholder_api_key_is_treated_as_not_configured(self):
        """Ключ-заглушка из .env.example не должен приводить к обращению к OpenAI."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "your_openai_api_key_here"}):
            with mock.patch.object(web_app, "create_rag_pipeline") as factory:
                response = self.client.post("/ask", data={"question": "Вопрос"})

        factory.assert_not_called()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.rows()[0]["status"], "error")

    def test_pipeline_error_returns_500_without_leaking_details(self):
        web_app._pipeline = FakePipeline(
            error=RuntimeError("boom at /srv/secret/path with key sk-abcdef123456")
        )

        response = self.client.post("/ask", data={"question": "Вопрос"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("Не удалось получить ответ", response.text)
        self.assertNotIn("secret/path", response.text)
        self.assertNotIn("sk-abcdef", response.text)

        row = self.rows()[0]
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["model"], "fake-model")
        self.assertEqual(row["top_k"], 5)
        self.assertIn("RuntimeError", row["error_message"])
        self.assertNotIn("sk-abcdef", row["error_message"])

    def test_logger_failure_does_not_mask_pipeline_error(self):
        web_app._pipeline = FakePipeline(error=RuntimeError("boom"))
        web_app._logger = BrokenLogger()

        response = self.client.post("/ask", data={"question": "Вопрос"})

        # TestClient пробрасывает необработанные исключения сервера, поэтому сам получен
        # ответ уже означает: сбой журнала не привёл к падению обработчика.
        self.assertEqual(response.status_code, 500)
        self.assertIn("Не удалось получить ответ", response.text)

    def test_failed_initialization_is_not_repeated_by_error_handler(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock.patch.object(
                web_app, "create_rag_pipeline", side_effect=RuntimeError("init failed")
            ) as factory:
                response = self.client.post("/ask", data={"question": "Вопрос"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.rows()[0]["status"], "error")

    def test_pipeline_is_initialized_once_under_concurrency(self):
        calls = []

        def slow_factory():
            calls.append(1)
            time.sleep(0.2)
            return FakePipeline()

        results = []
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock.patch.object(web_app, "create_rag_pipeline", side_effect=slow_factory):
                threads = [
                    threading.Thread(target=lambda: results.append(web_app.get_pipeline()))
                    for _ in range(8)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 8)
        self.assertEqual(len({id(p) for p in results}), 1)


class TestStatsDegradation(WebAppTestCase):
    def test_stats_with_records(self):
        web_app._pipeline = FakePipeline()
        self.client.post("/ask", data={"question": "Вопрос"})

        response = self.client.get("/stats")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Вопрос", response.text)
        self.assertIn("status-success", response.text)

    def test_stats_returns_503_page_when_logger_is_unavailable(self):
        web_app._logger = BrokenLogger()

        response = self.client.get("/stats")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Журнал запросов временно недоступен", response.text)


if __name__ == "__main__":
    unittest.main()
