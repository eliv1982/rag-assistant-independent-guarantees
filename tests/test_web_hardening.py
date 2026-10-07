"""
Тесты защиты web-поверхности: приватность /stats и журнала, ошибки, экранирование, заголовки.
Без обращений к OpenAI и сети: pipeline подменяется заглушкой.
"""

import logging
import os
import re
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from fastapi.testclient import TestClient

import web_app
from db_logger import DatabaseLogger
from test_web_app_basic import BrokenLogger, FakePipeline, WebAppTestCase

QUESTION = "УНИКАЛЬНЫЙ-ВОПРОС про банковскую гарантию"
ANSWER = "УНИКАЛЬНЫЙ-ОТВЕТ про статью 368"
FAKE_KEY = "sk-proj-TESTONLY1234567890"
INTERNAL_PATH = "/srv/secret/internal/path.py"
WINDOWS_PATH = "C:\\Users\\someone\\secret\\config.py"
LEAKY_MESSAGE = (
    f"boom at {INTERNAL_PATH} and {WINDOWS_PATH} with key {FAKE_KEY} "
    "and header Bearer abcdefghijklmnop123"
)
LEAK_FRAGMENTS = (
    "secret/internal",
    "someone",
    FAKE_KEY,
    "TESTONLY",
    "abcdefghijklmnop123",
    "Traceback",
    'File "',
    "RuntimeError",
)

TEMPLATES_DIR = Path(web_app.__file__).resolve().parent / "templates"
STATIC_DIR = Path(web_app.__file__).resolve().parent / "static"


def result_with(answer, context_docs=None, model="fake-model"):
    return {
        "query": QUESTION,
        "answer": answer,
        "from_cache": False,
        "model": model,
        "context_docs": context_docs or [],
    }


def metric_values(html):
    """Целочисленные плитки метрик: всего, успешных, с ошибкой, попаданий в кеш."""
    return re.findall(r'<span class="metric-value">\s*(\d+)\s*</span>', html)


class TestStatsPrivacy(WebAppTestCase):
    def use_text_logger(self):
        web_app._logger = DatabaseLogger(
            db_path=os.path.join(self.temp_dir.name, "text_logs.db"), store_text=True
        )

    def test_default_stats_hides_questions_and_answers_even_if_text_is_stored(self):
        self.use_text_logger()
        web_app._pipeline = FakePipeline(result=result_with(ANSWER))
        self.client.post("/ask", data={"question": QUESTION})
        self.assertEqual(web_app._logger.get_recent(limit=1)[0]["query"], QUESTION)  # текст в БД есть

        page = self.client.get("/stats").text

        self.assertNotIn("УНИКАЛЬНЫЙ-ВОПРОС", page)
        self.assertNotIn("УНИКАЛЬНЫЙ-ОТВЕТ", page)

    def test_recent_table_is_off_by_default(self):
        web_app._pipeline = FakePipeline()
        self.client.post("/ask", data={"question": "Вопрос"})

        page = self.client.get("/stats").text

        self.assertNotIn("Последние запросы", page)
        self.assertNotIn("status-success", page)

    def test_recent_table_requires_explicit_opt_in(self):
        web_app._pipeline = FakePipeline()
        self.client.post("/ask", data={"question": "Вопрос"})

        for value in ("", "0", "false", "no", "off", "enabled"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"STATS_SHOW_RECENT": value}):
                    self.assertNotIn("Последние запросы", self.client.get("/stats").text)
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"STATS_SHOW_RECENT": value}):
                    self.assertIn("Последние запросы", self.client.get("/stats").text)

    def test_opt_in_recent_table_shows_metadata_only(self):
        self.use_text_logger()
        web_app._pipeline = FakePipeline(result=result_with(ANSWER))
        self.client.post("/ask", data={"question": QUESTION})
        web_app._pipeline = FakePipeline(error=RuntimeError(LEAKY_MESSAGE))
        self.client.post("/ask", data={"question": QUESTION})

        with mock.patch.dict(os.environ, {"STATS_SHOW_RECENT": "1"}):
            page = self.client.get("/stats").text

        self.assertIn("Последние запросы", page)
        self.assertIn("status-success", page)
        self.assertIn("status-error", page)
        self.assertIn("fake-model", page)
        self.assertNotIn("УНИКАЛЬНЫЙ", page)
        for fragment in ("secret/internal", "someone", "TESTONLY", "abcdefghijklmnop123", "boom at"):
            self.assertNotIn(fragment, page)

    def test_aggregates_remain_available(self):
        web_app._pipeline = FakePipeline()
        self.client.post("/ask", data={"question": "Один"})
        self.client.post("/ask", data={"question": "Два"})
        web_app._pipeline = FakePipeline(error=RuntimeError("boom"))
        self.client.post("/ask", data={"question": "Три"})

        page = self.client.get("/stats").text

        self.assertEqual(metric_values(page)[:4], ["3", "2", "1", "0"])  # всего / успешных / с ошибкой / кеш
        self.assertIn("Использование моделей", page)
        self.assertIn("fake-model", page)
        self.assertIn("Максимальное время ответа", page)

    def test_stats_page_escapes_values_from_the_journal(self):
        web_app._logger.log_interaction(
            query="Q", response="A", model="<script>alert(1)</script>", response_time_ms=5
        )

        with mock.patch.dict(os.environ, {"STATS_SHOW_RECENT": "1"}):
            page = self.client.get("/stats").text

        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)


class TestStatsFailureIsolation(WebAppTestCase):
    def test_broken_logger_gives_generic_503_page(self):
        web_app._logger = BrokenLogger()

        with self.assertLogs(web_app.log, level="ERROR") as logs:
            response = self.client.get("/stats")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Журнал запросов временно недоступен", response.text)
        self.assertNotIn("unable to open database file", response.text)
        self.assertNotIn("OperationalError", response.text)
        self.assertEqual(metric_values(response.text)[:4], ["0", "0", "0", "0"])
        # Причина остаётся в серверном логе.
        self.assertIn("unable to open database file", "\n".join(logs.output))

    def test_logger_construction_failure_gives_generic_503_page(self):
        web_app._logger = None

        with mock.patch.object(
            web_app, "DatabaseLogger", side_effect=PermissionError("/secret/path/logs.db: read-only")
        ):
            with self.assertLogs(web_app.log, level="ERROR"):
                response = self.client.get("/stats")

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("/secret/path", response.text)
        self.assertNotIn("PermissionError", response.text)

    def test_corrupted_database_file_degrades_stats_and_does_not_break_ask(self):
        Path(web_app._logger.db_path).write_bytes(b"this is not a sqlite database " * 50)

        with self.assertLogs(web_app.log, level="ERROR"):
            stats_response = self.client.get("/stats")
        self.assertEqual(stats_response.status_code, 503)
        self.assertNotIn("not a database", stats_response.text)
        self.assertNotIn(self.temp_dir.name, stats_response.text)

        web_app._pipeline = FakePipeline()
        with self.assertLogs(web_app.log, level="ERROR"):
            ask_response = self.client.post("/ask", data={"question": "Вопрос"})
        self.assertEqual(ask_response.status_code, 200)
        self.assertIn("<strong>Ответ</strong>", ask_response.text)

    def test_unexpected_failure_returns_generic_500_without_details(self):
        class MalformedStatsLogger:
            def get_stats(self):
                return {}  # шаблон упадёт на отсутствующих полях: ошибка вне try в обработчике

            def get_recent(self, limit=10):
                return []

        web_app._logger = MalformedStatsLogger()
        client = TestClient(web_app.app, raise_server_exceptions=False)

        response = client.get("/stats")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.text, "Internal Server Error")
        self.assertNotIn("Undefined", response.text)
        self.assertNotIn("Traceback", response.text)


class TestAskErrorExposure(WebAppTestCase):
    def ask_failing(self, error):
        web_app._pipeline = FakePipeline(error=error)
        with self.assertLogs(web_app.log, level="ERROR") as logs:
            response = self.client.post("/ask", data={"question": QUESTION})
        return response, "\n".join(logs.output)

    def test_internal_exception_text_never_reaches_the_browser(self):
        errors = (
            RuntimeError(LEAKY_MESSAGE),
            ValueError(LEAKY_MESSAGE),
            OSError(LEAKY_MESSAGE),
            sqlite3.OperationalError(LEAKY_MESSAGE),
            KeyError(LEAKY_MESSAGE),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                response, _ = self.ask_failing(error)

                self.assertEqual(response.status_code, 500)
                self.assertIn("Не удалось получить ответ", response.text)
                for fragment in LEAK_FRAGMENTS + (type(error).__name__,):
                    self.assertNotIn(fragment, response.text)

    def test_secrets_are_redacted_from_journal_and_server_log_but_diagnostics_stay(self):
        response, server_log = self.ask_failing(RuntimeError(LEAKY_MESSAGE))

        self.assertEqual(response.status_code, 500)
        journal = self.rows()[0]["error_message"]
        for text in (journal, server_log):
            self.assertNotIn(FAKE_KEY, text)
            self.assertNotIn("TESTONLY", text)
            self.assertNotIn("abcdefghijklmnop123", text)
            self.assertIn("[REDACTED]", text)
            self.assertIn("RuntimeError", text)  # тип ошибки остаётся для отладки
            self.assertIn("boom at", text)
        self.assertIn("Traceback", server_log)  # traceback в серверном логе сохраняется

    def test_value_of_openai_api_key_is_redacted_from_journal_and_server_log(self):
        key = "plain-env-key-99887766"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}):
            _, server_log = self.ask_failing(RuntimeError(f"401 Unauthorized for key {key}"))

        self.assertNotIn(key, server_log)
        self.assertNotIn(key, self.rows()[0]["error_message"])

    def test_malformed_config_is_distinguishable_in_logs_but_not_in_browser(self):
        message = "RAG_MAX_DISTANCE='abc': ожидается число (косинусное расстояние, 0..2)"

        response, server_log = self.ask_failing(ValueError(message))

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("RAG_MAX_DISTANCE", response.text)
        self.assertIn(message, server_log)
        self.assertIn(message, self.rows()[0]["error_message"])
        self.assertIn("ValueError", self.rows()[0]["error_message"])

    def test_unavailable_assistant_does_not_leak_configuration_details(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertLogs(web_app.log, level="ERROR"):
                response = self.client.post("/ask", data={"question": QUESTION})

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("OPENAI", response.text)
        self.assertNotIn("AssistantUnavailableError", response.text)

    def test_log_filter_fails_closed_instead_of_breaking_the_error_path(self):
        record = logging.LogRecord(
            name="assistant_api.web", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="bad format %d with key " + FAKE_KEY, args=("not-a-number",), exc_info=None,
        )

        self.assertTrue(web_app._RedactingFilter().filter(record))  # не бросает

        message = record.getMessage()
        self.assertNotIn(FAKE_KEY, message)
        self.assertNotIn("TESTONLY", message)

    def test_app_keeps_serving_after_a_failed_request(self):
        self.ask_failing(RuntimeError(LEAKY_MESSAGE))

        web_app._pipeline = FakePipeline()
        response = self.client.post("/ask", data={"question": "Вопрос"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/stats").status_code, 200)

    def test_markdown_render_failure_falls_back_to_escaped_text(self):
        web_app._pipeline = FakePipeline(result=result_with("<b>готовый</b> ответ & текст"))

        with mock.patch.object(web_app, "render_markdown_safe", side_effect=RuntimeError("render boom")):
            with self.assertLogs(web_app.log, level="ERROR"):
                response = self.client.post("/ask", data={"question": QUESTION})

        self.assertEqual(response.status_code, 200)
        self.assertIn("&lt;b&gt;готовый&lt;/b&gt; ответ &amp; текст", response.text)
        self.assertNotIn("render boom", response.text)
        self.assertEqual(self.rows()[0]["status"], "success")


class TestJournalPrivacyThroughWeb(WebAppTestCase):
    """Журнал, который web создаёт сам (get_logger), по умолчанию не хранит тексты."""

    def make_default_logger(self):
        web_app._logger = None
        with mock.patch.dict(os.environ, {"LOGS_DB_PATH": os.path.join(self.temp_dir.name, "env_logs.db")}):
            return web_app.get_logger()

    def test_default_web_journal_stores_no_question_or_answer(self):
        logger = self.make_default_logger()
        web_app._pipeline = FakePipeline(result=result_with(ANSWER))

        self.client.post("/ask", data={"question": QUESTION})
        web_app._pipeline = FakePipeline(error=RuntimeError("boom"))
        with self.assertLogs(web_app.log, level="ERROR"):
            self.client.post("/ask", data={"question": QUESTION})

        rows = logger.get_recent(limit=5)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["query"] for row in rows}, {""})
        self.assertEqual({row["response"] for row in rows}, {None})
        raw = Path(logger.db_path).read_bytes()
        self.assertNotIn("УНИКАЛЬНЫЙ".encode("utf-8"), raw)

    def test_text_is_stored_only_with_explicit_opt_in(self):
        with mock.patch.dict(os.environ, {"LOGS_STORE_TEXT": "1"}):
            logger = self.make_default_logger()
        web_app._pipeline = FakePipeline(result=result_with(ANSWER))

        self.client.post("/ask", data={"question": QUESTION})

        row = logger.get_recent(limit=1)[0]
        self.assertEqual(row["query"], QUESTION)
        self.assertIn("УНИКАЛЬНЫЙ-ОТВЕТ", row["response"])


class TestTemplateEscaping(WebAppTestCase):
    XSS = "</textarea><script>alert(1)</script><img src=x onerror=alert(2)>"

    def assert_escaped(self, html):
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;/textarea&gt;&lt;script&gt;alert(1)&lt;/script&gt;", html)

    def test_question_is_escaped_on_success(self):
        web_app._pipeline = FakePipeline()
        self.assert_escaped(self.client.post("/ask", data={"question": self.XSS}).text)

    def test_question_is_escaped_on_pipeline_error(self):
        web_app._pipeline = FakePipeline(error=RuntimeError("boom"))
        with self.assertLogs(web_app.log, level="ERROR"):
            response = self.client.post("/ask", data={"question": self.XSS})
        self.assertEqual(response.status_code, 500)
        self.assert_escaped(response.text)

    def test_question_is_escaped_when_assistant_is_unavailable(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertLogs(web_app.log, level="ERROR"):
                response = self.client.post("/ask", data={"question": self.XSS})
        self.assertEqual(response.status_code, 503)
        self.assert_escaped(response.text)

    def test_sources_and_metadata_are_escaped(self):
        doc = {
            "text": "<img src=x onerror=alert(1)> текст нормы",
            "metadata": {"source_display": "<b>ГК РФ</b>", "section_heading": "<i>Статья</i>"},
        }
        web_app._pipeline = FakePipeline(result=result_with("Ответ", [doc], model="<u>model</u>"))

        page = self.client.post("/ask", data={"question": "Вопрос"}).text

        for raw in ("<img src=x", "<b>ГК РФ</b>", "<i>Статья</i>", "<u>model</u>"):
            self.assertNotIn(raw, page)
        for escaped in ("&lt;img src=x", "&lt;b&gt;ГК РФ&lt;/b&gt;", "&lt;i&gt;Статья&lt;/i&gt;", "&lt;u&gt;model&lt;/u&gt;"):
            self.assertIn(escaped, page)

    def test_model_answer_html_is_sanitized_before_it_reaches_the_page(self):
        answer = "<script>alert(1)</script> **безопасно** [x](javascript:alert(1)) <img src=x onerror=alert(2)>"
        web_app._pipeline = FakePipeline(result=result_with(answer))

        page = self.client.post("/ask", data={"question": "Вопрос"}).text

        self.assertIn("<strong>безопасно</strong>", page)
        self.assertNotIn("<script", page.lower())
        self.assertNotIn("javascript:", page.lower())
        self.assertNotIn("onerror", page.lower())

    def test_safe_filter_is_used_only_for_the_sanitized_answer(self):
        found = []
        for template in sorted(TEMPLATES_DIR.glob("*.html")):
            source = template.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"\{%-?\s*autoescape\s+false")
            found += [(template.name, m) for m in re.findall(r"\{\{[^}]*\|\s*safe\b[^}]*\}\}", source)]
        self.assertEqual(found, [("index.html", "{{ answer_html | safe }}")])


class TestSecurityHeaders(WebAppTestCase):
    def assert_hardened(self, response):
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("referrer-policy"), "no-referrer")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertIn("default-src 'self'", response.headers.get("content-security-policy", ""))

    def test_headers_on_pages_static_and_errors(self):
        web_app._pipeline = FakePipeline()
        responses = {
            "GET /": self.client.get("/"),
            "GET /stats": self.client.get("/stats"),
            "GET /health": self.client.get("/health"),
            "GET /static/styles.css": self.client.get("/static/styles.css"),
            "GET /missing (404)": self.client.get("/missing"),
            "POST /ask (200)": self.client.post("/ask", data={"question": "Вопрос"}),
            "POST /ask (400)": self.client.post("/ask", data={"question": " "}),
        }
        web_app._pipeline = FakePipeline(error=RuntimeError("boom"))
        with self.assertLogs(web_app.log, level="ERROR"):
            responses["POST /ask (500)"] = self.client.post("/ask", data={"question": "Вопрос"})

        self.assertEqual(
            [r.status_code for r in responses.values()], [200, 200, 200, 200, 404, 200, 400, 500]
        )
        for name, response in responses.items():
            with self.subTest(name):
                self.assert_hardened(response)

    def test_csp_is_strict_where_the_app_allows_it(self):
        policy = self.client.get("/").headers["content-security-policy"]

        for directive in ("script-src 'none'", "object-src 'none'", "frame-ancestors 'none'", "base-uri 'none'", "form-action 'self'"):
            self.assertIn(directive, policy)
        self.assertNotIn("unsafe-inline", policy)
        self.assertNotIn("unsafe-eval", policy)
        self.assertNotIn("*", policy)

    def test_no_hsts_because_the_app_cannot_assume_https(self):
        self.assertNotIn("strict-transport-security", self.client.get("/").headers)

    def test_builtin_api_docs_keep_working_and_are_exempt_from_csp(self):
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertNotIn("content-security-policy", response.headers)

    def test_pages_have_no_inline_or_external_active_content_so_the_csp_does_not_break_them(self):
        web_app._pipeline = FakePipeline()
        pages = [
            self.client.get("/").text,
            self.client.get("/stats").text,
            self.client.post("/ask", data={"question": "Вопрос"}).text,
            self.client.post("/ask", data={"question": " "}).text,
        ]
        for page in pages:
            self.assertNotIn("<script", page.lower())
            self.assertNotIn("<style", page.lower())
            self.assertNotRegex(page, r"(?i)\sstyle\s*=")
            self.assertNotRegex(page, r"(?i)\son[a-z]+\s*=")
            for url in re.findall(r'<link[^>]*\shref="([^"]*)"', page):
                self.assertTrue(url.startswith("/"), url)
        css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
        self.assertNotRegex(css, r"(?i)@import|url\(\s*['\"]?\s*(https?:)?//")


if __name__ == "__main__":
    unittest.main()
