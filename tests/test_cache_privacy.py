"""
Приватность постоянного кеша ответов (RAG_CACHE_ENABLED) и отсутствие текста вопроса в stdout/логах.

Кеш хранит вопросы и ответы в SQLite, поэтому по умолчанию он выключен. VectorStore и OpenAI-клиент
подменены заглушками: сеть и Chroma не используются.
"""

import contextlib
import io
import logging
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

import app as cli_app
import rag_pipeline
from cache import cache_enabled_from_env

QUESTION = "УНИКАЛЬНЫЙ-МАРКЕР-ВОПРОСА-9137: когда гарант вправе приостановить платеж?"
ANSWER = "Ответ по ст. 376 ГК РФ (УНИКАЛЬНЫЙ-МАРКЕР-ОТВЕТА-5521)."
QUESTION_MARKER = "УНИКАЛЬНЫЙ-МАРКЕР-ВОПРОСА-9137"
ANSWER_MARKER = "УНИКАЛЬНЫЙ-МАРКЕР-ОТВЕТА-5521"
CITATION = "ГК РФ, ст. 376, п. 2"


def make_doc(distance=0.4, doc_id="doc_1", text="Гарант имеет право приостановить платеж."):
    return {
        "id": doc_id,
        "text": text,
        "distance": distance,
        "metadata": {
            "source": "gk_rf_368_379",
            "source_display": "ГК РФ, ст. 368–379",
            "source_kind": "law",
            "section_heading": "Статья 376. Отказ гаранта удовлетворить требование бенефициара — п. 2",
            "citation": CITATION,
        },
    }


class FakeCollection:
    metadata = {"hnsw:space": "cosine"}

    def count(self):
        return 5


class FakeVectorStore:
    docs = []
    searches = []

    def __init__(self, collection_name, persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = FakeCollection()

    def get_collection_stats(self):
        return {"name": self.collection_name, "count": 5, "persist_directory": self.persist_directory}

    def load_corpus(self, entries, base_dir=None):
        pass

    def search(self, query, top_k=5):
        type(self).searches.append((query, top_k))
        return [dict(d) for d in type(self).docs]

    @classmethod
    def reset(cls):
        cls.docs = [make_doc()]
        cls.searches = []


class FakeChatClient:
    def __init__(self, content=ANSWER, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []
        client = self

        class _Completions:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                message = SimpleNamespace(content=client.content)
                return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=client.finish_reason)])

        self.chat = SimpleNamespace(completions=_Completions())


class CachePrivacyTestCase(unittest.TestCase):
    cache_env = None  # значение RAG_CACHE_ENABLED в тесте; None = переменная не задана

    def setUp(self):
        FakeVectorStore.reset()
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp_dir.cleanup)
        self.dir = Path(self.temp_dir.name)
        self.cache_path = self.dir / "cache.db"
        (self.dir / "a.txt").write_text("Статья 1. Первый источник.\n", encoding="utf-8")
        self.entries = [
            {
                "path": self.dir / "a.txt",
                "source": "a",
                "source_display": "A",
                "source_kind": "law",
                "doc_type": "statute",
            }
        ]
        self.chat = FakeChatClient()
        patches = [
            mock.patch.object(rag_pipeline, "VectorStore", FakeVectorStore),
            mock.patch.object(rag_pipeline, "get_openai_client", return_value=self.chat),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        for var in ("RAG_CHROMA_PATH", "RAG_CACHE_DB_PATH", "RAG_MAX_DISTANCE", "RAG_CACHE_ENABLED"):
            os.environ.pop(var, None)
        if self.cache_env is not None:
            os.environ["RAG_CACHE_ENABLED"] = self.cache_env

    def build(self, **kwargs):
        kwargs.setdefault("cache_db_path", str(self.cache_path))
        kwargs.setdefault("persist_directory", str(self.dir / "chroma"))
        with contextlib.redirect_stdout(io.StringIO()):
            return rag_pipeline.RAGPipeline(corpus_entries=self.entries, **kwargs)

    def ask(self, pipeline, question=QUESTION, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.query(question, **kwargs)

    def stored_rows(self):
        conn = sqlite3.connect(self.cache_path)
        try:
            return conn.execute("SELECT query, answer FROM cache").fetchall()
        finally:
            conn.close()

    def bytes_on_disk(self):
        return b"".join(p.read_bytes() for p in sorted(self.dir.rglob("*")) if p.is_file())


class TestCacheSwitchParsing(unittest.TestCase):
    def test_only_explicit_true_values_enable_the_cache(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("RAG_CACHE_ENABLED", None)
            self.assertFalse(cache_enabled_from_env())
            for value in ("", " ", "\t", "0", "false", "no", "off", "maybe", "2", "enabled"):
                os.environ["RAG_CACHE_ENABLED"] = value
                self.assertFalse(cache_enabled_from_env(), repr(value))
            for value in ("1", "true", "TRUE", "Yes", "on", " 1 "):
                os.environ["RAG_CACHE_ENABLED"] = value
                self.assertTrue(cache_enabled_from_env(), repr(value))


class TestCacheIsOffByDefault(CachePrivacyTestCase):
    """RAG_CACHE_ENABLED не задан: вопросы и ответы нигде не сохраняются."""

    def test_cache_object_is_not_built_and_no_file_is_created(self):
        self.assertNotIn("RAG_CACHE_ENABLED", os.environ)
        with mock.patch.object(rag_pipeline, "RAGCache") as cache_cls:
            # без cache_db_path: иначе путь по умолчанию (assistant_api/api_rag_cache.db) мог бы быть создан
            pipeline = self.build(cache_db_path=None)

        cache_cls.assert_not_called()
        self.assertIsNone(pipeline.cache)
        self.assertFalse(pipeline.cache_enabled)

    def test_question_and_answer_text_is_not_persisted(self):
        pipeline = self.build()

        first = self.ask(pipeline)
        second = self.ask(pipeline)

        self.assertEqual(first["answer"], ANSWER)
        self.assertEqual(second["answer"], ANSWER)
        self.assertFalse(self.cache_path.exists())
        on_disk = self.bytes_on_disk()
        self.assertNotIn(QUESTION_MARKER.encode("utf-8"), on_disk)
        self.assertNotIn(ANSWER_MARKER.encode("utf-8"), on_disk)

    def test_configured_cache_path_is_not_created_either(self):
        env_path = self.dir / "from_env" / "env_cache.db"
        with mock.patch.dict(os.environ, {"RAG_CACHE_DB_PATH": str(env_path)}):
            pipeline = self.build(cache_db_path=None)
            self.ask(pipeline)

        self.assertFalse(env_path.exists())
        self.assertFalse(env_path.parent.exists())

    def test_blank_value_means_disabled(self):
        for blank in ("", "   "):
            with mock.patch.dict(os.environ, {"RAG_CACHE_ENABLED": blank}):
                pipeline = self.build()
            self.assertIsNone(pipeline.cache, repr(blank))
        self.assertFalse(self.cache_path.exists())

    def test_normal_grounded_answer_does_not_need_the_cache(self):
        pipeline = self.build()

        result = self.ask(pipeline)

        self.assertEqual(result["answer"], ANSWER)
        self.assertFalse(result["from_cache"])
        self.assertEqual(result["model"], pipeline.model)
        self.assertEqual(result["query"], QUESTION)
        self.assertEqual(len(result["context_docs"]), 1)
        self.assertEqual(result["context_docs"][0]["metadata"]["citation"], CITATION)
        self.assertEqual(len(self.chat.calls), 1)
        prompt = self.chat.calls[0]["messages"][1]["content"]
        self.assertIn("Гарант имеет право приостановить платеж.", prompt)  # ответ строится по найденному контексту
        self.assertIn(QUESTION, prompt)

    def test_repeated_question_runs_retrieval_and_model_again(self):
        pipeline = self.build()

        self.ask(pipeline)
        second = self.ask(pipeline)

        self.assertFalse(second["from_cache"])
        self.assertEqual(len(FakeVectorStore.searches), 2)
        self.assertEqual(len(self.chat.calls), 2)

    def test_existing_cache_file_is_neither_read_nor_modified_nor_deleted(self):
        with mock.patch.dict(os.environ, {"RAG_CACHE_ENABLED": "1"}):
            enabled = self.build()
        self.ask(enabled)  # кеш был включён раньше и сохранил запись
        self.assertEqual(len(self.chat.calls), 1)
        before = self.cache_path.read_bytes()

        disabled = self.build()
        result = self.ask(disabled)

        self.assertFalse(result["from_cache"])  # старый ответ из файла не отдан
        self.assertEqual(len(self.chat.calls), 2)
        self.assertTrue(self.cache_path.exists())  # файл не удалён
        self.assertEqual(self.cache_path.read_bytes(), before)  # и не изменён
        self.assertEqual(len(self.stored_rows()), 1)

    def test_stats_report_disabled_cache_without_opening_the_file(self):
        stats = self.build().get_stats()["cache"]

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["total_entries"], 0)
        self.assertFalse(self.cache_path.exists())


class TestCacheExplicitOptIn(CachePrivacyTestCase):
    """RAG_CACHE_ENABLED=1: кеш работает как раньше (namespace и происхождение ответа из этапа 2)."""

    cache_env = "1"

    def test_second_identical_question_is_served_from_cache_with_provenance(self):
        pipeline = self.build()

        first = self.ask(pipeline)
        second = self.ask(pipeline)

        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(second["answer"], ANSWER)
        self.assertEqual(len(self.chat.calls), 1)
        self.assertEqual(len(FakeVectorStore.searches), 1)
        self.assertIsNotNone(second["cached_at"])
        cached_doc = second["context_docs"][0]
        self.assertEqual(cached_doc["text"], "Гарант имеет право приостановить платеж.")
        self.assertEqual(cached_doc["metadata"]["citation"], CITATION)  # ссылки на источники сохраняются
        self.assertEqual(cached_doc["id"], "doc_1")

    def test_cache_survives_a_new_pipeline_instance(self):
        self.ask(self.build())

        hit = self.ask(self.build())

        self.assertTrue(hit["from_cache"])
        self.assertEqual(len(self.chat.calls), 1)

    def test_opt_in_stores_question_and_answer_locally(self):
        pipeline = self.build()

        self.ask(pipeline)

        self.assertEqual(self.stored_rows(), [(QUESTION, ANSWER)])
        self.assertTrue(pipeline.get_stats()["cache"]["enabled"])
        self.assertEqual(pipeline.get_stats()["cache"]["total_entries"], 1)

    def test_every_documented_true_value_enables_the_cache(self):
        for value in ("1", "true", "TRUE", "yes", "on", " 1 "):
            with mock.patch.dict(os.environ, {"RAG_CACHE_ENABLED": value}):
                pipeline = self.build()
            self.assertIsNotNone(pipeline.cache, repr(value))

    def test_use_cache_false_still_bypasses_an_enabled_cache(self):
        pipeline = self.build()

        self.ask(pipeline, use_cache=False)
        self.ask(pipeline, use_cache=False)

        self.assertEqual(len(self.chat.calls), 2)
        self.assertEqual(self.stored_rows(), [])

    def test_namespace_keeps_every_stage2_component(self):
        pipeline = self.build()

        expected = (
            f"{pipeline.corpus_id}|{pipeline.model}|{pipeline.top_k}|{rag_pipeline.PROMPT_VERSION}|{pipeline.max_distance}"
        )

        self.assertEqual(pipeline._cache_namespace(), expected)
        self.assertEqual(pipeline.cache.namespace, expected)

    def test_hit_is_not_served_after_model_prompt_threshold_or_corpus_change(self):
        original_text = (self.dir / "a.txt").read_text(encoding="utf-8")
        self.ask(self.build(model="model-a"))
        calls = len(self.chat.calls)

        def changed_prompt():
            with mock.patch.object(rag_pipeline, "PROMPT_VERSION", "other-prompt"):
                return self.build(model="model-a")

        def changed_threshold():
            with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.9"}):
                return self.build(model="model-a")

        def changed_corpus():
            (self.dir / "a.txt").write_text("Статья 1. Изменённый источник.\n", encoding="utf-8")
            return self.build(model="model-a")

        variants = {
            "model": lambda: self.build(model="model-b"),
            "prompt": changed_prompt,
            "threshold": changed_threshold,
            "corpus": changed_corpus,
        }
        for name, make_pipeline in variants.items():
            with self.subTest(changed=name):
                result = self.ask(make_pipeline())
                self.assertFalse(result["from_cache"])
                calls += 1
                self.assertEqual(len(self.chat.calls), calls)

        (self.dir / "a.txt").write_text(original_text, encoding="utf-8")
        again = self.ask(self.build(model="model-a"))  # исходная конфигурация по-прежнему попадает в кеш
        self.assertTrue(again["from_cache"])
        self.assertEqual(len(self.chat.calls), calls)

    def test_incomplete_and_no_evidence_answers_are_still_not_cached(self):
        self.chat.finish_reason = "length"
        self.ask(self.build())
        self.assertEqual(self.stored_rows(), [])

        self.chat.finish_reason = "stop"
        FakeVectorStore.docs = []
        result = self.ask(self.build(), "Другой вопрос")
        self.assertTrue(result["no_evidence"])
        self.assertEqual(self.stored_rows(), [])


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class TestQuestionIsNotLogged(CachePrivacyTestCase):
    """Текст вопроса не попадает ни в stdout (docker logs), ни в stderr, ни в logging."""

    def run_and_capture(self, pipeline, question=QUESTION, **kwargs):
        out, err, handler = io.StringIO(), io.StringIO(), _ListHandler()
        root = logging.getLogger()
        previous_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        self.addCleanup(root.removeHandler, handler)
        self.addCleanup(root.setLevel, previous_level)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = pipeline.query(question, **kwargs)
        return result, out.getvalue() + err.getvalue() + "\n".join(handler.lines)

    def assert_clean(self, output):
        self.assertIn("Запрос получен", output)  # перехват работает, а не пуст
        self.assertIn("[*] Поиск релевантных документов", output)  # прочие рабочие сообщения на месте
        self.assertNotIn(QUESTION_MARKER, output)
        self.assertNotIn(QUESTION, output)

    def test_cache_disabled(self):
        result, output = self.run_and_capture(self.build())

        self.assert_clean(output)
        self.assertEqual(result["query"], QUESTION)  # вызывающему вопрос возвращается как раньше

    def test_cache_enabled_miss_and_hit(self):
        with mock.patch.dict(os.environ, {"RAG_CACHE_ENABLED": "1"}):
            pipeline = self.build()

        _, miss_output = self.run_and_capture(pipeline)
        hit, hit_output = self.run_and_capture(pipeline)

        self.assert_clean(miss_output)
        self.assertTrue(hit["from_cache"])
        self.assertIn("Запрос получен", hit_output)
        self.assertIn("Ответ найден в кеше", hit_output)
        self.assertNotIn(QUESTION_MARKER, hit_output)
        self.assertNotIn(QUESTION, hit_output)

    def test_no_evidence_path(self):
        FakeVectorStore.docs = [make_doc(0.95)]
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.75"}):
            pipeline = self.build()

        result, output = self.run_and_capture(pipeline)

        self.assertTrue(result["no_evidence"])
        self.assertIn("Запрос получен", output)
        self.assertIn("Недостаточно данных в корпусе", output)
        self.assertNotIn(QUESTION_MARKER, output)

    def test_truncated_answer_path(self):
        self.chat.finish_reason = "length"

        result, output = self.run_and_capture(self.build())

        self.assertTrue(result["incomplete"])
        self.assertIn("Ответ неполный", output)
        self.assertNotIn(QUESTION_MARKER, output)


class TestCliWithoutCache(CachePrivacyTestCase):
    """CLI (app.py) не падает, когда кеш отключён: stats и clear объясняют это вместо обращения к файлу."""

    def test_print_stats_says_cache_is_disabled(self):
        pipeline = self.build()
        out = io.StringIO()

        with contextlib.redirect_stdout(out):
            cli_app.print_stats(pipeline)

        self.assertIn("Отключён", out.getvalue())
        self.assertNotIn("Записей:", out.getvalue())

    def test_clear_command_does_not_ask_for_confirmation_or_touch_a_file(self):
        prompts = []
        answers = iter(["stats", "clear", "exit"])

        def fake_input(prompt=""):
            prompts.append(prompt)
            try:
                return next(answers)
            except StopIteration:
                raise KeyboardInterrupt  # защита от зацикливания, если main() запросит лишний ввод

        env = {
            "LOGS_DB_PATH": str(self.dir / "logs.db"),
            "RAG_CHROMA_PATH": str(self.dir / "chroma"),
            "RAG_CACHE_DB_PATH": str(self.cache_path),
        }
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), mock.patch("builtins.input", fake_input):
            with contextlib.redirect_stdout(out):
                cli_app.main()

        self.assertEqual(len(prompts), 3)
        self.assertFalse(any("очистить кеш" in p for p in prompts))
        self.assertIn("очищать нечего", out.getvalue())
        self.assertFalse(self.cache_path.exists())


if __name__ == "__main__":
    unittest.main()
