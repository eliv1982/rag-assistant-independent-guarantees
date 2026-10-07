"""
Тесты шлюза достаточности данных, обработки finish_reason и узких ссылок на источники.
Чат-модель, эмбеддинги и VectorStore подменены заглушками: сеть не используется.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

import chromadb
from fastapi.testclient import TestClient

import app as cli_app
import evidence
import rag_helpers
import rag_pipeline
import web_app
from db_logger import DatabaseLogger
from evidence import (
    NO_EVIDENCE_ANSWER,
    assess_evidence,
    collection_metric,
    max_distance_from_env,
    to_cosine_distance,
)

QUESTION = "Когда гарант вправе приостановить платеж?"


def make_doc(distance, citation="ГК РФ, ст. 376, п. 2", text="Гарант имеет право приостановить платеж.", doc_id="doc_1"):
    return {
        "id": doc_id,
        "text": text,
        "distance": distance,
        "metadata": {
            "source": "gk_rf_368_379",
            "source_display": "ГК РФ, ст. 368–379",
            "source_kind": "law",
            "section_heading": "Статья 376. Отказ гаранта удовлетворить требование бенефициара — п. 2",
            "citation": citation,
        },
    }


class TestMaxDistanceConfig(unittest.TestCase):
    def test_unset_or_blank_disables_the_threshold(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("RAG_MAX_DISTANCE", None)
            self.assertIsNone(max_distance_from_env())
            for blank in ("", "  ", "\t"):
                os.environ["RAG_MAX_DISTANCE"] = blank
                self.assertIsNone(max_distance_from_env(), repr(blank))

    def test_there_is_no_built_in_default_threshold(self):
        # Порог нельзя вернуть «по умолчанию»: его не на чем откалибровать без реальных эмбеддингов.
        self.assertFalse(hasattr(evidence, "DEFAULT_MAX_DISTANCE"))

    def test_environment_override(self):
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.42"}):
            self.assertEqual(max_distance_from_env(), 0.42)

    def test_invalid_values_are_explicit_errors(self):
        for bad in ("abc", "0", "-0.5", "2.5", "nan", "inf", "-inf", "0,5"):
            with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": bad}):
                with self.assertRaisesRegex(ValueError, "RAG_MAX_DISTANCE", msg=bad):
                    max_distance_from_env()


class TestMetricFromRealChromaCollections(unittest.TestCase):
    """Метрика берётся из самой коллекции Chroma; семантика расстояний проверяется на реальных векторах."""

    def test_metric_detection_and_normalisation_to_cosine_distance(self):
        stored, query = [1.0, 0.0, 0.0], [0.6, 0.8, 0.0]  # косинус = 0.6, расстояние по косинусу = 0.4
        cases = (
            ("cosine_space", {"hnsw:space": "cosine"}, "cosine", 0.4),
            ("l2_space", {"hnsw:space": "l2"}, "l2", 0.8),
            ("ip_space", {"hnsw:space": "ip"}, "ip", 0.4),
            ("default_space", None, "l2", 0.8),  # Chroma по умолчанию использует l2
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with mock.patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "False"}):
                client = chromadb.PersistentClient(path=tmp)
                for name, metadata, expected_metric, raw_distance in cases:
                    created = client.create_collection(name=name, metadata=metadata) if metadata else client.create_collection(name=name)
                    collection = client.get_collection(name)  # как делает VectorStore
                    collection.add(ids=["a"], embeddings=[stored], documents=["a"])
                    result = collection.query(query_embeddings=[query], n_results=1)
                    distance = result["distances"][0][0]
                    self.assertEqual(collection_metric(collection), expected_metric, name)
                    self.assertAlmostEqual(distance, raw_distance, places=4, msg=name)
                    self.assertAlmostEqual(to_cosine_distance(distance, collection_metric(collection)), 0.4, places=4, msg=name)
                    del created
                del client

    def test_unknown_or_missing_metric_is_none(self):
        self.assertIsNone(collection_metric(object()))
        self.assertIsNone(collection_metric(SimpleNamespace(metadata={"hnsw:space": "weird"})))
        self.assertIsNone(to_cosine_distance(0.3, None))
        self.assertIsNone(to_cosine_distance(None, "cosine"))


class TestAssessEvidence(unittest.TestCase):
    def test_close_chunk_is_sufficient(self):
        result = assess_evidence([make_doc(0.9), make_doc(0.4)], "cosine", 0.75)
        self.assertTrue(result.sufficient)
        self.assertEqual(result.best_distance, 0.4)
        self.assertEqual(result.reason, "ok")

    def test_best_chunk_decides_not_the_worst(self):
        self.assertTrue(assess_evidence([make_doc(0.95), make_doc(0.7)], "cosine", 0.75).sufficient)

    def test_far_chunks_are_insufficient(self):
        result = assess_evidence([make_doc(0.9), make_doc(0.8)], "cosine", 0.75)
        self.assertFalse(result.sufficient)
        self.assertEqual(result.reason, "too_far")
        self.assertEqual(result.best_distance, 0.8)

    def test_threshold_is_inclusive(self):
        self.assertTrue(assess_evidence([make_doc(0.75)], "cosine", 0.75).sufficient)

    def test_l2_distance_is_converted_before_comparison(self):
        # l2 = 2 - 2*cos: косинусное расстояние 0.5 соответствует l2 = 1.0
        self.assertTrue(assess_evidence([make_doc(1.0)], "l2", 0.75).sufficient)
        self.assertFalse(assess_evidence([make_doc(1.8)], "l2", 0.75).sufficient)  # косинусное 0.9

    def test_no_results_is_insufficient(self):
        result = assess_evidence([], "cosine", 0.75)
        self.assertFalse(result.sufficient)
        self.assertEqual(result.reason, "no_results")

    def test_unknown_metric_or_missing_distances_never_block_an_answer(self):
        self.assertTrue(assess_evidence([make_doc(5.0)], None, 0.75).sufficient)
        self.assertTrue(assess_evidence([make_doc(None)], "cosine", 0.75).sufficient)

    def test_without_a_threshold_distance_never_blocks(self):
        for metric in ("cosine", "l2", "ip", None):
            result = assess_evidence([make_doc(1.99)], metric, None)
            self.assertTrue(result.sufficient, metric)
            self.assertEqual(result.reason, "threshold_disabled")
            self.assertIsNone(result.max_distance)

    def test_empty_retrieval_is_insufficient_even_without_a_threshold(self):
        result = assess_evidence([], "cosine", None)
        self.assertFalse(result.sufficient)
        self.assertEqual(result.reason, "no_results")


class FakeCollection:
    def __init__(self, metric="cosine"):
        self.metadata = {"hnsw:space": metric}

    def count(self):
        return 5


class FakeVectorStore:
    docs = []
    metric = "cosine"
    searches = []

    def __init__(self, collection_name, persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = FakeCollection(type(self).metric)

    def get_collection_stats(self):
        return {"name": self.collection_name, "count": 5}

    def load_corpus(self, entries, base_dir=None):
        pass

    def search(self, query, top_k=5):
        type(self).searches.append((query, top_k))
        return [dict(d) for d in type(self).docs]


class FakeChatClient:
    def __init__(self, content="Ответ по ст. 376 ГК РФ.", finish_reason="stop"):
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


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        FakeVectorStore.docs = [make_doc(0.4)]
        FakeVectorStore.metric = "cosine"
        FakeVectorStore.searches = []
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.temp_dir.name)
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
        for var in ("RAG_CHROMA_PATH", "RAG_CACHE_DB_PATH", "RAG_MAX_DISTANCE"):
            os.environ.pop(var, None)
        self.addCleanup(self.temp_dir.cleanup)

    def build(self):
        return rag_pipeline.RAGPipeline(
            corpus_entries=self.entries,
            cache_db_path=str(self.dir / "cache.db"),
            persist_directory=str(self.dir / "chroma"),
        )

    def ask(self, pipeline, question=QUESTION, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.query(question, **kwargs)


class TestThresholdIsOptional(PipelineTestCase):
    """RAG_MAX_DISTANCE не задан или пуст: расстояние само по себе чат-модель не блокирует."""

    FAR_DOCS = (0.93, 0.88)

    def far_retrieval(self):
        # Разный текст и ссылки: одинаковые фрагменты схлопнула бы дедупликация контекста.
        FakeVectorStore.docs = [
            make_doc(d, citation=f"ГК РФ, ст. 37{i}", text=f"Текст фрагмента {i}.", doc_id=f"doc_{i}")
            for i, d in enumerate(self.FAR_DOCS)
        ]

    def test_unset_threshold_does_not_block_a_distant_retrieval(self):
        self.far_retrieval()
        self.assertNotIn("RAG_MAX_DISTANCE", os.environ)
        pipeline = self.build()

        result = self.ask(pipeline)

        self.assertIsNone(pipeline.max_distance)
        self.assertEqual(len(self.chat.calls), 1)
        self.assertNotIn("no_evidence", result)
        self.assertEqual(result["answer"], "Ответ по ст. 376 ГК РФ.")
        self.assertEqual(len(result["context_docs"]), 2)

    def test_blank_threshold_does_not_block_a_distant_retrieval(self):
        self.far_retrieval()
        for index, blank in enumerate(("", "   ")):
            with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": blank}):
                pipeline = self.build()
            result = self.ask(pipeline, f"Вопрос {index}")
            self.assertIsNone(pipeline.max_distance, repr(blank))
            self.assertNotIn("no_evidence", result, repr(blank))
        self.assertEqual(len(self.chat.calls), 2)

    def test_unset_threshold_does_not_read_the_collection_metric(self):
        self.far_retrieval()
        FakeVectorStore.metric = "weird"  # метрика неопределима, но без порога она не нужна
        pipeline = self.build()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            pipeline.query(QUESTION)
        self.assertNotIn("Метрика расстояния", out.getvalue())
        self.assertEqual(len(self.chat.calls), 1)

    def test_stats_report_no_threshold(self):
        self.assertIsNone(self.build().get_stats()["max_distance"])

    def test_threshold_is_part_of_the_cache_namespace(self):
        unset = self.build()
        unset.cache.set("Вопрос", "ответ")
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.75"}):
            explicit = self.build()
        self.assertIsNone(explicit.cache.get("Вопрос"))
        self.assertNotEqual(unset._cache_namespace(), explicit._cache_namespace())

    def test_empty_retrieval_still_has_no_context_to_answer_from(self):
        FakeVectorStore.docs = []
        result = self.ask(self.build())
        self.assertTrue(result["no_evidence"])
        self.assertEqual(result["evidence"]["reason"], "no_results")
        self.assertIsNone(result["evidence"]["max_distance"])
        self.assertEqual(self.chat.calls, [])


class TestMalformedThreshold(PipelineTestCase):
    """Неверный RAG_MAX_DISTANCE — ошибка конфигурации, а не молчаливая подстановка другого порога."""

    BAD_VALUES = ("abc", "0", "-0.5", "2.5", "nan", "inf", "0,5")

    def test_pipeline_refuses_to_start_and_names_the_variable(self):
        FakeVectorStore.docs = [make_doc(0.93)]
        for bad in self.BAD_VALUES:
            with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": bad}):
                with mock.patch.object(rag_pipeline, "VectorStore") as store:
                    with self.assertRaisesRegex(ValueError, "RAG_MAX_DISTANCE", msg=bad):
                        self.build()
            store.assert_not_called()  # до индексации (платных эмбеддингов) дело не доходит
        self.assertEqual(self.chat.calls, [])
        self.assertEqual(FakeVectorStore.searches, [])

    def test_malformed_threshold_never_falls_back_to_a_default(self):
        FakeVectorStore.docs = [make_doc(0.93)]
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "abc"}):
            with self.assertRaises(ValueError):
                self.build()
        self.assertEqual(self.chat.calls, [])


class TestMalformedThresholdOverWeb(PipelineTestCase):
    def setUp(self):
        super().setUp()
        self.original_logger = web_app._logger
        self.original_pipeline = web_app._pipeline
        web_app._logger = DatabaseLogger(db_path=str(self.dir / "logs.db"))
        web_app._pipeline = None
        self.client = TestClient(web_app.app)
        self.addCleanup(self.restore)

    def restore(self):
        web_app._logger = self.original_logger
        web_app._pipeline = self.original_pipeline

    def test_ask_fails_with_generic_message_and_logs_the_cause(self):
        os.environ["RAG_MAX_DISTANCE"] = "abc"

        with self.assertLogs(web_app.log, level="ERROR"):
            response = self.client.post("/ask", data={"question": QUESTION})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("RAG_MAX_DISTANCE", response.text)  # пользователю — общее сообщение
        self.assertEqual(self.chat.calls, [])
        self.assertIsNone(web_app._pipeline)  # сломанный pipeline не кешируется
        row = web_app._logger.get_recent(limit=1)[0]
        self.assertEqual(row["status"], "error")
        self.assertIn("RAG_MAX_DISTANCE", row["error_message"])


class TestNoEvidencePath(PipelineTestCase):
    """Явно заданный порог (0.75 здесь — произвольное тестовое значение, а не рекомендация)."""

    def setUp(self):
        super().setUp()
        os.environ["RAG_MAX_DISTANCE"] = "0.75"

    def test_far_retrieval_returns_grounded_message_without_calling_the_chat_model(self):
        FakeVectorStore.docs = [make_doc(0.93), make_doc(0.88, doc_id="doc_2")]
        pipeline = self.build()

        result = self.ask(pipeline)

        self.assertEqual(self.chat.calls, [])
        self.assertTrue(result["no_evidence"])
        self.assertEqual(result["answer"], NO_EVIDENCE_ANSWER)
        self.assertEqual(result["context_docs"], [])
        self.assertFalse(result["from_cache"])
        self.assertIsNone(result["model"])
        self.assertEqual(result["evidence"]["reason"], "too_far")
        self.assertAlmostEqual(result["evidence"]["best_distance"], 0.88)
        self.assertEqual(len(FakeVectorStore.searches), 1)  # поиск выполнен, чат — нет

    def test_message_is_russian_and_does_not_claim_the_law_has_no_answer(self):
        self.assertIn("В текущем корпусе недостаточно данных", NO_EVIDENCE_ANSWER)
        self.assertIn("не означает", NO_EVIDENCE_ANSWER)
        self.assertNotIn("законодательство не содержит", NO_EVIDENCE_ANSWER.lower())

    def test_empty_retrieval_is_also_no_evidence(self):
        FakeVectorStore.docs = []
        result = self.ask(self.build())
        self.assertTrue(result["no_evidence"])
        self.assertEqual(self.chat.calls, [])

    def test_no_evidence_answer_is_not_cached(self):
        FakeVectorStore.docs = [make_doc(0.95)]
        pipeline = self.build()
        self.ask(pipeline)
        self.assertIsNone(pipeline.cache.get(QUESTION))
        self.ask(pipeline)
        self.assertEqual(len(FakeVectorStore.searches), 2)
        self.assertEqual(self.chat.calls, [])

    def test_close_retrieval_calls_the_chat_model_once(self):
        FakeVectorStore.docs = [make_doc(0.41)]
        result = self.ask(self.build())
        self.assertEqual(len(self.chat.calls), 1)
        self.assertNotIn("no_evidence", result)
        self.assertEqual(result["answer"], "Ответ по ст. 376 ГК РФ.")
        self.assertEqual(len(result["context_docs"]), 1)

    def test_threshold_is_configurable_through_environment(self):
        FakeVectorStore.docs = [make_doc(0.5)]
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.3"}):
            strict = self.build()
            self.assertTrue(self.ask(strict)["no_evidence"])
        self.assertEqual(self.chat.calls, [])
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.6"}):
            lenient = self.build()
            self.assertNotIn("no_evidence", self.ask(lenient, "Другой вопрос"))
        self.assertEqual(len(self.chat.calls), 1)

    def test_gate_uses_the_collection_metric(self):
        FakeVectorStore.metric = "l2"
        FakeVectorStore.docs = [make_doc(1.0)]  # l2 = 1.0 -> косинусное 0.5 -> достаточно
        self.assertNotIn("no_evidence", self.ask(self.build()))
        FakeVectorStore.docs = [make_doc(1.8)]  # косинусное 0.9 -> недостаточно
        self.assertTrue(self.ask(self.build(), "Другой вопрос")["no_evidence"])

    def test_cache_namespace_depends_on_threshold(self):
        first = self.build()
        first.cache.set("Вопрос", "ответ")
        with mock.patch.dict(os.environ, {"RAG_MAX_DISTANCE": "0.5"}):
            second = self.build()
        self.assertIsNone(second.cache.get("Вопрос"))

    def test_stats_report_the_threshold(self):
        self.assertEqual(self.build().get_stats()["max_distance"], 0.75)


class TestFinishReason(PipelineTestCase):
    def test_complete_answer_is_cached(self):
        pipeline = self.build()
        result = self.ask(pipeline)
        self.assertFalse(result["incomplete"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["answer"], "Ответ по ст. 376 ГК РФ.")
        self.assertIsNotNone(pipeline.cache.get(QUESTION))

    def test_truncated_answer_is_flagged_and_not_cached(self):
        self.chat.finish_reason = "length"
        pipeline = self.build()

        result = self.ask(pipeline)

        self.assertTrue(result["incomplete"])
        self.assertEqual(result["finish_reason"], "length")
        self.assertTrue(result["answer"].startswith("Ответ по ст. 376 ГК РФ."))
        self.assertIn("оборван", result["answer"])
        self.assertIn("неполным", result["answer"])
        self.assertIsNone(pipeline.cache.get(QUESTION))
        # повторный вопрос снова идёт в модель, а не отдаёт обрезанный текст из кеша
        self.ask(pipeline)
        self.assertEqual(len(self.chat.calls), 2)

    def test_content_filter_is_flagged_as_incomplete(self):
        self.chat.finish_reason = "content_filter"
        result = self.ask(self.build())
        self.assertTrue(result["incomplete"])
        self.assertIn("фильтром содержимого", result["answer"])

    def test_missing_finish_reason_from_compatible_gateway_is_not_treated_as_truncation(self):
        self.chat.finish_reason = None
        result = self.ask(self.build())
        self.assertFalse(result["incomplete"])

    def test_empty_model_output_is_an_error_not_a_blank_answer(self):
        self.chat.content = None
        with self.assertRaises(RuntimeError):
            self.ask(self.build())

    def test_cached_answers_are_marked_complete(self):
        pipeline = self.build()
        self.ask(pipeline)
        cached = self.ask(pipeline)
        self.assertTrue(cached["from_cache"])
        self.assertFalse(cached["incomplete"])


class TestSourceRenderingInPrompt(PipelineTestCase):
    def test_context_block_uses_narrow_citation(self):
        pipeline = self.build()
        block = pipeline._format_context_block(make_doc(0.3, citation="44-ФЗ, ст. 45, ч. 6"), 1)
        self.assertIn("Фрагмент 1 [44-ФЗ, ст. 45, ч. 6", block)
        self.assertIn("Заголовок/якорь: Статья 376", block)

    def test_context_block_falls_back_to_source_display_for_old_metadata(self):
        pipeline = self.build()
        doc = make_doc(0.3)
        del doc["metadata"]["citation"]
        self.assertIn("[ГК РФ, ст. 368–379", pipeline._format_context_block(doc, 1))

    def test_prompt_passed_to_the_model_contains_narrow_citations(self):
        FakeVectorStore.docs = [make_doc(0.3, citation="223-ФЗ, ст. 3.4, ч. 14.1")]
        self.ask(self.build())
        prompt = self.chat.calls[0]["messages"][1]["content"]
        self.assertIn("223-ФЗ, ст. 3.4, ч. 14.1", prompt)


class TestLogFields(unittest.TestCase):
    PIPELINE = SimpleNamespace(model="gpt-test", top_k=5)

    def test_no_evidence_result_logs_no_model_and_zero_sources(self):
        result = {"answer": NO_EVIDENCE_ANSWER, "from_cache": False, "context_docs": [], "model": None, "no_evidence": True}
        for fields in (
            rag_helpers.interaction_log_fields(result, self.PIPELINE),
            cli_app._interaction_log_fields(result, self.PIPELINE),
        ):
            self.assertIsNone(fields["model"])
            self.assertEqual(fields["sources_count"], 0)

    def test_regular_result_still_logs_the_model(self):
        result = {"answer": "x", "from_cache": False, "context_docs": [make_doc(0.1)], "model": "gpt-test"}
        self.assertEqual(rag_helpers.interaction_log_fields(result, self.PIPELINE)["model"], "gpt-test")

    def test_normalize_sources_uses_narrow_label(self):
        sources = rag_helpers.normalize_sources([make_doc(0.2, citation="ГК РФ, ст. 376, п. 2")])
        self.assertEqual(sources[0]["label"], "ГК РФ, ст. 376, п. 2")
        self.assertTrue(sources[0]["heading"].startswith("Статья 376"))

    def test_normalize_sources_keeps_source_display_for_old_cached_metadata(self):
        sources = rag_helpers.normalize_sources([{"text": "t", "metadata": {"source_display": "ГК РФ, ст. 368–379"}}])
        self.assertEqual(sources[0]["label"], "ГК РФ, ст. 368–379")


class TestCliOutput(unittest.TestCase):
    def render(self, result):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli_app.print_response(result)
        return buffer.getvalue()

    def test_no_evidence_is_explained_and_model_is_not_claimed(self):
        output = self.render(
            {
                "query": QUESTION,
                "answer": NO_EVIDENCE_ANSWER,
                "from_cache": False,
                "context_docs": [],
                "no_evidence": True,
                "evidence": {"best_distance": 0.91, "max_distance": 0.75},
            }
        )
        self.assertIn("В текущем корпусе недостаточно данных", output)
        self.assertIn("чат-модель не вызывалась", output)
        self.assertNotIn("Источник: OpenAI API", output)

    def test_context_list_shows_narrow_citation(self):
        output = self.render(
            {
                "query": QUESTION,
                "answer": "Ответ",
                "from_cache": False,
                "model": "m",
                "context_docs": [make_doc(0.2, citation="44-ФЗ, ст. 45, ч. 6")],
            }
        )
        self.assertIn("[44-ФЗ, ст. 45, ч. 6]", output)


class NoEvidencePipeline:
    model = "fake-model"
    top_k = 5

    def __init__(self, result):
        self._result = result

    def query(self, question):
        return self._result


class TestWebPath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.original_logger = web_app._logger
        self.original_pipeline = web_app._pipeline
        web_app._logger = DatabaseLogger(db_path=os.path.join(self.temp_dir.name, "logs.db"))
        self.client = TestClient(web_app.app)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        web_app._logger = self.original_logger
        web_app._pipeline = self.original_pipeline
        self.temp_dir.cleanup()

    def test_no_evidence_answer_is_shown_and_logged_without_a_model(self):
        web_app._pipeline = NoEvidencePipeline(
            {
                "query": QUESTION,
                "answer": NO_EVIDENCE_ANSWER,
                "from_cache": False,
                "context_docs": [],
                "model": None,
                "no_evidence": True,
                "evidence": {"best_distance": 0.9, "max_distance": 0.75},
            }
        )

        response = self.client.post("/ask", data={"question": "Как приготовить борщ?"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("В текущем корпусе недостаточно данных", response.text)
        row = web_app._logger.get_recent(limit=5)[0]
        self.assertEqual(row["status"], "success")
        self.assertIsNone(row["model"])
        self.assertEqual(row["sources_count"], 0)

    def test_web_shows_narrow_source_labels(self):
        web_app._pipeline = NoEvidencePipeline(
            {
                "query": QUESTION,
                "answer": "Ответ",
                "from_cache": False,
                "model": "fake-model",
                "context_docs": [make_doc(0.2, citation="Обзор ВС РФ от 05.06.2019, позиция 11")],
            }
        )

        response = self.client.post("/ask", data={"question": QUESTION})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Обзор ВС РФ от 05.06.2019, позиция 11", response.text)


if __name__ == "__main__":
    unittest.main()
