"""
Тесты сборки RAGPipeline: имя коллекции, переиндексация, кеш и пути хранилища.
VectorStore и OpenAI-клиент подменены заглушками: сеть и Chroma не используются.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

import rag_pipeline
from corpus_config import collection_name_for, compute_corpus_id


class FakeCollection:
    def __init__(self, store, name):
        self._store = store
        self._name = name

    def count(self):
        return self._store.counts.get(self._name, 0)


class FakeVectorStore:
    """Имитирует персистентное хранилище: коллекции живут между созданиями pipeline."""

    counts = {}
    loads = []
    constructed = []

    def __init__(self, collection_name, persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = FakeCollection(type(self), collection_name)
        type(self).constructed.append((collection_name, persist_directory))

    def get_collection_stats(self):
        return {"name": self.collection_name, "count": self.collection.count()}

    def load_corpus(self, entries, base_dir=None):
        type(self).loads.append((self.collection_name, [e["source"] for e in entries]))
        type(self).counts[self.collection_name] = len(entries)

    @classmethod
    def reset(cls):
        cls.counts = {}
        cls.loads = []
        cls.constructed = []


class PipelineWiringTestCase(unittest.TestCase):
    def setUp(self):
        FakeVectorStore.reset()
        self.temp_dir = tempfile.TemporaryDirectory()
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
        patches = [
            mock.patch.object(rag_pipeline, "VectorStore", FakeVectorStore),
            mock.patch.object(rag_pipeline, "get_openai_client", return_value=object()),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        for var in ("RAG_CHROMA_PATH", "RAG_CACHE_DB_PATH"):
            os.environ.pop(var, None)

    def tearDown(self):
        self.temp_dir.cleanup()

    def build(self, **kwargs):
        kwargs.setdefault("cache_db_path", str(self.dir / "cache.db"))
        kwargs.setdefault("persist_directory", str(self.dir / "chroma"))
        return rag_pipeline.RAGPipeline(corpus_entries=self.entries, **kwargs)


class TestCollectionNaming(PipelineWiringTestCase):
    def test_default_collection_name_is_derived_from_corpus_id(self):
        pipeline = self.build()

        expected_id = compute_corpus_id(self.entries)
        self.assertEqual(pipeline.corpus_id, expected_id)
        self.assertEqual(pipeline.vector_store.collection_name, collection_name_for(expected_id))
        self.assertEqual(pipeline.get_stats()["corpus_id"], expected_id)

    def test_explicit_collection_name_is_respected(self):
        pipeline = self.build(collection_name="custom_name")
        self.assertEqual(pipeline.vector_store.collection_name, "custom_name")

    def test_corpus_is_loaded_once_per_corpus_version(self):
        self.build()
        self.build()

        self.assertEqual(len(FakeVectorStore.loads), 1)

    def test_changed_corpus_gets_new_collection_instead_of_reusing_stale_index(self):
        """Регрессия: прежний индекс (например, со старым корпусом) не должен считаться актуальным."""
        old = self.build()
        (self.dir / "a.txt").write_text("Статья 1. Изменённый источник.\n", encoding="utf-8")
        new = self.build()

        self.assertNotEqual(old.vector_store.collection_name, new.vector_store.collection_name)
        self.assertEqual(len(FakeVectorStore.loads), 2)
        self.assertEqual(
            {name for name, _ in FakeVectorStore.loads},
            {old.vector_store.collection_name, new.vector_store.collection_name},
        )

    def test_legacy_collection_name_is_not_reused_by_default(self):
        FakeVectorStore.counts["api_rag_collection"] = 999  # индекс прежней версии приложения

        pipeline = self.build()

        self.assertNotEqual(pipeline.vector_store.collection_name, "api_rag_collection")
        self.assertEqual(len(FakeVectorStore.loads), 1)

    def test_default_corpus_is_used_when_no_entries_given(self):
        pipeline = rag_pipeline.RAGPipeline(
            cache_db_path=str(self.dir / "cache.db"),
            persist_directory=str(self.dir / "chroma"),
        )

        sources = FakeVectorStore.loads[0][1]
        self.assertEqual(len(sources), 7)
        self.assertEqual(pipeline.corpus_id, compute_corpus_id())


class TestCacheIsolation(PipelineWiringTestCase):
    def test_cached_answer_is_not_reused_after_corpus_change(self):
        old = self.build()
        old.cache.set("Вопрос", "ответ по старому корпусу")
        self.assertIsNotNone(old.cache.get("Вопрос"))

        (self.dir / "a.txt").write_text("Статья 1. Изменённый источник.\n", encoding="utf-8")
        new = self.build()

        self.assertIsNone(new.cache.get("Вопрос"))

    def test_cached_answer_is_not_reused_after_model_change(self):
        first = self.build(model="model-a")
        first.cache.set("Вопрос", "ответ")

        second = self.build(model="model-b")

        self.assertIsNone(second.cache.get("Вопрос"))

    def test_cached_answer_is_not_reused_after_prompt_change(self):
        first = self.build()
        first.cache.set("Вопрос", "ответ")

        with mock.patch.object(rag_pipeline, "PROMPT_VERSION", "other-prompt"):
            second = self.build()

        self.assertIsNone(second.cache.get("Вопрос"))

    def test_same_configuration_reuses_cache(self):
        self.build().cache.set("Вопрос", "ответ")

        hit = self.build().cache.get("Вопрос")

        self.assertEqual(hit["answer"], "ответ")


class TestStoragePaths(PipelineWiringTestCase):
    def test_paths_come_from_environment(self):
        chroma = str(self.dir / "env_chroma")
        cache = str(self.dir / "env_runtime" / "env_cache.db")

        with mock.patch.dict(os.environ, {"RAG_CHROMA_PATH": chroma, "RAG_CACHE_DB_PATH": cache}):
            pipeline = rag_pipeline.RAGPipeline(corpus_entries=self.entries)

        self.assertEqual(pipeline.vector_store.persist_directory, chroma)
        self.assertEqual(pipeline.cache.db_path, cache)
        self.assertTrue(os.path.exists(cache))

    def test_blank_environment_values_fall_back_to_defaults(self):
        with mock.patch.dict(os.environ, {"RAG_CHROMA_PATH": "", "RAG_CACHE_DB_PATH": ""}):
            with mock.patch.object(rag_pipeline, "RAGCache") as cache_cls:
                pipeline = rag_pipeline.RAGPipeline(corpus_entries=self.entries)

        self.assertTrue(pipeline.vector_store.persist_directory.endswith("chroma_db"))
        self.assertTrue(cache_cls.call_args.kwargs["db_path"].endswith("api_rag_cache.db"))

    def test_explicit_arguments_override_environment(self):
        with mock.patch.dict(os.environ, {"RAG_CHROMA_PATH": "/env/chroma"}):
            pipeline = self.build(persist_directory=str(self.dir / "arg_chroma"))

        self.assertEqual(pipeline.vector_store.persist_directory, str(self.dir / "arg_chroma"))


if __name__ == "__main__":
    unittest.main()
