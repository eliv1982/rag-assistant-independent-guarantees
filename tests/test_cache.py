"""
Тесты кеша ответов: ключ зависит от namespace (корпус, модель, промпт).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from cache import RAGCache


class TestRAGCacheNamespace(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "cache.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_same_namespace_hits(self):
        cache = RAGCache(self.db_path, namespace="corpusA|model|5|p1")
        cache.set("Что такое гарантия?", "ответ", [{"text": "t", "metadata": {}}])

        hit = RAGCache(self.db_path, namespace="corpusA|model|5|p1").get("что  такое ГАРАНТИЯ?")

        self.assertIsNotNone(hit)
        self.assertEqual(hit["answer"], "ответ")

    def test_changed_corpus_does_not_return_stale_answer(self):
        RAGCache(self.db_path, namespace="corpusA|model|5|p1").set("Вопрос", "старый ответ")

        self.assertIsNone(RAGCache(self.db_path, namespace="corpusB|model|5|p1").get("Вопрос"))

    def test_changed_prompt_model_or_top_k_do_not_return_stale_answer(self):
        RAGCache(self.db_path, namespace="corpusA|model|5|p1").set("Вопрос", "ответ")

        for other in ("corpusA|model|5|p2", "corpusA|other-model|5|p1", "corpusA|model|8|p1"):
            self.assertIsNone(RAGCache(self.db_path, namespace=other).get("Вопрос"), other)

    def test_default_namespace_uses_corpus_version(self):
        with mock.patch.dict(os.environ, {"RAG_CORPUS_VERSION": "1"}):
            RAGCache(self.db_path).set("Вопрос", "v1")
            self.assertEqual(RAGCache(self.db_path).get("Вопрос")["answer"], "v1")
        with mock.patch.dict(os.environ, {"RAG_CORPUS_VERSION": "2"}):
            self.assertIsNone(RAGCache(self.db_path).get("Вопрос"))

    def test_creates_missing_parent_directory(self):
        nested = os.path.join(self.temp_dir.name, "runtime", "cache.db")

        cache = RAGCache(nested, namespace="n")
        cache.set("Q", "A")

        self.assertTrue(os.path.exists(nested))
        self.assertEqual(cache.get("Q")["answer"], "A")


if __name__ == "__main__":
    unittest.main()
