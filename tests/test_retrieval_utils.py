"""
Тесты deduplication для retrieval.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from retrieval_utils import deduplicate_context_docs


class TestDeduplicateContextDocs(unittest.TestCase):
    def test_removes_exact_duplicate_dicts(self):
        docs = [
            {"text": "Статья 378. Основания прекращения", "metadata": {"source": "gk_rf"}},
            {"text": "Статья 378. Основания прекращения", "metadata": {"source": "gk_rf"}},
            {"text": "Статья 45 44-ФЗ об условиях независимой гарантии", "metadata": {"source": "fz_44_art_45"}},
        ]
        result = deduplicate_context_docs(docs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], docs[0]["text"])
        self.assertEqual(result[1]["text"], docs[2]["text"])

    def test_keeps_different_documents(self):
        docs = [
            {"text": "Статья 378 ГК РФ о прекращении гарантии"},
            {"text": "Статья 370 ГК РФ об изменении гарантии"},
            {"content": "Часть 6 статьи 45 44-ФЗ об основаниях отказа в принятии гарантии"},
        ]
        result = deduplicate_context_docs(docs)
        self.assertEqual(len(result), 3)

    def test_max_docs_limits_result(self):
        docs = [{"text": f"doc {i}"} for i in range(5)]
        result = deduplicate_context_docs(docs, max_docs=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "doc 0")
        self.assertEqual(result[1]["text"], "doc 1")

    def test_removes_near_duplicates(self):
        base = (
            "Article 25. Expiry of the guarantee. The guarantee expires when "
            "the principal obligation is fulfilled in full according to law."
        )
        near = base + " "
        docs = [
            {"text": base, "metadata": {"section": "a"}},
            {"text": near, "metadata": {"section": "b"}},
            {"text": "Completely different legal text about bank guarantees and courts."},
        ]
        result = deduplicate_context_docs(docs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], base)
        self.assertIn("Completely different", result[1]["text"])


if __name__ == "__main__":
    unittest.main()
