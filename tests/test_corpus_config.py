"""
Тесты конфигурации корпуса и отпечатка corpus_id (без OpenAI).
"""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

import corpus_config
from corpus_config import (
    collection_name_for,
    compute_corpus_id,
    default_corpus_entries,
    index_settings,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "assistant_api" / "data"
# Правило Chroma для имён коллекций: 3-63 символа, буквы/цифры в начале и конце, внутри ._-
_CHROMA_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


class TestDefaultCorpus(unittest.TestCase):
    def test_entries_point_to_existing_files(self):
        entries = default_corpus_entries()
        self.assertEqual(len(entries), 7)
        for entry in entries:
            self.assertTrue(Path(entry["path"]).is_file(), entry["path"])

    def test_every_data_file_is_configured_and_vice_versa(self):
        configured = {Path(e["path"]).name for e in default_corpus_entries()}
        on_disk = {p.name for p in _DATA_DIR.glob("*.txt")}
        self.assertEqual(configured, on_disk)

    def test_entries_have_unique_sources_and_complete_metadata(self):
        entries = default_corpus_entries()
        self.assertEqual(len({e["source"] for e in entries}), len(entries))
        for entry in entries:
            for key in ("path", "source", "source_display", "source_kind", "doc_type", "chunker", "citation"):
                self.assertTrue(entry.get(key), f"{entry.get('source')}: нет {key}")
            self.assertIn(entry["doc_type"], {"statute", "overview"})

    def test_source_kinds_have_labels_in_vector_store(self):
        from vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)
        for entry in default_corpus_entries():
            label = store._kind_label(entry["source_kind"])
            self.assertNotEqual(label, entry["source_kind"], f"нет подписи для {entry['source_kind']}")

    def test_removed_source_absent_from_corpus_config_and_data(self):
        needle = "urd" + "g"
        for path in [Path(corpus_config.__file__), *_DATA_DIR.glob("*")]:
            self.assertNotIn(needle, path.read_text(encoding="utf-8").lower(), path.name)


class TestArticle369StatusLine(unittest.TestCase):
    """Статья 369 сохранена только как строка-статус, без утратившего силу содержания."""

    LINE = "Статья 369. Утратила силу с 1 июня 2015 года. - Федеральный закон от 08.03.2015 N 42-ФЗ."

    def test_status_line_present_exactly_once(self):
        text = (_DATA_DIR / "GK_368_379.txt").read_text(encoding="utf-8")
        self.assertEqual(text.count("Статья 369"), 1)
        self.assertIn(self.LINE, text.splitlines())

    def test_article_369_section_is_the_single_status_line(self):
        text = (_DATA_DIR / "GK_368_379.txt").read_text(encoding="utf-8")
        lines = text.splitlines()
        start = lines.index(self.LINE)
        following = lines[start + 1] if start + 1 < len(lines) else ""
        self.assertTrue(following.startswith("Статья 370"), following)


class TestChunkingDryRun(unittest.TestCase):
    """Нарезка каждого источника работает локально: эмбеддинги не запрашиваются."""

    def test_every_source_produces_chunks_and_article_369_is_retrievable(self):
        from vector_store import VectorStore

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                store = VectorStore(collection_name="dry_run", persist_directory=tmp)
                all_chunks = {}
                for entry in default_corpus_entries():
                    text = Path(entry["path"]).read_text(encoding="utf-8")
                    chunks = store._build_chunks_for_file(
                        text,
                        source=entry["source"],
                        source_display=entry["source_display"],
                        source_kind=entry["source_kind"],
                        doc_type=entry["doc_type"],
                        chunker=entry["chunker"],
                        citation=entry["citation"],
                    )
                    self.assertGreater(len(chunks), 0, entry["source"])
                    all_chunks[entry["source"]] = chunks
                del store

        status_chunks = [
            doc
            for doc, meta in all_chunks["gk_rf_368_379"]
            if "Статья 369" in meta["section_heading"]
        ]
        self.assertEqual(len(status_chunks), 1)
        self.assertIn("Утратила силу с 1 июня 2015 года", status_chunks[0])


class TestCorpusId(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp_dir.name)
        (self.dir / "a.txt").write_text("Статья 1. Текст одного источника.\n", encoding="utf-8")
        (self.dir / "b.txt").write_text("Статья 2. Текст другого источника.\n", encoding="utf-8")
        self.entries = [
            {
                "path": self.dir / "a.txt",
                "source": "a",
                "source_display": "A",
                "source_kind": "law",
                "doc_type": "statute",
            },
            {
                "path": self.dir / "b.txt",
                "source": "b",
                "source_display": "B",
                "source_kind": "law",
                "doc_type": "statute",
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deterministic_and_chroma_safe(self):
        first = compute_corpus_id(self.entries)
        self.assertEqual(first, compute_corpus_id(self.entries))
        self.assertRegex(first, r"^[0-9a-f]{12}$")
        self.assertRegex(collection_name_for(first), _CHROMA_NAME)

    def test_changes_when_file_content_changes(self):
        before = compute_corpus_id(self.entries)
        (self.dir / "a.txt").write_text("Статья 1. Изменённый текст.\n", encoding="utf-8")
        self.assertNotEqual(before, compute_corpus_id(self.entries))

    def test_changes_when_source_metadata_changes(self):
        before = compute_corpus_id(self.entries)
        self.entries[0]["source_display"] = "Другое название"
        self.assertNotEqual(before, compute_corpus_id(self.entries))

    def test_changes_when_entries_are_added_or_removed(self):
        self.assertNotEqual(compute_corpus_id(self.entries), compute_corpus_id(self.entries[:1]))

    def test_changes_with_indexing_settings(self):
        base = index_settings()
        before = compute_corpus_id(self.entries, settings=base)
        for key, value in (
            ("chunk_size", base["chunk_size"] + 1),
            ("chunk_overlap", base["chunk_overlap"] + 1),
            ("min_chunk_len", base["min_chunk_len"] + 1),
            ("embedding_model", "another-embedding-model"),
        ):
            changed = dict(base, **{key: value})
            self.assertNotEqual(before, compute_corpus_id(self.entries, settings=changed), key)

    def test_changes_when_chunking_strategy_or_citation_changes(self):
        before = compute_corpus_id(self.entries)
        self.entries[0]["chunker"] = "civil_code"
        with_chunker = compute_corpus_id(self.entries)
        self.assertNotEqual(before, with_chunker)
        self.entries[0]["citation"] = "ГК РФ"
        self.assertNotEqual(with_chunker, compute_corpus_id(self.entries))

    def test_defaults_are_explicit_generic_chunker_and_source_display(self):
        explicit = [dict(e, chunker="generic", citation=e["source_display"]) for e in self.entries]
        self.assertEqual(compute_corpus_id(self.entries), compute_corpus_id(explicit))

    def test_index_schema_version_covers_structure_aware_chunking(self):
        self.assertGreaterEqual(corpus_config.INDEX_SCHEMA_VERSION, 2)

    def test_changes_with_manual_corpus_version(self):
        with mock.patch.dict(os.environ, {"RAG_CORPUS_VERSION": "1"}):
            one = compute_corpus_id(self.entries)
        with mock.patch.dict(os.environ, {"RAG_CORPUS_VERSION": "2"}):
            two = compute_corpus_id(self.entries)
        self.assertNotEqual(one, two)

    def test_independent_of_location_and_line_endings(self):
        before = compute_corpus_id(self.entries)

        other = self.dir / "copy"
        other.mkdir()
        moved = []
        for entry in self.entries:
            target = other / Path(entry["path"]).name
            # write_text на Windows уже даёт CRLF: сначала приводим к LF, затем явно к CRLF.
            raw = Path(entry["path"]).read_bytes().replace(b"\r\n", b"\n")
            target.write_bytes(raw.replace(b"\n", b"\r\n"))
            moved.append(dict(entry, path=target))

        self.assertEqual(before, compute_corpus_id(moved))

    def test_relative_paths_resolve_against_base_dir(self):
        relative = [dict(e, path=Path(e["path"]).name) for e in self.entries]
        self.assertEqual(
            compute_corpus_id(self.entries),
            compute_corpus_id(relative, base_dir=self.dir),
        )

    def test_missing_file_raises_clear_error(self):
        self.entries[0]["path"] = self.dir / "missing.txt"
        with self.assertRaises(FileNotFoundError):
            compute_corpus_id(self.entries)


if __name__ == "__main__":
    unittest.main()
