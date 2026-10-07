"""
Тесты разрешения пути к SQLite-логам.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from db_logger import get_logs_db_path


class TestLogsDbPath(unittest.TestCase):
    def test_default_path_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            path = get_logs_db_path()
            self.assertTrue(path.endswith("logs.db"))
            self.assertIn("assistant_api", path.replace("\\", "/"))

    def test_env_overrides_default(self):
        custom = "/app/runtime/logs.db"
        with mock.patch.dict(os.environ, {"LOGS_DB_PATH": custom}, clear=True):
            self.assertEqual(get_logs_db_path(), custom)

    def test_blank_env_falls_back_to_default(self):
        for blank in ("", "   "):
            with mock.patch.dict(os.environ, {"LOGS_DB_PATH": blank}, clear=True):
                path = get_logs_db_path()
                self.assertTrue(path.endswith("logs.db"), blank)
                self.assertIn("assistant_api", path.replace("\\", "/"))

    def test_env_example_does_not_set_docker_path_for_local_runs(self):
        """.env.example копируют в .env для локального запуска: /app/runtime там быть не должно."""
        example = Path(__file__).resolve().parent.parent / ".env.example"
        active = [
            line.strip()
            for line in example.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for line in active:
            name, _, value = line.partition("=")
            if name in {"LOGS_DB_PATH", "RAG_CHROMA_PATH", "RAG_CACHE_DB_PATH"}:
                self.fail(f"{name} не должен быть активен в .env.example: {line}")
            self.assertNotIn("/app/runtime", value, line)


if __name__ == "__main__":
    unittest.main()
