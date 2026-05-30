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


if __name__ == "__main__":
    unittest.main()
