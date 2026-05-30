"""
Базовые тесты web UI без инициализации RAG pipeline.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from fastapi.testclient import TestClient

import web_app
from db_logger import DatabaseLogger


class TestWebAppBasic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_logger = web_app._logger
        self.original_pipeline = web_app._pipeline
        web_app._logger = None
        web_app._pipeline = None
        web_app._logger = DatabaseLogger(
            db_path=os.path.join(self.temp_dir.name, "test_logs.db")
        )

    def tearDown(self):
        web_app._logger = self.original_logger
        web_app._pipeline = self.original_pipeline
        self.temp_dir.cleanup()

    def test_health(self):
        client = TestClient(web_app.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_index_returns_200(self):
        client = TestClient(web_app.app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("RAG Assistant: Independent Guarantees", response.text)
        self.assertIn("Спросить ассистента", response.text)


if __name__ == "__main__":
    unittest.main()
