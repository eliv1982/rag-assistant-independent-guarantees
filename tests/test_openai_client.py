"""
Тесты проверки OPENAI_API_KEY: пустое значение и заглушка из .env.example — не настоящий ключ.
Клиент OpenAI не обращается к сети при создании; запросы в тестах не выполняются.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from openai_client import api_key_configured, get_openai_client


def _env_example_key() -> str:
    example = Path(__file__).resolve().parent.parent / ".env.example"
    for line in example.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.partition("=")[2].strip()
    raise AssertionError("OPENAI_API_KEY не найден в .env.example")


class TestApiKeyConfigured(unittest.TestCase):
    def test_missing_blank_and_placeholder_are_not_configured(self):
        for value in (None, "", "   ", "your_openai_api_key_here", " your_openai_api_key_here "):
            with mock.patch.dict(os.environ):
                os.environ.pop("OPENAI_API_KEY", None)
                if value is not None:
                    os.environ["OPENAI_API_KEY"] = value
                self.assertFalse(api_key_configured(), repr(value))

    def test_real_looking_key_is_configured(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-0000000000"}):
            self.assertTrue(api_key_configured())

    def test_env_example_placeholder_is_recognized_as_placeholder(self):
        """Если заглушку в .env.example переименуют, проверка должна измениться вместе с ней."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": _env_example_key()}):
            self.assertFalse(api_key_configured())

    def test_get_openai_client_rejects_placeholder(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "your_openai_api_key_here"}):
            with self.assertRaises(ValueError):
                get_openai_client()

    def test_get_openai_client_builds_client_without_network(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-0000000000"}):
            client = get_openai_client()
        self.assertEqual(client.api_key, "sk-test-0000000000")


if __name__ == "__main__":
    unittest.main()
