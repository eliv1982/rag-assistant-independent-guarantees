"""
Тесты инструкций prompt без OpenAI/API.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from rag_pipeline import PROMPT_INSTRUCTIONS


class TestPromptInstructions(unittest.TestCase):
    def test_prompt_requires_full_enumeration(self):
        self.assertIn("перечень", PROMPT_INSTRUCTIONS.lower())
        self.assertIn("извлеки все элементы перечня полностью", PROMPT_INSTRUCTIONS)

    def test_prompt_requires_gk_rf_and_urdg_blocks(self):
        self.assertIn("По ГК РФ", PROMPT_INSTRUCTIONS)
        self.assertIn("По URDG", PROMPT_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
