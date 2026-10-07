"""
Тесты инструкций prompt без OpenAI/API.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from rag_pipeline import (
    LEGAL_SYSTEM_PROMPT,
    PROMPT_INSTRUCTIONS,
    PROMPT_INTRO,
    PROMPT_VERSION,
)

_ROOT = Path(__file__).resolve().parent.parent


class TestPromptInstructions(unittest.TestCase):
    def test_prompt_handles_list_questions(self):
        self.assertIn("перечень", PROMPT_INSTRUCTIONS.lower())
        self.assertIn("нумерованным списком", PROMPT_INSTRUCTIONS)

    def test_prompt_does_not_demand_unsupported_exhaustive_lists(self):
        """Модель не должна быть вынуждена «дособирать» перечень, которого нет во фрагментах."""
        combined = f"{LEGAL_SYSTEM_PROMPT}\n{PROMPT_INSTRUCTIONS}".lower()
        for pressure in (
            "извлеки все элементы перечня",
            "извлекай полный перечень",
            "список всех элементов",
            "всех элементов из контекста",
            "полностью",
        ):
            self.assertNotIn(pressure, combined)
        # вместо этого: только названное во фрагментах и честная оговорка о неполноте
        self.assertIn("прямо названы", combined)
        self.assertIn("не добавляй элементы, которых во фрагментах нет", combined)
        self.assertIn("не называй перечень полным", combined)
        self.assertIn("может быть неполным", combined)
        self.assertIn("не дополняй перечень по памяти", combined)

    def test_prompt_keeps_grounding_discipline(self):
        self.assertIn("Отвечай только на основании найденного контекста", PROMPT_INSTRUCTIONS)
        self.assertIn("Не придумывай номера статей", PROMPT_INSTRUCTIONS)
        self.assertIn("Отвечай строго на основании переданных фрагментов", LEGAL_SYSTEM_PROMPT)

    def test_prompt_asks_for_narrow_source_format(self):
        self.assertIn("«44-ФЗ, ст. 45, ч. 6»", PROMPT_INSTRUCTIONS)
        self.assertIn("позиция 11", PROMPT_INSTRUCTIONS)

    def test_prompt_describes_all_source_layers(self):
        for expected in (
            "ГК РФ",
            "44-ФЗ",
            "223-ФЗ",
            "№ 1005",
            "№ 1397",
            "обзоры практики ВС РФ",
        ):
            self.assertIn(expected, PROMPT_INSTRUCTIONS)

    def test_prompt_requires_source_blocks_and_attribution(self):
        self.assertIn("По ГК РФ", PROMPT_INSTRUCTIONS)
        self.assertIn("По 44-ФЗ / 223-ФЗ", PROMPT_INSTRUCTIONS)
        self.assertIn("Коротко", PROMPT_INSTRUCTIONS)
        self.assertIn("укажи источник", PROMPT_INSTRUCTIONS)

    def test_prompt_handles_repealed_provisions(self):
        self.assertIn("утратила силу", PROMPT_INSTRUCTIONS)

    def test_prompt_intro_names_current_sources(self):
        self.assertIn("44-ФЗ", PROMPT_INTRO)
        self.assertIn("223-ФЗ", PROMPT_INTRO)

    def test_prompt_version_is_short_stable_hash(self):
        self.assertRegex(PROMPT_VERSION, r"^[0-9a-f]{8}$")


class TestNoStaleSourceReferences(unittest.TestCase):
    """В поддерживаемых файлах не должно остаться ссылок на исключённый из корпуса источник."""

    NEEDLE = "urd" + "g"

    def test_prompts_have_no_removed_source(self):
        for text in (LEGAL_SYSTEM_PROMPT, PROMPT_INTRO, PROMPT_INSTRUCTIONS):
            self.assertNotIn(self.NEEDLE, text.lower())

    def test_application_files_have_no_removed_source(self):
        files = [
            _ROOT / "README.md",
            *(_ROOT / "assistant_api").glob("*.py"),
            *(_ROOT / "assistant_api" / "templates").glob("*.html"),
            *(_ROOT / "tests").glob("*.py"),
        ]
        self.assertGreater(len(files), 10)
        for path in files:
            self.assertNotIn(self.NEEDLE, path.read_text(encoding="utf-8").lower(), str(path))


if __name__ == "__main__":
    unittest.main()
