"""
Тесты безопасного Markdown rendering.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from rag_helpers import render_markdown_safe


class TestRenderMarkdownSafe(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(render_markdown_safe(""), "")

    def test_bold_renders_strong(self):
        html = render_markdown_safe("**bold**")
        self.assertIn("<strong>bold</strong>", html)

    def test_script_is_stripped(self):
        html = render_markdown_safe("<script>alert(1)</script>")
        self.assertNotIn("<script", html.lower())


if __name__ == "__main__":
    unittest.main()
