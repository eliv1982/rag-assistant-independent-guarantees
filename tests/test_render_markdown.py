"""
Тесты безопасного Markdown rendering.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from rag_helpers import render_markdown_safe


def hrefs(html: str):
    """href только у настоящих тегов <a ...>: экранированный текст (&lt;a href=...) ссылкой не считается."""
    return re.findall(r'<a\b[^>]*?\bhref="([^"]*)"', html)


class TestRenderMarkdownSafe(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(render_markdown_safe(""), "")

    def test_bold_renders_strong(self):
        html = render_markdown_safe("**bold**")
        self.assertIn("<strong>bold</strong>", html)

    def test_script_is_stripped(self):
        html = render_markdown_safe("<script>alert(1)</script>")
        self.assertNotIn("<script", html.lower())


class TestSanitizerRemovesActiveContent(unittest.TestCase):
    def test_script_tag_and_its_body_are_removed(self):
        html = render_markdown_safe("до <script>alert('xss')</script> после")
        self.assertNotIn("script", html.lower())
        self.assertNotIn("alert", html)
        self.assertIn("до", html)
        self.assertIn("после", html)

    def test_script_variants_are_removed(self):
        for payload in (
            "<SCRIPT SRC=//evil.example/x.js></SCRIPT>",
            "<scr<script>ipt>alert(1)</scr</script>ipt>",
            "<script\n>alert(1)</script\n>",
            "<style>body{background:url(javascript:alert(1))}</style>",
        ):
            with self.subTest(payload=payload):
                html = render_markdown_safe(payload).lower()
                self.assertNotIn("<script", html)
                self.assertNotIn("<style", html)
                self.assertNotIn("alert(1)", html)

    def test_dangerous_tags_are_removed(self):
        for payload in (
            "<iframe src='https://evil.example'></iframe>",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)><circle/></svg>",
            "<object data='x'></object><embed src='x'>",
            "<form action='https://evil.example'><input name=p></form>",
            "<math><mi xlink:href='javascript:alert(1)'>x</mi></math>",
            "<meta http-equiv='refresh' content='0;url=https://evil.example'>",
            "<base href='https://evil.example/'>",
        ):
            with self.subTest(payload=payload):
                html = render_markdown_safe(payload).lower()
                for tag in ("<iframe", "<img", "<svg", "<object", "<embed", "<form", "<input", "<math", "<meta", "<base"):
                    self.assertNotIn(tag, html)
                self.assertNotIn("onerror", html)
                self.assertNotIn("onload", html)

    def test_event_handlers_style_and_target_attributes_are_stripped(self):
        html = render_markdown_safe(
            '<a href="https://ok.example" onclick="steal()" style="position:fixed" '
            'target="_blank" id="x" class="y">ссылка</a> '
            '<p onmouseover="steal()" style="color:red">текст</p>'
        )
        for attr in ("onclick", "onmouseover", "style", "target", "id=", "class="):
            self.assertNotIn(attr, html)
        self.assertIn('href="https://ok.example"', html)
        self.assertIn("ссылка", html)

    def test_html_comments_are_removed(self):
        html = render_markdown_safe("a <!-- <script>alert(1)</script> --> b")
        self.assertNotIn("<!--", html)
        self.assertNotIn("alert", html)


class TestSanitizerLinkProtocols(unittest.TestCase):
    UNSAFE = (
        "[x](javascript:alert(1))",
        "[x](JaVaScRiPt:alert(1))",
        "[x](  javascript:alert(1))",
        "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
        "[x](vbscript:msgbox(1))",
        "[x](file:///etc/passwd)",
        "[x](ftp://example.com/file)",
        '<a href="javascript:alert(1)">x</a>',
        '<a href="&#106;avascript:alert(1)">x</a>',
        '<a href="&#x6A;avascript:alert(1)">x</a>',
        '<a href="jav&#x09;ascript:alert(1)">x</a>',
        '<a href="java\tscript:alert(1)">x</a>',
        '<a href="\x01javascript:alert(1)">x</a>',
    )

    def test_unsafe_protocols_are_removed(self):
        for payload in self.UNSAFE:
            with self.subTest(payload=payload):
                html = render_markdown_safe(payload)
                self.assertEqual(hrefs(html), [], html)
                self.assertNotIn("javascript", html.lower())
                self.assertNotIn("data:", html.lower())
                self.assertIn("x</a>", html)  # текст ссылки остаётся

    def test_html_that_markdown_cannot_parse_is_inert_text(self):
        # '>' внутри атрибута ломает разбор HTML в Markdown: фрагмент выводится экранированным текстом.
        html = render_markdown_safe('<a href="data:text/html,<script>alert(1)</script>">x</a>')
        self.assertEqual(hrefs(html), [])
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("<a", html)
        self.assertIn("&lt;", html)

    def test_relative_and_protocol_relative_links_are_removed(self):
        for payload in ("[x](/stats)", "[x](//evil.example/path)", "[x](#anchor)", "[x](page.html)"):
            with self.subTest(payload=payload):
                self.assertEqual(hrefs(render_markdown_safe(payload)), [])

    def test_safe_protocols_are_kept(self):
        html = render_markdown_safe(
            "[a](https://example.com/a?x=1&y=2) [b](http://example.com/b) [c](mailto:user@example.com)"
        )
        self.assertEqual(
            hrefs(html),
            ["https://example.com/a?x=1&amp;y=2", "http://example.com/b", "mailto:user@example.com"],
        )

    def test_links_get_noopener_noreferrer(self):
        html = render_markdown_safe("[a](https://example.com)")
        self.assertIn('rel="noopener noreferrer"', html)

    def test_rel_cannot_be_overridden_from_raw_html(self):
        html = render_markdown_safe('<a href="https://example.com" rel="opener">a</a>')
        self.assertNotIn('rel="opener"', html)
        self.assertIn('rel="noopener noreferrer"', html)


class TestSafeMarkdownStillRenders(unittest.TestCase):
    def test_headings(self):
        html = render_markdown_safe("# Один\n\n## Два\n\n### Три\n\n#### Четыре")
        for level, title in enumerate(("Один", "Два", "Три", "Четыре"), start=1):
            self.assertIn(f"<h{level}>{title}</h{level}>", html)

    def test_paragraphs_and_line_breaks(self):
        html = render_markdown_safe("первый абзац\nвторая строка\n\nвторой абзац")
        self.assertEqual(html.count("<p>"), 2)
        self.assertIn("<br", html)

    def test_lists(self):
        html = render_markdown_safe("- а\n- б\n\n1. раз\n2. два")
        self.assertIn("<ul>", html)
        self.assertIn("<ol>", html)
        self.assertEqual(html.count("<li>"), 4)

    def test_emphasis_and_code(self):
        html = render_markdown_safe("**жирный** и *курсив* и `код`")
        self.assertIn("<strong>жирный</strong>", html)
        self.assertIn("<em>курсив</em>", html)
        self.assertIn("<code>код</code>", html)

    def test_fenced_code_block_is_escaped_not_executed(self):
        html = render_markdown_safe("```\n<script>alert(1)</script>\n```")
        self.assertIn("<pre>", html)
        self.assertIn("<code>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script", html)

    def test_blockquote(self):
        html = render_markdown_safe("> цитата статьи 368")
        self.assertIn("<blockquote>", html)
        self.assertIn("цитата статьи 368", html)

    def test_link_with_title(self):
        html = render_markdown_safe('[ГК РФ](https://example.com/gk "Гражданский кодекс")')
        self.assertIn('href="https://example.com/gk"', html)
        self.assertIn('title="Гражданский кодекс"', html)
        self.assertIn(">ГК РФ</a>", html)

    def test_plain_angle_brackets_are_text_not_markup(self):
        html = render_markdown_safe("если a &lt; b и 5 < 6, то все хорошо")
        self.assertIn("a &lt; b", html)
        self.assertIn("5 &lt; 6", html)


if __name__ == "__main__":
    unittest.main()
