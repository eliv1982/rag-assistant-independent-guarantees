"""
requirements*.txt должны быть ASCII: pip читает их в кодировке локали (cp1251 на русской Windows),
и кириллица в комментариях роняет `pip install -r` с UnicodeDecodeError. На Linux/CI (UTF-8) это не видно.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestRequirementsFilesAreAscii(unittest.TestCase):
    def test_requirements_files_are_ascii_only(self):
        files = sorted(ROOT.glob("requirements*.txt"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(file=path.name):
                try:
                    path.read_bytes().decode("ascii")
                except UnicodeDecodeError as exc:
                    self.fail(f"{path.name}: не-ASCII символ на позиции {exc.start}")


if __name__ == "__main__":
    unittest.main()
