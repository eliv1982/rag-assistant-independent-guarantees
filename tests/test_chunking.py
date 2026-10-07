"""
Тесты нарезки корпуса с учётом структуры источников (без OpenAI, без сети).
Нарезка запускается на реальных файлах корпуса с фиксированными параметрами 800/200/80.
"""

import re
import sys
import unittest
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant_api"))

from chunk_splitter import (
    ChunkSizes,
    normalize_lines,
    pack_pieces,
    split_block,
    split_sentences,
    split_text,
)
from chunking import (
    ChunkingConfig,
    ChunkingError,
    build_chunks,
    known_chunkers,
    source_label,
)
from corpus_config import default_corpus_entries

CONFIG = ChunkingConfig(chunk_size=800, chunk_overlap=200, min_chunk_len=80)
SOFT_MAX = CONFIG.sizes.soft_max

# Крупные фрагменты допускаются только с письменным обоснованием: (источник, anchor) -> причина.
JUSTIFIED_OVERSIZE = {}

GK = "gk_rf_368_379"
FZ44 = "fz_44_art_45"
FZ223 = "fz_223_art_3_4"
PP1005 = "pp_1005_2013"
PP1397 = "pp_1397_2022"
VS2019 = "vs_independent_guarantee_2019"
VS2017 = "vs_contract_system_2017"


@lru_cache(maxsize=None)
def _entry(source):
    return next(e for e in default_corpus_entries() if e["source"] == source)


@lru_cache(maxsize=None)
def _text(source):
    return Path(_entry(source)["path"]).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def chunks_for(source):
    entry = _entry(source)
    return tuple(
        build_chunks(
            _text(source),
            source=entry["source"],
            source_display=entry["source_display"],
            source_kind=entry["source_kind"],
            doc_type=entry["doc_type"],
            chunker=entry["chunker"],
            citation=entry["citation"],
            config=CONFIG,
        )
    )


def all_chunks():
    return [(e["source"], c) for e in default_corpus_entries() for c in chunks_for(e["source"])]


def by(source, **conditions):
    return [c for c in chunks_for(source) if all(c.metadata.get(k) == v for k, v in conditions.items())]


def _item_numbers(chunks):
    """Номера подпунктов, покрытые диапазонами «1–3» в метаданных item."""
    numbers = set()
    for chunk in chunks:
        value = chunk.metadata.get("item")
        if not value:
            continue
        first, _, last = str(value).partition("–")
        numbers.update(range(int(first), int(last or first) + 1))
    return numbers


_HEADING_PREFIX = re.compile(r"^(§\s*\d+|Статья\s+\d|Глава\s|Раздел\s|Приложение\sN|Позиция\s+\d)")


def _is_heading_line(line):
    if any(c.isalpha() for c in line) and line == line.upper():
        return True
    if _HEADING_PREFIX.match(line) and not line.rstrip().endswith((".", ";", ":")):
        return True
    return bool(re.fullmatch(r"(?:[IVX]+\.|§\s*\d+\.)\s+[^.;:]{0,120}", line))


class TestChunkQuality(unittest.TestCase):
    def test_every_source_is_chunked_and_no_chunk_is_empty(self):
        for source, chunk in all_chunks():
            self.assertTrue(chunk.body.strip(), (source, chunk.metadata["anchor"]))
            self.assertIn(chunk.body, chunk.text)
        for entry in default_corpus_entries():
            self.assertGreater(len(chunks_for(entry["source"])), 0, entry["source"])

    def test_no_heading_only_chunks(self):
        for source, chunk in all_chunks():
            lines = [line for line in chunk.body.split("\n") if line.strip()]
            self.assertFalse(
                all(_is_heading_line(line) for line in lines),
                f"{source} {chunk.metadata['anchor']}: фрагмент состоит только из заголовков: {lines[:2]}",
            )
            self.assertGreaterEqual(len(chunk.body.split()), 3, (source, chunk.metadata["anchor"], chunk.body))

    def test_civil_code_section_heading_is_not_a_chunk(self):
        for chunk in chunks_for(GK):
            self.assertNotEqual(chunk.body.strip(), "§ 6. Независимая гарантия")
        # заголовок параграфа сохранён в метаданных статей
        self.assertTrue(all(c.metadata.get("section") == "§ 6. Независимая гарантия" for c in chunks_for(GK)))

    def test_no_mid_word_splits(self):
        for source, chunk in all_chunks():
            source_tokens = set(_text(source).split())
            for token in chunk.body.split():
                self.assertIn(token, source_tokens, f"{source} {chunk.metadata['anchor']}: «{token}» не слово исходника")

    def test_chunks_start_at_sentence_or_unit_boundaries(self):
        """Фрагмент не начинается с союза/строчной буквы, кроме продолжения подпунктов и перечислений."""
        for source, chunk in all_chunks():
            first = chunk.body.split("\n", 1)[0]
            if first[:1].isalpha() and first[:1].islower():
                # допустимо: маркированные строки перечней («права заказчика ...», «документ, ...»)
                self.assertGreater(len(first), 25, (source, chunk.metadata["anchor"], first))

    def test_no_pathological_oversized_chunks(self):
        for source, chunk in all_chunks():
            if len(chunk.body) > SOFT_MAX:
                key = (source, chunk.metadata["anchor"])
                self.assertIn(
                    key,
                    JUSTIFIED_OVERSIZE,
                    f"{key}: {len(chunk.body)} символов > {SOFT_MAX} без обоснования",
                )

    def test_pp1397_known_defect_is_gone(self):
        """Раньше PP_1397 давал фрагмент ~8,7 тыс. символов при цели ~800."""
        biggest = max(len(c.body) for c in chunks_for(PP1397))
        self.assertLessEqual(biggest, SOFT_MAX)
        self.assertLess(biggest, 2 * CONFIG.chunk_size)

    def test_size_distribution_is_bounded(self):
        sizes = sorted(len(c.body) for _, c in all_chunks())
        p95 = sizes[int(0.95 * (len(sizes) - 1))]
        self.assertLessEqual(p95, CONFIG.chunk_size * 1.2)
        self.assertLessEqual(sizes[-1], SOFT_MAX)

    def test_metadata_is_chroma_compatible_and_anchors_are_unique(self):
        seen = set()
        for source, chunk in all_chunks():
            for key, value in chunk.metadata.items():
                self.assertIsInstance(value, (str, int), (source, key))
                self.assertNotIsInstance(value, bool)
                if isinstance(value, str):
                    self.assertTrue(value, (source, key))
            meta = chunk.metadata
            self.assertEqual(meta["source"], source)
            for key in ("source_display", "source_kind", "doc_type", "section_heading", "citation", "anchor"):
                self.assertIn(key, meta)
            self.assertLess(meta["chunk_index"], meta["chunk_count"])
            identity = (source, meta["anchor"], meta["chunk_index"])
            self.assertNotIn(identity, seen)
            seen.add(identity)

    def test_chunk_text_has_source_and_context_header(self):
        for source, chunk in all_chunks():
            self.assertTrue(chunk.text.startswith(f"[Источник: {chunk.metadata['source_display']} | "))
            self.assertIn(f"[Фрагмент: {chunk.metadata['section_heading']}]", chunk.text)

    def test_chunking_is_deterministic(self):
        entry = _entry(PP1397)
        again = build_chunks(
            _text(PP1397),
            source=entry["source"],
            source_display=entry["source_display"],
            source_kind=entry["source_kind"],
            doc_type=entry["doc_type"],
            chunker=entry["chunker"],
            citation=entry["citation"],
            config=CONFIG,
        )
        self.assertEqual([(c.text, c.metadata) for c in again], [(c.text, c.metadata) for c in chunks_for(PP1397)])

    def test_crlf_and_extra_whitespace_do_not_change_chunks(self):
        entry = _entry(FZ44)
        noisy = _text(FZ44).replace("\n", "\r\n").replace(" ", "  ", 50) + "\r\n\r\n"
        again = build_chunks(
            noisy,
            source=entry["source"],
            source_display=entry["source_display"],
            source_kind=entry["source_kind"],
            doc_type=entry["doc_type"],
            chunker=entry["chunker"],
            citation=entry["citation"],
            config=CONFIG,
        )
        self.assertEqual([c.text for c in again], [c.text for c in chunks_for(FZ44)])


def _apparatus_lines(source, lines):
    """Строки, которые разборщик осознанно превращает в структуру (метаданные/подписи), а не в текст."""
    skip = set()
    for i, line in enumerate(lines):
        if source == GK and re.match(r"^§\s*\d+", line):
            skip.add(i)
        if source in (PP1005, PP1397) and re.fullmatch(r"(?i)утвержден[оаы]?", line):
            j = i
            while j < min(len(lines), i + 5):
                skip.add(j)
                if re.search(r"\bN\s*\d+\s*$", lines[j]):
                    break
                j += 1
        if source == PP1397 and re.fullmatch(r"Приложение N \d+", line):
            j = i
            while j < len(lines) and lines[j]:
                skip.add(j)
                j += 1
        if source == PP1397 and re.match(r"^(?:I{1,3}|IV|VI{0,3}|IX|X)\.\s+\S", line) and (i == 0 or not lines[i - 1]):
            j = i
            while j < len(lines) and lines[j]:
                skip.add(j)
                j += 1
        if source == VS2019 and re.fullmatch(r"Позиция \d+", line):
            skip.add(i)
        if source == VS2017 and line == line.upper() and any(c.isalpha() for c in line) and i > 5:
            skip.add(i)
    return skip


class TestNoSilentTruncation(unittest.TestCase):
    def test_every_source_word_is_present_in_chunks(self):
        for entry in default_corpus_entries():
            source = entry["source"]
            lines = _text(source).replace("\r\n", "\n").split("\n")
            skip = _apparatus_lines(source, lines)
            source_words = Counter()
            for i, line in enumerate(lines):
                if i not in skip:
                    source_words.update(line.split())
            chunk_words = Counter()
            for chunk in chunks_for(source):
                # заголовки статей и разделов живут в подписи фрагмента (section_heading), а не в теле
                chunk_words.update(chunk.body.split())
                chunk_words.update(chunk.metadata["section_heading"].split())
            missing = source_words - chunk_words
            self.assertFalse(missing, f"{source}: слова исходника отсутствуют во фрагментах: {list(missing.items())[:8]}")

    def test_every_substantive_source_line_is_in_some_chunk(self):
        for entry in default_corpus_entries():
            source = entry["source"]
            joined = "\n".join(c.body + "\n" + c.metadata["section_heading"] for c in chunks_for(source))
            lines = _text(source).replace("\r\n", "\n").split("\n")
            skip = _apparatus_lines(source, lines)
            for i, line in enumerate(lines):
                line = " ".join(line.split())
                if i in skip or len(line) < 30:
                    continue
                covered = line in joined or all(s in joined for s in split_sentences(line))
                if not covered:
                    # Одно предложение длиннее фрагмента поделено по «;»/«,»: каждая четвёрка слов подряд
                    # должна встретиться, кроме тех, что пересекают границу реза (до 3 на каждый рез).
                    flat = joined.replace("\n", " ")
                    words = line.split()
                    windows = [" ".join(words[k : k + 4]) for k in range(max(1, len(words) - 3))]
                    missing = [w for w in windows if w not in flat]
                    cuts = len(line) // CONFIG.chunk_size + 1
                    covered = len(missing) <= 3 * cuts
                self.assertTrue(covered, f"{source}, строка {i + 1}: {line[:80]}")


class TestCivilCode(unittest.TestCase):
    def test_every_article_is_traceable(self):
        articles = {c.metadata["article"] for c in chunks_for(GK)}
        self.assertEqual(
            articles,
            {"368", "369", "370", "371", "372", "373", "374", "375", "375.1", "376", "377", "378", "379"},
        )
        for chunk in chunks_for(GK):
            self.assertTrue(chunk.metadata["citation"].startswith(f"ГК РФ, ст. {chunk.metadata['article']}"))

    def test_article_heading_is_repeated_in_every_chunk_of_a_split_article(self):
        for article, title in (
            ("368", "Понятие и форма независимой гарантии"),
            ("376", "Отказ гаранта удовлетворить требование бенефициара"),
        ):
            chunks = by(GK, article=article)
            self.assertGreater(len(chunks), 1, article)
            for chunk in chunks:
                self.assertTrue(chunk.metadata["section_heading"].startswith(f"Статья {article}. {title}"))
            self.assertEqual(
                {c.metadata["point"] for c in chunks}, {"1", "2", "3", "4", "5"}, f"статья {article}: пункты"
            )

    def test_small_articles_stay_whole(self):
        for article in ("370", "373", "375.1", "377"):
            self.assertEqual(len(by(GK, article=article)), 1, article)

    def test_point_level_anchor_for_suspension_grounds(self):
        (chunk,) = by(GK, article="376", point="2")
        self.assertEqual(chunk.metadata["citation"], "ГК РФ, ст. 376, п. 2")
        self.assertIn("приостановить платеж", chunk.body)
        self.assertIn("4) исполнение по основному обязательству", chunk.body)

    def test_repealed_article_369_is_a_single_status_chunk(self):
        (chunk,) = by(GK, article="369")
        self.assertIn("Утратила силу с 1 июня 2015 года", chunk.body)
        self.assertEqual(chunk.metadata["citation"], "ГК РФ, ст. 369")


class TestFederalLaws(unittest.TestCase):
    PARTS_44 = [
        "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "2", "3", "3.1", "4", "5", "6", "7",
        "8", "8.1", "8.2", "9", "10", "11", "12", "13",
    ]

    def test_44fz_every_part_has_its_own_anchor(self):
        parts = []
        for chunk in chunks_for(FZ44):
            part = chunk.metadata["part"]
            if part not in parts:
                parts.append(part)
            self.assertEqual(chunk.metadata["article"], "45")
            self.assertTrue(chunk.metadata["citation"].startswith(f"44-ФЗ, ст. 45, ч. {part}"))
        self.assertEqual(parts, self.PARTS_44)

    def test_44fz_part_6_is_attributable(self):
        (chunk,) = by(FZ44, part="6")
        self.assertEqual(chunk.metadata["citation"], "44-ФЗ, ст. 45, ч. 6")
        self.assertIn("Основанием для отказа в принятии независимой гарантии", chunk.body)
        self.assertIn("Статья 45", chunk.metadata["section_heading"])
        self.assertIn("ч. 6", chunk.metadata["section_heading"])

    def test_44fz_large_part_is_split_on_items_and_keeps_part_context(self):
        chunks = by(FZ44, part="2")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(_item_numbers(chunks), set(range(1, 8)))
        lead = "2. Независимая гарантия должна быть безотзывной и должна содержать:"
        for chunk in chunks:
            self.assertTrue(chunk.body.startswith(lead), chunk.body[:80])
            self.assertIn("ч. 2", chunk.metadata["section_heading"])
            self.assertTrue(chunk.metadata["citation"].startswith("44-ФЗ, ст. 45, ч. 2, п. "))
        # подпункт целиком в одном фрагменте: пункт 3 про неустойку
        penalty = [c for c in chunks if "0,1 процента" in c.body]
        self.assertEqual(len(penalty), 1)

    def test_223fz_selected_parts_are_separately_traceable(self):
        parts = []
        for chunk in chunks_for(FZ223):
            if chunk.metadata["part"] not in parts:
                parts.append(chunk.metadata["part"])
            self.assertEqual(chunk.metadata["article"], "3.4")
        self.assertEqual(parts, ["12", "14.1", "14.2", "14.3", "17", "31", "32"])

    def test_223fz_part_14_1_keeps_item_structure(self):
        chunks = by(FZ223, part="14.1")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(_item_numbers(chunks), {1, 2, 3, 4})
        for chunk in chunks:
            self.assertTrue(chunk.body.startswith("14.1. Независимая гарантия"))
            self.assertTrue(chunk.metadata["citation"].startswith("223-ФЗ, ст. 3.4, ч. 14.1"))
        item_4 = [c for c in chunks if "4) независимая гарантия должна содержать" in c.body]
        self.assertEqual(len(item_4), 1)
        self.assertIn("менее одного месяца", item_4[0].body)

    def test_223fz_part_32_keeps_item_structure(self):
        chunks = by(FZ223, part="32")
        self.assertEqual(_item_numbers(chunks), {1, 2, 3, 4, 5})
        self.assertTrue(all(c.body.startswith("32. Правительство Российской Федерации вправе установить:") for c in chunks))

    def test_223fz_citation_matches_brief_example(self):
        citations = {c.metadata["citation"] for c in by(FZ223, part="14.2")}
        self.assertEqual(citations, {"223-ФЗ, ст. 3.4, ч. 14.2"})


class TestGovernmentResolutions(unittest.TestCase):
    def test_pp1005_has_all_seven_annexes_and_the_resolution(self):
        roots = []
        for chunk in chunks_for(PP1005):
            root = chunk.metadata["anchor"].split("/")[0]
            if root not in roots:
                roots.append(root)
        self.assertEqual(
            roots,
            [
                "resolution",
                "requirements",
                "document_list",
                "registry_rules",
                "payment_demand_form",
                "closed_registry_rules",
                "bid_guarantee_form",
                "contract_guarantee_form",
            ],
        )

    def test_pp1005_contract_form_clause_10_matches_brief_example(self):
        (chunk,) = by(PP1005, form="contract_guarantee_form", point="10")
        self.assertEqual(
            chunk.metadata["citation"],
            "ПП РФ №1005, типовая форма обеспечения исполнения контракта, п. 10",
        )
        self.assertTrue(chunk.body.startswith("10. Гарант обязан уплатить бенефициару"))
        self.assertIn("Типовая форма независимой гарантии (обеспечение исполнения контракта)", chunk.metadata["section_heading"])

    def test_pp1005_forms_split_on_numbered_clauses(self):
        for form, count in (("bid_guarantee_form", 16), ("contract_guarantee_form", 17)):
            points = {c.metadata["point"] for c in by(PP1005, form=form) if "point" in c.metadata}
            self.assertEqual(points, {str(n) for n in range(1, count + 1)}, form)
            self.assertLessEqual(max(len(c.body) for c in by(PP1005, form=form)), SOFT_MAX)

    def test_pp1005_form_header_and_footnotes_are_kept_as_their_own_chunks(self):
        head = by(PP1005, form="contract_guarantee_form", anchor="contract_guarantee_form/head")
        self.assertEqual(len(head), 1)
        self.assertIn("Полное наименование гаранта", head[0].body)
        tail = by(PP1005, form="contract_guarantee_form", anchor="contract_guarantee_form/tail")
        self.assertIn("<8-1>", " ".join(c.body for c in tail))

    def test_pp1005_registry_rules_and_point_12_1(self):
        (chunk,) = by(PP1005, anchor="registry_rules/p.12(1)")
        self.assertEqual(chunk.metadata["citation"], "ПП РФ №1005, правила ведения реестра, п. 12(1)")

    def test_pp1397_regulation_points_are_addressable(self):
        points = [c.metadata["point"] for c in chunks_for(PP1397) if c.metadata["anchor"].startswith("regulation/")]
        self.assertEqual(sorted(set(points), key=int), [str(n) for n in range(1, 11)])
        (chunk,) = by(PP1397, anchor="regulation/p.7")
        self.assertEqual(chunk.metadata["citation"], "ПП РФ №1397, п. 7")
        self.assertTrue(chunk.body.startswith("7. Независимая гарантия не должна содержать условия:"))
        self.assertTrue(chunk.metadata["section"].startswith("II."))

    def test_pp1397_large_point_5_is_split_on_subitems_and_keeps_context(self):
        chunks = by(PP1397, anchor="regulation/p.5/item.а") + [
            c for c in chunks_for(PP1397) if c.metadata["anchor"].startswith("regulation/p.5/") and "item.а" not in c.metadata["anchor"]
        ]
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["point"], "5")
            self.assertTrue(chunk.metadata["citation"].startswith("ПП РФ №1397, п. 5, пп. "))
            self.assertTrue(chunk.body.startswith("5. Независимая гарантия"))

    def test_pp1397_appendices_are_forms_with_numbered_clauses(self):
        expected = {
            1: ("bid_guarantee_form", 17),
            2: ("bid_demand_form", 3),
            3: ("contract_guarantee_form", 18),
            4: ("contract_demand_form", 3),
        }
        for appendix, (form, count) in expected.items():
            chunks = by(PP1397, appendix=appendix)
            self.assertTrue(chunks, appendix)
            self.assertEqual({c.metadata["form"] for c in chunks}, {form})
            points = {c.metadata["point"] for c in chunks if "point" in c.metadata}
            self.assertEqual(points, {str(n) for n in range(1, count + 1)}, f"приложение {appendix}")
            self.assertLessEqual(max(len(c.body) for c in chunks), SOFT_MAX)

    def test_pp1397_form_clause_citation(self):
        (chunk,) = by(PP1397, appendix=3, point="14")
        self.assertEqual(
            chunk.metadata["citation"],
            "ПП РФ №1397, прил. 3, типовая форма обеспечения исполнения договора, п. 14",
        )
        self.assertTrue(chunk.body.startswith("14. Исключение банка"))

    def test_pp1397_resolution_points_are_separate_from_regulation_points(self):
        (chunk,) = by(PP1397, anchor="resolution/p.3")
        self.assertEqual(chunk.metadata["citation"], "ПП РФ №1397, постановление, п. 3")

    def test_pp1397_title_is_not_a_chunk_on_its_own(self):
        first = [c for c in chunks_for(PP1397) if c.metadata["anchor"].startswith("regulation/p.1")][0]
        self.assertIn("1. Настоящее Положение устанавливает", first.body)


class TestCourtReviews(unittest.TestCase):
    def test_2019_review_has_all_17_positions_individually(self):
        positions = sorted({c.metadata["position"] for c in chunks_for(VS2019) if "position" in c.metadata})
        self.assertEqual(positions, list(range(1, 18)))
        for number in range(1, 18):
            for chunk in by(VS2019, position=number):
                self.assertEqual(chunk.metadata["citation"], f"Обзор ВС РФ от 05.06.2019, позиция {number}")
                self.assertTrue(chunk.metadata["section_heading"].startswith(f"Позиция {number}. "))
                self.assertEqual(chunk.metadata["anchor"], f"pos.{number}")

    def test_2019_positions_do_not_share_one_generic_heading(self):
        headings = {c.metadata["position"]: c.metadata["section_heading"] for c in chunks_for(VS2019) if "position" in c.metadata}
        self.assertEqual(len(set(headings.values())), 17)
        self.assertIn("односторонн", headings[1])
        self.assertIn("анкротств", headings[14])

    def test_2019_long_position_keeps_number_and_thesis_in_every_chunk(self):
        chunks = by(VS2019, position=11)
        self.assertGreater(len(chunks), 1)
        thesis = chunks[0].metadata["section_heading"]
        self.assertIn("Обязательство гаранта перед бенефициаром не зависит", thesis)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["section_heading"], thesis)
            self.assertIn(f"[Фрагмент: {thesis}]", chunk.text)
        self.assertEqual([c.metadata["chunk_index"] for c in chunks], list(range(len(chunks))))

    def test_2019_position_text_is_not_mixed_between_positions(self):
        (preamble,) = by(VS2019, anchor="preamble")
        self.assertIn("ОБЗОР судебной практики", preamble.body)
        self.assertNotIn("position", preamble.metadata)
        for chunk in by(VS2019, position=14):
            self.assertNotIn("Позиция 15", chunk.body)

    def test_2017_review_retained_positions_are_independent(self):
        positions = sorted({c.metadata["position"] for c in chunks_for(VS2017) if "position" in c.metadata})
        self.assertEqual(positions, [25, 30])
        for number in (25, 30):
            chunks = by(VS2017, position=number)
            self.assertGreaterEqual(len(chunks), 1)
            for chunk in chunks:
                self.assertEqual(chunk.metadata["citation"], f"Обзор ВС РФ от 28.06.2017, позиция {number}")
                self.assertTrue(chunk.metadata["section_heading"].startswith(f"Позиция {number}. "))
        self.assertIn("уклонившимся", " ".join(c.body for c in by(VS2017, position=25)))
        self.assertIn("возмещение убытков", " ".join(c.body for c in by(VS2017, position=30)))

    def test_2017_section_title_goes_to_metadata_and_intro_stays_separate(self):
        positions = by(VS2017, position=25)
        self.assertIn("РЕЛЕВАНТНЫЕ ПРАВОВЫЕ ПОЗИЦИИ", positions[0].metadata["section"])
        intro = by(VS2017, anchor="preamble")
        self.assertGreaterEqual(len(intro), 1)
        self.assertTrue(all("position" not in c.metadata for c in intro))


class TestSourceRendering(unittest.TestCase):
    def test_narrow_label_is_used_when_available(self):
        examples = {
            (GK, "376", "2"): "ГК РФ, ст. 376, п. 2",
        }
        for (source, article, point), expected in examples.items():
            (chunk,) = by(source, article=article, point=point)
            self.assertEqual(source_label(chunk.metadata), expected)
        self.assertEqual(source_label(by(FZ44, part="6")[0].metadata), "44-ФЗ, ст. 45, ч. 6")
        self.assertEqual(source_label(by(FZ223, part="14.1")[0].metadata)[:24], "223-ФЗ, ст. 3.4, ч. 14.1")
        self.assertEqual(source_label(by(PP1397, anchor="regulation/p.7")[0].metadata), "ПП РФ №1397, п. 7")
        self.assertEqual(source_label(by(VS2019, position=11)[0].metadata), "Обзор ВС РФ от 05.06.2019, позиция 11")
        self.assertEqual(source_label(by(VS2017, position=30)[0].metadata), "Обзор ВС РФ от 28.06.2017, позиция 30")

    def test_label_falls_back_to_source_display_then_source(self):
        self.assertEqual(source_label({"source_display": "ГК РФ, ст. 368–379"}), "ГК РФ, ст. 368–379")
        self.assertEqual(source_label({"source": "x"}), "x")
        self.assertEqual(source_label({}), "источник")
        self.assertEqual(source_label(None), "источник")

    def test_every_chunk_has_a_narrower_label_than_the_source_display(self):
        for source, chunk in all_chunks():
            self.assertNotEqual(source_label(chunk.metadata), chunk.metadata["source_display"], source)


class TestChunkerSelection(unittest.TestCase):
    def test_unknown_chunker_is_an_error(self):
        with self.assertRaises(ChunkingError):
            build_chunks("Текст", source="s", source_display="S", source_kind="law", doc_type="statute", chunker="nope")

    def test_structured_chunkers_fail_loudly_without_anchors(self):
        for chunker in ("civil_code", "federal_law", "government_resolution", "court_review"):
            with self.assertRaises(ChunkingError, msg=chunker):
                build_chunks(
                    "Просто текст без нумерации и статей.",
                    source="s",
                    source_display="S",
                    source_kind="law",
                    doc_type="statute",
                    chunker=chunker,
                )

    def test_generic_chunker_handles_arbitrary_text(self):
        text = "\n\n".join(f"Абзац номер {i}. " + "Слово " * 60 for i in range(12))
        chunks = build_chunks(text, source="s", source_display="S", source_kind="unknown", doc_type="overview", config=CONFIG)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.body.strip() for c in chunks))
        self.assertTrue(all(len(c.body) <= SOFT_MAX + CONFIG.chunk_overlap for c in chunks))

    def test_known_chunkers_match_corpus_config(self):
        for entry in default_corpus_entries():
            self.assertIn(entry["chunker"], known_chunkers())


class TestSplitterPrimitives(unittest.TestCase):
    SIZES = ChunkSizes(target=200, soft_max=300, min_len=40)

    def test_sentences_respect_abbreviations_and_numbers(self):
        text = "Согласно ст. 45 Закона, п. 2 применяется и т.д. 14.1. Далее идет другое предложение. Конец."
        self.assertEqual(
            split_sentences(text),
            ["Согласно ст. 45 Закона, п. 2 применяется и т.д. 14.1.", "Далее идет другое предложение.", "Конец."],
        )

    def test_list_marker_is_not_a_sentence(self):
        self.assertEqual(split_sentences("14.1. Независимая гарантия должна соответствовать требованиям."),
                         ["14.1. Независимая гарантия должна соответствовать требованиям."])

    def test_pack_pieces_never_exceeds_limit_unless_piece_is_longer(self):
        pieces = ["a" * 30] * 10 + ["b" * 120]
        groups = pack_pieces(pieces, 100, " ")
        self.assertTrue(all(len(g) <= 100 or g == "b" * 120 for g in groups))
        self.assertEqual(" ".join(groups).split(), [p for p in pieces])

    def test_small_tail_is_merged_into_previous_group(self):
        groups = pack_pieces(["x" * 90, "y" * 5], 95, " ", min_tail=20, max_len=120)
        self.assertEqual(len(groups), 1)

    def test_split_text_keeps_every_word_and_never_cuts_words(self):
        text = " ".join(f"слово{i}" for i in range(300))
        parts = split_text(text, 120)
        self.assertGreater(len(parts), 1)
        self.assertEqual(" ".join(parts).split(), text.split())
        self.assertTrue(all(len(p) <= 120 for p in parts))

    def test_split_text_prefers_sentence_then_clause_boundaries(self):
        text = "Первое предложение довольно длинное и содержательное. Второе предложение тоже длинное и ясное."
        parts = split_text(text, 60)
        self.assertEqual(parts, ["Первое предложение довольно длинное и содержательное.", "Второе предложение тоже длинное и ясное."])
        clauses = split_text("первая часть фразы; вторая часть фразы; третья часть фразы", 30)
        self.assertEqual(clauses, ["первая часть фразы;", "вторая часть фразы;", "третья часть фразы"])

    def test_split_block_keeps_small_unit_whole(self):
        lines = ["5. Вводная строка:", "1) первое", "2) второе"]
        pieces = split_block(lines, self.SIZES)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].text, "\n".join(lines))
        self.assertEqual((pieces[0].item, pieces[0].subitem), ("", ""))

    def test_split_block_splits_on_items_and_repeats_lead(self):
        lines = ["5. Требования к гарантии:"] + [f"{i}) " + f"условие номер {i} " * 6 for i in range(1, 7)]
        pieces = split_block(lines, self.SIZES)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertTrue(piece.text.startswith("5. Требования к гарантии:"))
            self.assertTrue(piece.item)
        rejoined = "\n".join(line for piece in pieces for line in piece.text.split("\n")[1:])
        self.assertEqual(rejoined, "\n".join(lines[1:]))

    def test_split_block_nested_letters_get_subitem_labels(self):
        lines = (
            ["14. Общая вводная часть:", "4) гарантия должна содержать:"]
            + [f"{letter}) " + f"подробное условие {letter} " * 12 for letter in "абв"]
        )
        pieces = split_block(lines, ChunkSizes(target=400, soft_max=500, min_len=40))
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(p.item == "4" and p.subitem for p in pieces))
        self.assertTrue(all("4) гарантия должна содержать:" in p.text for p in pieces))

    def test_leading_caps_title_is_glued_to_content(self):
        lines = ["ПОЛОЖЕНИЕ О ГАРАНТИЯХ", "Длинный абзац. " + "Предложение номер один. " * 30]
        pieces = split_block(lines, self.SIZES, hierarchical=False)
        self.assertTrue(pieces[0].text.startswith("ПОЛОЖЕНИЕ О ГАРАНТИЯХ\nДлинный абзац."))
        self.assertTrue(all(p.text != "ПОЛОЖЕНИЕ О ГАРАНТИЯХ" for p in pieces))

    def test_normalize_lines(self):
        text = "Строка один  \r\n\r\n\r\n\tСтрока  два\r\n\r\n"
        self.assertEqual(normalize_lines(text), ["Строка один", "", "Строка два"])


if __name__ == "__main__":
    unittest.main()
