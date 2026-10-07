"""
Нарезка корпуса с учётом структуры источника (детерминированная, без LangChain/LlamaIndex).

Каждая стратегия разбирает свой тип источника на «блоки» — юридически адресуемые единицы
(статья, часть, пункт, пункт типовой формы, позиция обзора). Блок получает узкую привязку
(метаданные + человекочитаемая ссылка), затем при необходимости делится по подпунктам и предложениям
(см. chunk_splitter). В каждом фрагменте повторяется контекст: заголовок статьи/раздела/позиции.

Стратегии (поле chunker в corpus_config):
- civil_code            ГК РФ: статья -> пункт (если статья крупная);
- federal_law           44-ФЗ, 223-ФЗ: статья -> часть -> пункт/подпункт;
- government_resolution постановления Правительства: пункты, разделы, приложения, типовые формы;
- court_review          обзоры практики ВС РФ: нумерованные позиции;
- generic               запасной вариант для произвольного текста (абзацы -> предложения).
"""

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from .chunk_splitter import (
        SOFT_MAX_FACTOR,
        ChunkSizes,
        normalize_lines,
        pack_pieces,
        split_block,
        split_sentences,
        split_text,
    )
except ImportError:
    from chunk_splitter import (
        SOFT_MAX_FACTOR,
        ChunkSizes,
        normalize_lines,
        pack_pieces,
        split_block,
        split_sentences,
        split_text,
    )

MetaValue = Union[str, int]

KIND_LABELS = {
    "law": "закон РФ",
    "procurement_law": "закон о закупках",
    "government_resolution": "постановление Правительства РФ",
    "case_law_summary": "обзор судебной практики",
}

_THESIS_MAX = 300


class ChunkingError(ValueError):
    """Источник не соответствует ожидаемой структуре (индекс не должен строиться «молча» хуже)."""


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 200
    min_chunk_len: int = 80

    @property
    def sizes(self) -> ChunkSizes:
        return ChunkSizes(
            target=self.chunk_size,
            soft_max=int(self.chunk_size * SOFT_MAX_FACTOR),
            min_len=self.min_chunk_len,
        )


@dataclass
class Block:
    """Юридически адресуемая единица до деления по размеру."""

    lines: List[str]
    label: str  # строка «[Фрагмент: ...]»
    citation: str  # «44-ФЗ, ст. 45, ч. 6»
    anchor: str  # стабильный адрес внутри источника: art.45/part.6
    fields: Dict[str, MetaValue] = field(default_factory=dict)
    item_term: str = "п."  # как называть подпункты 1) в ссылке
    subitem_term: str = "пп."  # как называть подпункты а) под 1)
    hierarchical: bool = True


@dataclass(frozen=True)
class Chunk:
    text: str  # то, что индексируется: заголовок источника + контекст + тело
    body: str
    metadata: Dict[str, MetaValue]


def kind_label(source_kind: str) -> str:
    return KIND_LABELS.get(source_kind, source_kind)


def source_label(meta: Optional[Dict[str, object]]) -> str:
    """Подпись источника для ответа: узкая ссылка (ст., ч., п., позиция), иначе название источника."""
    meta = meta or {}
    return str(meta.get("citation") or meta.get("source_display") or meta.get("source") or "источник")


def _chars(lines: List[str]) -> int:
    return len("\n".join(lines)) if lines else 0


def _is_all_caps(line: str) -> bool:
    return any(c.isalpha() for c in line) and line == line.upper()


def _truncate_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
    return cut + "…"


def _split_numbered(
    lines: List[str], pattern: "re.Pattern[str]"
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Строки до первого пункта и список (номер, строки пункта). Строки без номера продолжают пункт."""
    lead: List[str] = []
    points: List[Tuple[str, List[str]]] = []
    for line in lines:
        match = pattern.match(line)
        if match:
            points.append((match.group(1), [line]))
        elif points:
            points[-1][1].append(line)
        else:
            lead.append(line)
    return lead, points


def _place_lead(
    lead: List[str], points: List[Tuple[str, List[str]]], merge: bool
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Вводные строки перед первым пунктом: присоединяются к нему (merge) или остаются отдельно."""
    if not lead or not points or not merge:
        return lead, points
    first_number, first_lines = points[0]
    return [], [(first_number, lead + first_lines)] + points[1:]


# ---------------------------------------------------------------- ГК РФ, 44-ФЗ, 223-ФЗ

_ARTICLE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$")
_SECTION_MARK = re.compile(r"^§\s*\d+")
_PART = re.compile(r"^(\d+(?:\.\d+)?)\.\s+\S")


@dataclass
class _Article:
    number: str
    title: str
    heading: str
    lines: List[str]
    section: str = ""


def _split_articles(lines: List[str]) -> Tuple[List[str], List[_Article]]:
    preamble: List[str] = []
    articles: List[_Article] = []
    section = ""
    for line in lines:
        if not line:
            continue
        if _SECTION_MARK.match(line):
            section = line
            continue
        match = _ARTICLE.match(line)
        if match:
            articles.append(_Article(match.group(1), match.group(2).strip(), line, [], section))
        elif articles:
            articles[-1].lines.append(line)
        else:
            preamble.append(line)
    return preamble, articles


def _article_label(article: _Article) -> str:
    return f"Статья {article.number}. {article.title}" if article.title else f"Статья {article.number}"


def _article_fields(article: _Article) -> Dict[str, MetaValue]:
    fields: Dict[str, MetaValue] = {"article": article.number}
    if article.section:
        fields["section"] = article.section
    return fields


def _require_articles(articles: List[_Article], what: str) -> None:
    if not articles:
        raise ChunkingError(f"{what}: не найдено ни одной строки «Статья N.»")


def parse_civil_code(lines: List[str], prefix: str, cfg: ChunkingConfig) -> List[Block]:
    """Статья остаётся одним блоком, если укладывается в soft_max; иначе — по нумерованным пунктам."""
    preamble, articles = _split_articles(lines)
    _require_articles(articles, "civil_code")
    sizes = cfg.sizes
    blocks: List[Block] = []
    if preamble:
        blocks.append(Block(preamble, "Вступительная часть", f"{prefix}, вступительная часть", "preamble"))
    for article in articles:
        label = _article_label(article)
        cite = f"{prefix}, ст. {article.number}"
        anchor = f"art.{article.number}"
        fields = _article_fields(article)
        if not article.lines:
            # Статья-статус (например, утратила силу): единственный текст — сама строка заголовка.
            blocks.append(
                Block([article.heading], f"Статья {article.number}", cite, anchor, fields, "пп.")
            )
            continue
        lead, points = _split_numbered(article.lines, _PART)
        if _chars(article.lines) <= sizes.soft_max or not points:
            blocks.append(Block(article.lines, label, cite, anchor, fields, "пп."))
            continue
        lead, points = _place_lead(lead, points, merge=_chars(lead) < sizes.min_len)
        if lead:
            blocks.append(Block(lead, label, cite, anchor, dict(fields), "пп."))
        for number, point_lines in points:
            blocks.append(
                Block(
                    point_lines,
                    f"{label} — п. {number}",
                    f"{cite}, п. {number}",
                    f"{anchor}/point.{number}",
                    dict(fields, point=number),
                    "пп.",
                )
            )
    return blocks


def parse_federal_law(lines: List[str], prefix: str, cfg: ChunkingConfig) -> List[Block]:
    """44-ФЗ / 223-ФЗ: каждая нумерованная часть статьи — отдельный адресуемый блок."""
    preamble, articles = _split_articles(lines)
    _require_articles(articles, "federal_law")
    sizes = cfg.sizes
    blocks: List[Block] = []
    if preamble:
        blocks.append(Block(preamble, "Вступительная часть", f"{prefix}, вступительная часть", "preamble"))
    for article in articles:
        label = _article_label(article)
        cite = f"{prefix}, ст. {article.number}"
        anchor = f"art.{article.number}"
        fields = _article_fields(article)
        if not article.lines:
            blocks.append(Block([article.heading], f"Статья {article.number}", cite, anchor, fields))
            continue
        lead, parts = _split_numbered(article.lines, _PART)
        if not parts:
            blocks.append(Block(article.lines, label, cite, anchor, fields))
            continue
        # «(введена Федеральным законом ...)» и подобные короткие строки уходят в первую часть.
        lead, parts = _place_lead(lead, parts, merge=_chars(lead) < sizes.min_len)
        if lead:
            blocks.append(Block(lead, label, cite, anchor, dict(fields)))
        for number, part_lines in parts:
            blocks.append(
                Block(
                    part_lines,
                    f"{label} — ч. {number}",
                    f"{cite}, ч. {number}",
                    f"{anchor}/part.{number}",
                    dict(fields, part=number),
                    item_term="п.",
                    subitem_term="пп.",
                )
            )
    return blocks


# ---------------------------------------------------------------- постановления Правительства РФ

_STAMP = re.compile(r"^(?:утвержден[оаы]?)$", re.IGNORECASE)
_STAMP_END = re.compile(r"\bN\s*\d+\s*$")
_APPENDIX = re.compile(r"^Приложение\s+N\s*(\d+)\s*$")
_ROMAN = re.compile(r"^(?:I{1,3}|IV|VI{0,3}|IX|X)\.\s+\S")
_PP_POINT = re.compile(r"^(\d+(?:\(\d+\))?)\.\s+\S")
_SIGNATURE = re.compile(r"^Уполномоченное лицо\b")

# (шаблон по окну заголовка в ВЕРХНЕМ регистре, slug, тип, подпись, краткая подпись для ссылки).
# Порядок важен: срабатывает первое совпадение. {subject} — «контракта» или «договора».
_PP_RULES: List[Tuple[str, str, str, str, str]] = [
    (r"^ПОЛОЖЕНИЕ\b", "regulation", "regulation", "Положение о независимых гарантиях", ""),
    (
        r"ПЕРЕЧЕНЬ ДОКУМЕНТОВ",
        "document_list",
        "rules",
        "Перечень документов, представляемых заказчиком гаранту",
        "перечень документов",
    ),
    (
        r"ДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ",
        "requirements",
        "rules",
        "Дополнительные требования к независимой гарантии",
        "дополнительные требования",
    ),
    (
        r"ПРАВИЛА ФОРМИРОВАНИЯ И ВЕДЕНИЯ ЗАКРЫТОГО РЕЕСТРА",
        "closed_registry_rules",
        "rules",
        "Правила формирования и ведения закрытого реестра независимых гарантий",
        "правила закрытого реестра",
    ),
    (
        r"ПРАВИЛА ВЕДЕНИЯ И РАЗМЕЩЕНИЯ",
        "registry_rules",
        "rules",
        "Правила ведения и размещения реестра независимых гарантий",
        "правила ведения реестра",
    ),
    (
        r"ТИПОВАЯ ФОРМА.*ЗАЯВК",
        "bid_guarantee_form",
        "form",
        "Типовая форма независимой гарантии (обеспечение заявки)",
        "типовая форма обеспечения заявки",
    ),
    (
        r"ТИПОВАЯ ФОРМА.*ИСПОЛНЕНИ",
        "contract_guarantee_form",
        "form",
        "Типовая форма независимой гарантии (обеспечение исполнения {subject})",
        "типовая форма обеспечения исполнения {subject}",
    ),
    (
        r"ФОРМА ТРЕБОВАНИЯ",
        "payment_demand_form",
        "form",
        "Форма требования об уплате денежной суммы по независимой гарантии",
        "форма требования об уплате",
    ),
    (
        r"ТРЕБОВАНИЕ.*ЗАЯВК",
        "bid_demand_form",
        "form",
        "Форма требования об уплате (обеспечение заявки)",
        "форма требования (обеспечение заявки)",
    ),
    (
        r"ТРЕБОВАНИЕ.*ИСПОЛНЕНИ",
        "contract_demand_form",
        "form",
        "Форма требования об уплате (обеспечение исполнения {subject})",
        "форма требования (обеспечение исполнения {subject})",
    ),
]
_WINDOW_LINES = 7


@dataclass
class _Segment:
    slug: str
    kind: str  # resolution | regulation | rules | form
    label: str
    short: str
    appendix: Optional[int]
    lines: List[str]  # без реквизитов утверждения; пустые строки сохранены


def _classify_segment(content: List[str], appendix: Optional[int]) -> _Segment:
    head = " ".join([line for line in content if line][:_WINDOW_LINES]).upper()
    slug, kind, label, short = "annex", "rules", "Приложение", "приложение"
    for pattern, rule_slug, rule_kind, rule_label, rule_short in _PP_RULES:
        if re.search(pattern, head):
            subject = "договора" if "ИСПОЛНЕНИЯ ДОГОВОРА" in head else "контракта"
            slug, kind = rule_slug, rule_kind
            label, short = rule_label.format(subject=subject), rule_short.format(subject=subject)
            break
    else:
        first = next((line for line in content if line), "")
        label = _truncate_words(first, 120) or label
        short = label.lower()
        if appendix is None:
            slug = "annex"
    if appendix is not None:
        label = f"Приложение N {appendix}. {label}"
        short = f"прил. {appendix}, {short}" if short else f"прил. {appendix}"
    return _Segment(slug, kind, label, short, appendix, content)


def _split_resolution_segments(lines: List[str]) -> List[_Segment]:
    """
    Постановление = преамбула (текст постановления) + утверждаемые акты. Начало акта — строка
    «УТВЕРЖДЕНЫ/Утверждено» с реквизитами постановления (1005, 1397) или «Приложение N k» (1397).
    Строки реквизитов утверждения — аппарат документа: смысл несёт заголовок акта, он остаётся в тексте.
    """
    n = len(lines)
    starts: List[Tuple[int, int, Optional[int]]] = []
    for i, line in enumerate(lines):
        appendix = _APPENDIX.match(line)
        if appendix:
            j = i + 1
            while j < n and lines[j]:
                j += 1
            starts.append((i, j, int(appendix.group(1))))
        elif _STAMP.match(line):
            j = i + 1
            limit = min(n, i + 5)
            while j < limit and not _STAMP_END.search(lines[j]):
                j += 1
            starts.append((i, min(j + 1, n), None))
    segments: List[_Segment] = []
    first_start = starts[0][0] if starts else n
    preamble = lines[:first_start]
    if any(preamble):
        segments.append(_Segment("resolution", "resolution", "Постановление", "постановление", None, preamble))
    for index, (_, content_start, appendix) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else n
        content = lines[content_start:end]
        if any(content):
            segments.append(_classify_segment(content, appendix))
    return segments


def _point_blocks(
    lines: List[str],
    *,
    label: str,
    cite: str,
    anchor: str,
    fields: Dict[str, MetaValue],
    merge_preface: bool,
    preface_label: str,
) -> List[Block]:
    lead, points = _split_numbered(lines, _PP_POINT)
    if not points:
        return [Block(lead, label, cite, anchor, dict(fields), "пп.")] if lead else []
    lead, points = _place_lead(lead, points, merge_preface)
    blocks: List[Block] = []
    if lead:
        blocks.append(
            Block(lead, f"{label} — {preface_label}", f"{cite}, {preface_label}", f"{anchor}/head", dict(fields), "пп.")
        )
    for number, point_lines in points:
        blocks.append(
            Block(
                point_lines,
                f"{label} — п. {number}",
                f"{cite}, п. {number}",
                f"{anchor}/p.{number}",
                dict(fields, point=number),
                "пп.",
            )
        )
    return blocks


def _regulation_sections(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """Разделы «I. ...», «II. ...»: заголовок может занимать несколько строк до пустой строки."""
    sections: List[Tuple[str, List[str]]] = []
    heading = ""
    current: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if _ROMAN.match(line) and (i == 0 or not lines[i - 1]):
            if heading or current:
                sections.append((heading, current))
            parts = []
            while i < n and lines[i]:
                parts.append(lines[i])
                i += 1
            heading, current = " ".join(parts), []
            continue
        if line:
            current.append(line)
        i += 1
    if heading or current:
        sections.append((heading, current))
    return sections


def _segment_blocks(segment: _Segment, prefix: str) -> List[Block]:
    cite_base = f"{prefix}, {segment.short}" if segment.short else prefix
    anchor_base = segment.slug + (f"/app.{segment.appendix}" if segment.appendix is not None else "")
    fields: Dict[str, MetaValue] = {}
    if segment.appendix is not None:
        fields["appendix"] = segment.appendix

    if segment.kind == "regulation":
        sections = _regulation_sections(segment.lines)
        blocks: List[Block] = []
        carry: List[str] = []  # заголовок акта («ПОЛОЖЕНИЕ О ...») до первого раздела
        for heading, section_lines in sections:
            if not heading:
                carry = section_lines
                continue
            blocks.extend(
                _point_blocks(
                    carry + section_lines,
                    label=f"{segment.label}, раздел {heading}",
                    cite=cite_base,
                    anchor=anchor_base,
                    fields=dict(fields, section=heading),
                    merge_preface=True,
                    preface_label="начало раздела",
                )
            )
            carry = []
        if carry:
            blocks.extend(
                _point_blocks(
                    carry, label=segment.label, cite=cite_base, anchor=anchor_base, fields=fields,
                    merge_preface=True, preface_label="начало",
                )
            )
        return blocks

    content = [line for line in segment.lines if line]
    if segment.kind == "form":
        form_fields = dict(fields, form=segment.slug, section=segment.label)
        tail_at = next((i for i, line in enumerate(content) if _SIGNATURE.match(line)), None)
        body, tail = (content, []) if tail_at is None else (content[:tail_at], content[tail_at:])
        has_points = any(_PP_POINT.match(line) for line in body)
        blocks = _point_blocks(
            body,
            label=segment.label,
            cite=cite_base,
            anchor=anchor_base,
            fields=form_fields,
            merge_preface=False,
            preface_label="начало формы (реквизиты)" if has_points else "текст формы",
        )
        if tail:
            blocks.append(
                Block(
                    tail,
                    f"{segment.label} — подпись и сноски",
                    f"{cite_base}, подпись и сноски",
                    f"{anchor_base}/tail",
                    dict(form_fields),
                    "пп.",
                )
            )
        return blocks

    return _point_blocks(
        content,
        label=segment.label,
        cite=cite_base,
        anchor=anchor_base,
        fields=dict(fields, section=segment.label),
        merge_preface=True,
        preface_label="начало",
    )


def parse_government_resolution(lines: List[str], prefix: str, cfg: ChunkingConfig) -> List[Block]:
    segments = _split_resolution_segments(lines)
    if not segments:
        raise ChunkingError("government_resolution: пустой источник")
    blocks: List[Block] = []
    for segment in segments:
        blocks.extend(_segment_blocks(segment, prefix))
    if not any("point" in block.fields for block in blocks):
        raise ChunkingError("government_resolution: не найдено ни одного нумерованного пункта")
    return blocks


# ---------------------------------------------------------------- обзоры практики ВС РФ

_POSITION_HEADING = re.compile(r"^Позиция\s+(\d+)\s*$")
_POSITION_INLINE = re.compile(r"^(\d{1,3})\.\s*(?=[А-ЯЁ«\"])(.+)$")


def _thesis(text: str) -> str:
    sentences = split_sentences(text)
    return _truncate_words(sentences[0] if sentences else text, _THESIS_MAX)


def parse_court_review(lines: List[str], prefix: str, cfg: ChunkingConfig) -> List[Block]:
    """Позиция N — отдельная единица; её номер и тезис повторяются в каждом фрагменте позиции."""
    content = [line for line in lines if line]
    starts: List[Tuple[int, int, str]] = []  # (индекс строки, номер, тезис для строки «Позиция N»)
    for i, line in enumerate(content):
        match = _POSITION_HEADING.match(line)
        if match:
            starts.append((i, int(match.group(1)), ""))
    heading_style = bool(starts)
    if not heading_style:
        last = 0
        for i, line in enumerate(content):
            match = _POSITION_INLINE.match(line)
            if match and int(match.group(1)) > last:
                last = int(match.group(1))
                starts.append((i, last, match.group(2)))
    if not starts:
        raise ChunkingError("court_review: не найдено ни одной нумерованной позиции")

    blocks: List[Block] = []
    preamble = content[: starts[0][0]]
    section = ""
    if preamble and _is_all_caps(preamble[-1]):
        section = preamble.pop()
    if preamble:
        blocks.append(
            Block(
                preamble,
                "Вводная часть обзора",
                f"{prefix}, вводная часть",
                "preamble",
                {"section": "вводная часть"},
                hierarchical=False,
            )
        )
    for index, (start, number, inline) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(content)
        body = content[start + 1 : end] if heading_style else content[start:end]
        if not body:
            raise ChunkingError(f"court_review: позиция {number} без текста")
        thesis = _thesis(body[0] if heading_style else inline)
        fields: Dict[str, MetaValue] = {"position": number}
        if section:
            fields["section"] = section
        blocks.append(
            Block(
                body,
                f"Позиция {number}. {thesis}",
                f"{prefix}, позиция {number}",
                f"pos.{number}",
                fields,
                hierarchical=False,
            )
        )
    return blocks


# ---------------------------------------------------------------- произвольный текст

def parse_generic(lines: List[str], prefix: str, cfg: ChunkingConfig) -> List[Block]:
    """Запасной вариант: абзацы (разделены пустой строкой) -> предложения, с перекрытием в одно предложение."""
    sizes = cfg.sizes
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines + [""]:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    atoms: List[str] = []
    for paragraph in paragraphs:
        atoms.extend(split_text(paragraph, sizes.target, sizes.min_len, sizes.soft_max))
    groups = pack_pieces(atoms, sizes.target, "\n\n", sizes.min_len, sizes.soft_max)
    blocks: List[Block] = []
    previous = ""
    for index, group in enumerate(groups):
        text = group
        if cfg.chunk_overlap > 0 and previous:
            tail = split_sentences(previous)[-1]
            if len(tail) <= cfg.chunk_overlap and tail != group:
                text = f"{tail}\n\n{group}"
        previous = group
        heading = _truncate_words(group.split("\n", 1)[0], 120)
        blocks.append(
            Block(text.split("\n"), heading, prefix, f"chunk.{index}", hierarchical=False)
        )
    return blocks


_PARSERS: Dict[str, Callable[[List[str], str, ChunkingConfig], List[Block]]] = {
    "generic": parse_generic,
    "civil_code": parse_civil_code,
    "federal_law": parse_federal_law,
    "government_resolution": parse_government_resolution,
    "court_review": parse_court_review,
}


def known_chunkers() -> List[str]:
    return sorted(_PARSERS)


# ---------------------------------------------------------------- сборка фрагментов

def _suffix(term: str, value: str) -> str:
    return f", {term} {value}" if value else ""


def build_chunks(
    text: str,
    *,
    source: str,
    source_display: str,
    source_kind: str,
    doc_type: str,
    chunker: str = "generic",
    citation: Optional[str] = None,
    config: Optional[ChunkingConfig] = None,
) -> List[Chunk]:
    """Фрагменты источника: (текст для индекса, метаданные — только скаляры str/int, совместимые с Chroma)."""
    cfg = config or ChunkingConfig()
    parser = _PARSERS.get(chunker)
    if parser is None:
        raise ChunkingError(f"{source}: неизвестная стратегия нарезки «{chunker}» (доступны: {known_chunkers()})")
    prefix = citation or source_display
    try:
        blocks = parser(normalize_lines(text), prefix, cfg)
    except ChunkingError as exc:
        raise ChunkingError(f"{source}: {exc}") from exc
    if not blocks:
        raise ChunkingError(f"{source}: после нарезки не осталось ни одного фрагмента")

    kind = kind_label(source_kind)
    out: List[Chunk] = []
    for block in blocks:
        pieces = split_block(block.lines, cfg.sizes, block.hierarchical)
        for index, piece in enumerate(pieces):
            body = piece.text.strip()
            if not body:
                raise ChunkingError(f"{source}: пустой фрагмент в {block.anchor}")
            suffix = _suffix(block.item_term, piece.item) + _suffix(block.subitem_term, piece.subitem)
            label = block.label + suffix
            anchor = block.anchor
            if piece.item:
                anchor += f"/item.{piece.item}"
            if piece.subitem:
                anchor += f"/sub.{piece.subitem}"
            meta: Dict[str, MetaValue] = {
                "source": source,
                "source_display": source_display,
                "source_kind": source_kind,
                "doc_type": doc_type,
                "section_heading": label,
                "citation": block.citation + suffix,
                "anchor": anchor,
                "chunk_index": index,
                "chunk_count": len(pieces),
            }
            meta.update(block.fields)
            if piece.item:
                meta["item"] = piece.item
            if piece.subitem:
                meta["subitem"] = piece.subitem
            doc_text = f"[Источник: {source_display} | {kind}]\n[Фрагмент: {label}]\n\n{body}"
            out.append(Chunk(doc_text, body, meta))
    return out


# ---------------------------------------------------------------- статистика (python chunking.py)

def corpus_stats(chunks_by_source: Dict[str, List[Chunk]], cfg: ChunkingConfig) -> str:
    rows = [(source, c) for source, chunks in chunks_by_source.items() for c in chunks]
    sizes = sorted(len(c.text) for _, c in rows)
    bodies = sorted(len(c.body) for _, c in rows)

    def pct(values: List[int], p: float) -> int:
        return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]

    soft_max = cfg.sizes.soft_max
    lines = [
        f"chunk_size(target)={cfg.chunk_size} soft_max={soft_max} min_chunk_len={cfg.min_chunk_len}",
        f"total chunks: {len(rows)}",
        "text size (as indexed) min/median/p95/max: "
        f"{sizes[0]} / {int(statistics.median(sizes))} / {pct(sizes, 0.95)} / {sizes[-1]}",
        "body size min/median/p95/max: "
        f"{bodies[0]} / {int(statistics.median(bodies))} / {pct(bodies, 0.95)} / {bodies[-1]}",
        f"bodies > target ({cfg.chunk_size}): {sum(1 for b in bodies if b > cfg.chunk_size)}",
        f"bodies > soft_max ({soft_max}): {sum(1 for b in bodies if b > soft_max)}",
        "chunks per source:",
    ]
    for source, chunks in chunks_by_source.items():
        lines.append(f"  {source}: {len(chunks)} (max body {max(len(c.body) for c in chunks)})")
    for source, c in rows:
        if len(c.body) > soft_max:
            lines.append(f"  OVERSIZED {source} {c.metadata['anchor']} body={len(c.body)}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        from .corpus_config import default_corpus_entries, index_settings
    except ImportError:
        from corpus_config import default_corpus_entries, index_settings

    settings = index_settings()
    config = ChunkingConfig(settings["chunk_size"], settings["chunk_overlap"], settings["min_chunk_len"])
    by_source: Dict[str, List[Chunk]] = {}
    for entry in default_corpus_entries():
        by_source[entry["source"]] = build_chunks(
            Path(entry["path"]).read_text(encoding="utf-8"),
            source=entry["source"],
            source_display=entry["source_display"],
            source_kind=entry["source_kind"],
            doc_type=entry["doc_type"],
            chunker=entry.get("chunker", "generic"),
            citation=entry.get("citation"),
            config=config,
        )
    print(corpus_stats(by_source, config))
