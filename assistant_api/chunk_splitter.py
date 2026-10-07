"""
Детерминированные примитивы нарезки юридического текста.

Без внешних зависимостей (в том числе без pysbd): результат не зависит от версий библиотек.

Размер задаётся целевым значением (target), а не жёсткой отсечкой:
- юридическая единица (часть, пункт, позиция), которая укладывается в soft_max, остаётся целой;
- более крупная делится по подпунктам (1), 2), а), б)), затем по абзацам, затем по предложениям,
  затем по ; и , и лишь в крайнем случае по пробелам. Слова не разрываются, текст не отбрасывается;
- у каждого следующего фрагмента повторяется «вводная» строка пункта (оканчивается на «:», короткая),
  чтобы подпункты не теряли смысл вне полного текста.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Единица, не длиннее target * SOFT_MAX_FACTOR, не делится: целостность нормы важнее ровного размера.
SOFT_MAX_FACTOR = 1.5

# Доля target, которую может занимать повторяемый «контекст» (вводная строка пункта) в каждом фрагменте.
_CONTEXT_SHARE = 0.5
# Короткая первая строка узла повторяется в продолжении (например, «а) условия о следующих правах:»).
_HEAD_SHARE = 0.3

_SPACES = re.compile(r"[ \t    ]+")
_NUM_ITEM = re.compile(r"^(\d+)\)\s")
_LET_ITEM = re.compile(r"^([а-яё])\)\s")
_SENTENCE_END = re.compile(r"[.!?…]+[\"»”)\]]*(?=\s)")
_CLAUSE_END = {";": re.compile(r";(?=\s)"), ",": re.compile(r",(?=\s)")}
_LAST_TOKEN = re.compile(r"(\S+)$")
_DOTTED_ABBREVIATION = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]")  # т.д, т.п, т.е, и.о

# Сокращения, после которых точка не заканчивает предложение (однобуквенные обрабатываются отдельно).
_ABBREVIATIONS = frozenset(
    {
        "ст", "пп", "гл", "разд", "абз", "подп", "ред", "изм", "гг", "см", "ср", "напр", "др",
        "пр", "прим", "им", "руб", "коп", "тыс", "млн", "млрд", "рис", "табл", "стр", "тел",
        "доп", "обл", "кв", "мин", "сек", "мп",
    }
)


@dataclass(frozen=True)
class ChunkSizes:
    target: int
    soft_max: int
    min_len: int


@dataclass
class Piece:
    """Результат деления блока: текст и узкая привязка (диапазоны подпунктов), если делили по ним."""

    text: str
    item: str = ""
    subitem: str = ""


def normalize_lines(text: str) -> List[str]:
    """
    Строки текста: LF, схлопнутые пробелы, без пробелов по краям. Подряд идущие пустые строки
    сводятся к одной (она нужна разборщикам для заголовков из нескольких строк).
    """
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    for line in raw:
        line = _SPACES.sub(" ", line).strip()
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def _chars(lines: List[str]) -> int:
    return len("\n".join(lines)) if lines else 0


def split_sentences(text: str) -> List[str]:
    """Деление на предложения с защитой сокращений («ст.», «п.», «т.д.»), инициалов и номеров пунктов."""
    sentences: List[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        following = text[end:].lstrip()[:1]
        if not following or not (following.isupper() or following.isdigit() or following in "«\"“(["):
            continue
        before = text[: match.start()]
        token_match = _LAST_TOKEN.search(before)
        if token_match and text[match.start()] == ".":
            token = token_match.group(1).lstrip("(«\"“[").lower()
            if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()) or _DOTTED_ABBREVIATION.fullmatch(token):
                continue
            # «14.1. Текст»: номер в начале строки не заканчивает предложение
            if re.fullmatch(r"[\d.]+", token) and before.strip() == token_match.group(1):
                continue
        sentences.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def pack_pieces(
    pieces: List[str],
    limit: int,
    separator: str,
    min_tail: int = 0,
    max_len: Optional[int] = None,
) -> List[str]:
    """
    Жадно собирает куски в группы не длиннее limit (кусок длиннее limit остаётся отдельной группой).
    Слишком короткий хвост (< min_tail) присоединяется к предыдущей группе, если итог не длиннее max_len.
    """
    groups: List[str] = []
    current: List[str] = []
    current_len = 0
    for piece in pieces:
        added = len(piece) + (len(separator) if current else 0)
        if current and current_len + added > limit:
            groups.append(separator.join(current))
            current, current_len = [piece], len(piece)
        else:
            current.append(piece)
            current_len += added
    if current:
        groups.append(separator.join(current))
    if (
        len(groups) >= 2
        and len(groups[-1]) < min_tail
        and max_len is not None
        and len(groups[-2]) + len(separator) + len(groups[-1]) <= max_len
    ):
        groups[-2] = groups[-2] + separator + groups[-1]
        groups.pop()
    return groups


def _split_clauses(text: str, mark: str) -> List[str]:
    parts: List[str] = []
    start = 0
    for match in _CLAUSE_END[mark].finditer(text):
        parts.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_text(text: str, limit: int, min_tail: int = 0, max_len: Optional[int] = None) -> List[str]:
    """
    Делит один абзац на куски не длиннее limit: по предложениям, затем по «;», по «,»,
    и только если иначе нельзя (одна «фраза» длиннее limit) — по пробелам. Слово никогда не режется.
    """
    if len(text) <= limit:
        return [text]
    sentences = split_sentences(text)
    if len(sentences) > 1:
        pieces: List[str] = []
        for sentence in sentences:
            pieces.extend(split_text(sentence, limit, 0, None) if len(sentence) > limit else [sentence])
        return pack_pieces(pieces, limit, " ", min_tail, max_len)
    for mark in (";", ","):
        clauses = _split_clauses(text, mark)
        if len(clauses) > 1:
            pieces = []
            for clause in clauses:
                pieces.extend(split_text(clause, limit, 0, None) if len(clause) > limit else [clause])
            return pack_pieces(pieces, limit, " ", min_tail, max_len)
    return pack_pieces(text.split(" "), limit, " ", min_tail, max_len)


@dataclass
class _Node:
    kind: str  # "root" | "num" (1)) | "let" (а))
    marker: str
    lines: List[str] = field(default_factory=list)
    children: List["_Node"] = field(default_factory=list)

    def all_lines(self) -> List[str]:
        out = list(self.lines)
        for child in self.children:
            out.extend(child.all_lines())
        return out

    def size(self) -> int:
        return _chars(self.all_lines())


def _parse_items(lines: List[str]) -> _Node:
    """
    Дерево подпунктов: «1)» — дети корня, «а)» — дети ближайшего «1)» (или корня, если «1)» не было).
    Строки без маркера продолжают предыдущий узел.
    """
    root = _Node("root", "")
    num: Optional[_Node] = None
    last = root
    for line in lines:
        match = _NUM_ITEM.match(line)
        if match:
            node = _Node("num", match.group(1), [line])
            root.children.append(node)
            num = node
            last = node
            continue
        match = _LET_ITEM.match(line)
        if match:
            node = _Node("let", match.group(1), [line])
            (num or root).children.append(node)
            last = node
            continue
        last.lines.append(line)
    return root


def _range_label(nodes: List[_Node]) -> str:
    first, last = nodes[0].marker, nodes[-1].marker
    return first if first == last else f"{first}–{last}"


def _join(lines: List[str]) -> str:
    return "\n".join(lines)


def _is_all_caps(line: str) -> bool:
    return any(c.isalpha() for c in line) and line == line.upper()


def _split_plain(lines: List[str], prefix: List[str], item: str, subitem: str, sizes: ChunkSizes) -> List[Piece]:
    """
    Узел без подпунктов: упаковка по абзацам, затем по предложениям.
    - Заглавные строки в начале (название акта) приклеиваются к первому содержательному фрагменту,
      чтобы не получился фрагмент из одного заголовка.
    - Первая строка-«вводная» (оканчивается на «:») повторяется в продолжениях.
    """
    titles: List[str] = []
    start = 0
    while start < len(lines) - 1 and _is_all_caps(lines[start]):
        titles.append(lines[start])
        start += 1
    body = lines[start:]
    head: List[str] = []
    if not titles and len(body) > 1 and len(body[0]) <= sizes.target * _HEAD_SHARE and body[0].endswith(":"):
        head, body = body[:1], body[1:]
    fixed = prefix + head
    room = max(sizes.target - _chars(fixed) - 1, sizes.target // 2)
    atoms: List[str] = []
    for index, line in enumerate(body):
        limit = room
        if index == 0 and titles:
            limit = max(room - _chars(titles) - 1, room // 2)
        atoms.extend(split_text(line, limit, sizes.min_len, sizes.soft_max) if len(line) > limit else [line])
    if titles:
        atoms[0] = _join(titles + [atoms[0]])
    groups = pack_pieces(atoms, room, "\n", sizes.min_len, max(sizes.soft_max - _chars(fixed) - 1, room))
    return [Piece(_join(fixed + [group]), item, subitem) for group in groups]


def _split_node(node: _Node, prefix: List[str], item: str, subitem: str, sizes: ChunkSizes) -> List[Piece]:
    own = node.lines
    if not node.children:
        return _split_plain(own, prefix, item, subitem, sizes)

    pieces: List[Piece] = []
    ctx_limit = sizes.target * _CONTEXT_SHARE
    lead_separate = _chars(prefix + own) > sizes.target
    if lead_separate:
        # Вводная часть сама по себе крупная: выносим её отдельными фрагментами и не повторяем.
        pieces.extend(_split_plain(own, prefix, item, subitem, sizes))
        first_lines = repeat_lines = prefix if _chars(prefix) <= ctx_limit else []
        lead_used = True
    else:
        first_lines = prefix + own
        repeat_lines = first_lines if _chars(first_lines) <= ctx_limit else (prefix if _chars(prefix) <= ctx_limit else [])
        lead_used = False

    def labels(group: List[_Node]) -> Tuple[str, str]:
        rng = _range_label(group)
        if group[0].kind == "let" and node.kind == "num":
            return item, rng
        return rng, ""

    current: List[_Node] = []

    def head_lines() -> List[str]:
        return repeat_lines if lead_used else first_lines

    def flush() -> None:
        nonlocal current, lead_used
        if not current:
            return
        group_item, group_subitem = labels(current)
        lines = head_lines() + [ln for child in current for ln in child.all_lines()]
        pieces.append(Piece(_join(lines), group_item, group_subitem))
        lead_used = True
        current = []

    for child in node.children:
        child_len = child.size() + 1
        if current and _chars(head_lines()) + sum(c.size() + 1 for c in current) + child_len > sizes.target:
            flush()
        if not current and _chars(head_lines()) + child_len > sizes.soft_max:
            child_item, child_subitem = (
                (item, child.marker) if child.kind == "let" and node.kind == "num" else (child.marker, "")
            )
            pieces.extend(_split_node(child, head_lines(), child_item, child_subitem, sizes))
            lead_used = True
            continue
        current.append(child)
    flush()
    return pieces


def split_block(lines: List[str], sizes: ChunkSizes, hierarchical: bool = True) -> List[Piece]:
    """
    Делит блок (одну юридическую единицу) на фрагменты. Блок в пределах soft_max остаётся целым.
    hierarchical=False отключает деление по подпунктам (например, для позиций обзора практики).
    """
    text = _join(lines)
    if len(text) <= sizes.soft_max:
        return [Piece(text)]
    if not hierarchical:
        return _split_plain(lines, [], "", "", sizes)
    return _split_node(_parse_items(lines), [], "", "", sizes)
