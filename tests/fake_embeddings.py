"""
Детерминированные офлайн-эмбеддинги для тестов: сеть и OpenAI не используются.

Это хэшированный TF-IDF по «основам» слов (первые 5 букв), нормированный до единичной длины.
Он лексический: ловит совпадение терминов вопроса и фрагмента, но не смысла. Поэтому на нём можно
проверять нарезку, метаданные и привязку к норме (находится ли нужная статья/часть/позиция), но
нельзя судить о качестве настоящей модели эмбеддингов и калибровать по нему порог для неё.
"""

import math
import re
import zlib
from collections import Counter
from types import SimpleNamespace
from typing import Dict, Iterable, List

DIM = 4096  # последний разряд зарезервирован под текст без значимых слов

_WORD = re.compile(r"[а-яёa-z0-9]+")
_STOP = frozenset(
    "для при или что как это его она они все так также если был была были быть может могут какие "
    "какой какая каких кто где когда ли не на по из от до за со об то же бы нет да ему ней них "
    "там тут здесь только этом этой этих такой такие которые который которая которое которых "
    "должен должна должно должны вправе ввести".split()
)


def stems(text: str) -> List[str]:
    words = _WORD.findall(text.lower().replace("ё", "е"))
    return [w[:5] for w in words if len(w) > 2 and w not in _STOP]


class HashingTfidfEmbedder:
    def __init__(self) -> None:
        self.idf: Dict[str, float] = {}
        self.default_idf = 1.0

    def fit(self, texts: Iterable[str]) -> "HashingTfidfEmbedder":
        documents = [set(stems(t)) for t in texts]
        frequency: Counter = Counter()
        for doc in documents:
            frequency.update(doc)
        total = len(documents)
        self.idf = {s: math.log((1 + total) / (1 + n)) + 1.0 for s, n in frequency.items()}
        self.default_idf = math.log(1 + total) + 1.0  # слово, которого нет в корпусе
        return self

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * DIM
        for stem, count in Counter(stems(text)).items():
            bucket = zlib.crc32(stem.encode("utf-8")) % (DIM - 1)
            vector[bucket] += (1.0 + math.log(count)) * self.idf.get(stem, self.default_idf)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[DIM - 1] = 1.0
            return vector
        return [v / norm for v in vector]


class FakeEmbeddingsClient:
    """Подмена openai.OpenAI для VectorStore: только embeddings.create."""

    def __init__(self, embedder: HashingTfidfEmbedder) -> None:
        self.embedder = embedder
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, input, model=None, **kwargs):  # имя input — как в клиенте OpenAI
        self.calls += 1
        texts = [input] if isinstance(input, str) else list(input)
        data = [SimpleNamespace(index=i, embedding=self.embedder.embed(t)) for i, t in enumerate(texts)]
        return SimpleNamespace(data=data)
