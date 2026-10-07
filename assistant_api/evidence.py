"""
Шлюз достаточности данных: если лучший найденный фрагмент слишком далёк от вопроса,
чат-модель не вызывается, а пользователь получает прямой ответ «в корпусе недостаточно данных».

Семантика расстояний Chroma проверена на реальной коллекции (все значения «чем меньше, тем ближе»):
- cosine: d = 1 - cos(a, b)       (диапазон 0..2);
- l2:     d = |a - b|^2           (у единичных векторов = 2 - 2*cos);
- ip:     d = 1 - (a . b)         (у единичных векторов = 1 - cos).
Эмбеддинги OpenAI нормированы (длина ≈ 1), поэтому любую из метрик можно привести к косинусному
расстоянию, а порог хранить в одной шкале. Метрика читается из самой коллекции, а не предполагается.

Порог RAG_MAX_DISTANCE — максимальное косинусное расстояние лучшего фрагмента, при котором ответ ещё
строится по корпусу. Он необязателен и по умолчанию не задан: пока он не задан, отказ по расстоянию
отключён, и чат-модель получает любую непустую выдачу. Шкала расстояний зависит от модели эмбеддингов
и корпуса, поэтому значение нужно подбирать на реальных запросах к развёрнутому индексу; встроенного
«безопасного» значения нет.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

ENV_MAX_DISTANCE = "RAG_MAX_DISTANCE"

NO_EVIDENCE_ANSWER = (
    "В текущем корпусе недостаточно данных для надежного ответа на этот вопрос. "
    "Корпус ограничен выбранными источниками (ГК РФ, ст. 368–379; 44-ФЗ, ст. 45; 223-ФЗ, ст. 3.4; "
    "постановления Правительства РФ № 1005 и № 1397; обзоры практики ВС РФ), поэтому это не означает, "
    "что ответа нет в законодательстве в целом. Переформулируйте вопрос по теме независимых гарантий "
    "или обратитесь к первоисточникам."
)


def max_distance_from_env() -> Optional[float]:
    """
    Порог из RAG_MAX_DISTANCE. Не задан или пуст — None (отказ по расстоянию отключён);
    неверное значение — явная ошибка, а не молчаливая подстановка другого порога.
    """
    raw = (os.getenv(ENV_MAX_DISTANCE) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{ENV_MAX_DISTANCE}={raw!r}: ожидается число (косинусное расстояние, 0..2)") from None
    if math.isnan(value) or not 0 < value <= 2:
        raise ValueError(f"{ENV_MAX_DISTANCE}={raw!r}: допустимы значения в диапазоне (0, 2]")
    return value


def collection_metric(collection: Any) -> Optional[str]:
    """Метрика расстояния коллекции Chroma: cosine | l2 | ip; None, если определить не удалось."""
    try:
        config = getattr(collection, "configuration", None)
        hnsw = config.get("hnsw") if config is not None else None
        space = hnsw.get("space") if hnsw else None
        if not space:
            metadata = getattr(collection, "metadata", None) or {}
            space = metadata.get("hnsw:space")
    except Exception:
        return None
    return space if space in ("cosine", "l2", "ip") else None


def to_cosine_distance(distance: Optional[float], metric: Optional[str]) -> Optional[float]:
    """Приведение расстояния Chroma к косинусному (для единичных векторов)."""
    if distance is None or metric is None:
        return None
    if metric == "l2":
        return distance / 2.0
    return distance  # cosine и ip (1 - ip) для единичных векторов совпадают с 1 - cos


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    best_distance: Optional[float]  # косинусное расстояние лучшего фрагмента (None, если неизвестно)
    metric: Optional[str]
    max_distance: Optional[float]  # None — порог не задан
    reason: str  # ok | too_far | no_results | no_distance_info | unknown_metric | threshold_disabled

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "best_distance": self.best_distance,
            "metric": self.metric,
            "max_distance": self.max_distance,
            "reason": self.reason,
        }


def assess_evidence(
    docs: List[Dict[str, Any]], metric: Optional[str], max_distance: Optional[float]
) -> EvidenceAssessment:
    """
    Достаточно ли данных: наименьшее косинусное расстояние среди найденных кандидатов
    не должно превышать max_distance. Если порог не задан (None) или расстояние определить нельзя
    (неизвестная метрика, нет значений) — не блокируем ответ: шлюз не должен отказывать без оснований.
    Пустая выдача блокируется всегда: это не вопрос порога, контекста для ответа просто нет.
    """
    if not docs:
        return EvidenceAssessment(False, None, metric, max_distance, "no_results")
    if max_distance is None:
        return EvidenceAssessment(True, None, metric, None, "threshold_disabled")
    if metric is None:
        return EvidenceAssessment(True, None, None, max_distance, "unknown_metric")
    distances = [
        d for d in (to_cosine_distance(doc.get("distance"), metric) for doc in docs) if d is not None
    ]
    if not distances:
        return EvidenceAssessment(True, None, metric, max_distance, "no_distance_info")
    best = min(distances)
    if best > max_distance:
        return EvidenceAssessment(False, best, metric, max_distance, "too_far")
    return EvidenceAssessment(True, best, metric, max_distance, "ok")
