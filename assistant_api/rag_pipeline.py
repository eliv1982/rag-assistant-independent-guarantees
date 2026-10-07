"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from .cache import RAGCache, cache_enabled_from_env
    from .chunking import source_label
    from .corpus_config import (
        collection_name_for,
        compute_corpus_id,
        default_corpus_entries,
        single_file_entry,
    )
    from .evidence import (
        NO_EVIDENCE_ANSWER,
        EvidenceAssessment,
        assess_evidence,
        collection_metric,
        max_distance_from_env,
    )
    from .openai_client import api_key_configured, get_openai_client
    from .retrieval_utils import deduplicate_context_docs
    from .vector_store import VectorStore
except ImportError:
    from cache import RAGCache, cache_enabled_from_env
    from chunking import source_label
    from corpus_config import (
        collection_name_for,
        compute_corpus_id,
        default_corpus_entries,
        single_file_entry,
    )
    from evidence import (
        NO_EVIDENCE_ANSWER,
        EvidenceAssessment,
        assess_evidence,
        collection_metric,
        max_distance_from_env,
    )
    from openai_client import api_key_configured, get_openai_client
    from retrieval_utils import deduplicate_context_docs
    from vector_store import VectorStore

_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    load_dotenv(_env)
else:
    load_dotenv()


LEGAL_SYSTEM_PROMPT = (
    "Ты — ассистент по вопросам независимых гарантий (банковских и иных). "
    "Отвечай строго на основании переданных фрагментов базы знаний. "
    "Если вопрос просит перечень оснований, способов, случаев или условий — "
    "приводи те элементы, которые прямо названы во фрагментах; не дополняй перечень по памяти "
    "и не называй его исчерпывающим, если фрагменты этого не подтверждают. "
    "Не выдавай юридических заключений и не подменяй консультацию юриста; "
    "формулируй осторожно, если контекст неполный."
)

PROMPT_INTRO = (
    "Ты помогаешь разбирать вопросы по независимым гарантиям с опорой на фрагменты ГК РФ, "
    "Федеральных законов № 44-ФЗ и № 223-ФЗ, постановлений Правительства РФ № 1005 и № 1397 "
    "и обзоров судебной практики ВС РФ."
)

PROMPT_INSTRUCTIONS = """- Отвечай только на основании найденного контекста. Если данных недостаточно, прямо укажи, чего не хватает (например, нет нужной статьи, пункта постановления или позиции суда).
- Если вопрос просит «способы», «основания», «случаи», «условия», «перечень» или похожий список — перечисли нумерованным списком те элементы, которые прямо названы в найденных фрагментах. Не добавляй элементы, которых во фрагментах нет.
- Не называй перечень полным (исчерпывающим), если фрагменты не подтверждают его полноту. Если перечень во фрагментах приведён не целиком или оборван, прямо скажи, что список может быть неполным и каких данных не хватает.
- Если во фрагменте есть статья или пункт с явным перечнем, передавай его элементы так, как они названы в тексте, а не общим пересказом одной фразой — но только в пределах найденного.
- Различай уровни регулирования: ГК РФ — общие нормы о независимой гарантии; 44-ФЗ и 223-ФЗ — специальные нормы о закупках; постановления Правительства РФ № 1005 и № 1397 — требования, реестры и типовые формы; обзоры практики ВС РФ — толкование и применение норм. Не смешивай уровни молча.
- Если в контексте есть нормы нескольких уровней, структурируй ответ блоками по источникам («По ГК РФ», «По 44-ФЗ / 223-ФЗ», «По постановлениям Правительства РФ», «По практике ВС РФ») и добавь блок «Коротко»; блоки, для которых нет фрагментов, пропускай.
- Для каждого существенного тезиса укажи источник — по тому, из какого фрагмента он взят, в формате из заголовка фрагмента (например, «44-ФЗ, ст. 45, ч. 6» или «Обзор ВС РФ от 05.06.2019, позиция 11»).
- Если во фрагменте указано, что норма утратила силу, не применяй её как действующую: сообщи, что она утратила силу, и приведи реквизиты из фрагмента.
- Не придумывай номера статей, пунктов, дел и цитат, которых нет во фрагментах. Не делай юридическую консультацию и не выдумывай нормы вне контекста.
- Ответ на русском языке; структурируй списком, если это улучшает ясность."""

# Версия промпта входит в ключ кеша: после правки промпта старые ответы не отдаются.
PROMPT_VERSION = hashlib.sha256(
    "\n".join((LEGAL_SYSTEM_PROMPT, PROMPT_INTRO, PROMPT_INSTRUCTIONS)).encode("utf-8")
).hexdigest()[:8]

# finish_reason, при которых ответ модели нельзя выдавать за полный.
INCOMPLETE_NOTICES = {
    "length": (
        "**Внимание:** ответ модели оборван из-за ограничения длины (RAG_MAX_TOKENS) и может быть неполным. "
        "Не считайте его исчерпывающим; задайте вопрос уже или по частям."
    ),
    "content_filter": (
        "**Внимание:** генерация ответа остановлена фильтром содержимого модели; ответ может быть неполным."
    ),
}


def _normalize_cached_context(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"text": item, "metadata": {}})
        elif isinstance(item, dict) and "text" in item:
            out.append(
                {
                    "text": item["text"],
                    "metadata": item.get("metadata") or {},
                    "id": item.get("id"),
                }
            )
    return out or None


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        cache_db_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        corpus_entries: Optional[List[Dict[str, Any]]] = None,
        data_file: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Пути и имя коллекции по умолчанию берутся из окружения и корпуса:
        RAG_CHROMA_PATH, RAG_CACHE_DB_PATH; коллекция называется guarantees_<corpus_id>,
        поэтому изменённый корпус индексируется заново, а не подмешивается к старому индексу.

        Постоянный кеш ответов хранит тексты вопросов и ответов, поэтому по умолчанию выключен
        (RAG_CACHE_ENABLED). Пока он выключен, self.cache = None: файл кеша не открывается,
        не читается и не пишется, а поиск и вызов модели работают как обычно.
        """
        if not api_key_configured():
            raise ValueError("OPENAI_API_KEY не установлен")

        self._base_dir = Path(__file__).resolve().parent
        self.model = model or os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "1500"))
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))
        self.max_distance = max_distance_from_env()
        self._distance_metric: Optional[str] = None
        self._metric_resolved = False

        self.openai_client = get_openai_client()

        if corpus_entries is None:
            if data_file:
                data_path = Path(data_file)
                if not data_path.is_absolute():
                    data_path = (self._base_dir / data_path).resolve()
                corpus_entries = [single_file_entry(data_path)]
            else:
                corpus_entries = default_corpus_entries()

        self.corpus_id = compute_corpus_id(corpus_entries, base_dir=self._base_dir)
        if collection_name is None:
            collection_name = collection_name_for(self.corpus_id)

        if persist_directory is None:
            persist_directory = os.getenv("RAG_CHROMA_PATH") or str(self._base_dir / "chroma_db")
        self.cache_enabled = cache_enabled_from_env()
        if self.cache_enabled and cache_db_path is None:
            cache_db_path = os.getenv("RAG_CACHE_DB_PATH") or str(self._base_dir / "api_rag_cache.db")

        print(f"Корпус: {len(corpus_entries)} источников, corpus_id={self.corpus_id}")
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        if self.vector_store.collection.count() == 0:
            print("Загрузка корпуса...")
            self.vector_store.load_corpus(corpus_entries, base_dir=self._base_dir)

        self.cache: Optional[RAGCache] = None
        if self.cache_enabled:
            print("Инициализация кеша...")
            self.cache = RAGCache(db_path=cache_db_path, namespace=self._cache_namespace())
        else:
            print("Кеш ответов отключён (RAG_CACHE_ENABLED не включён): вопросы и ответы не сохраняются")

        print("RAG Pipeline инициализирован (API mode)")

    def _cache_namespace(self) -> str:
        """Всё, от чего зависит ответ, кроме текста вопроса."""
        return f"{self.corpus_id}|{self.model}|{self.top_k}|{PROMPT_VERSION}|{self.max_distance}"

    def _format_context_block(self, doc: Dict[str, Any], index: int) -> str:
        meta = doc.get("metadata") or {}
        src = source_label(meta)
        kind = meta.get("source_kind", "")
        heading = meta.get("section_heading", "")
        head = f"Фрагмент {index} [{src}"
        if kind:
            head += f", тип: {kind}"
        head += "]"
        if heading:
            head += f"\nЗаголовок/якорь: {heading}"
        return f"{head}\n{doc['text']}\n"

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        parts = [self._format_context_block(d, i) for i, d in enumerate(context_docs, start=1)]
        context = "\n---\n".join(parts)

        return f"""{PROMPT_INTRO}

Фрагменты базы знаний:
{context}

Вопрос пользователя: {query}

Инструкции:
{PROMPT_INSTRUCTIONS}

Ответ:"""

    def _generate_answer(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Текст ответа и finish_reason модели (stop, length, content_filter, ...)."""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        answer = (choice.message.content or "").strip()
        if not answer:
            raise RuntimeError(f"Модель вернула пустой ответ (finish_reason={finish_reason})")
        return answer, finish_reason

    def _resolve_metric(self) -> Optional[str]:
        """Метрика расстояния коллекции читается один раз и не предполагается."""
        if not self._metric_resolved:
            self._distance_metric = collection_metric(self.vector_store.collection)
            self._metric_resolved = True
            if self._distance_metric is None:
                print("[!] Метрика расстояния коллекции не определена: проверка достаточности данных отключена")
        return self._distance_metric

    def _no_evidence_result(self, user_query: str, assessment: EvidenceAssessment) -> Dict[str, Any]:
        """Ответ без вызова чат-модели: в корпусе нет достаточно близких фрагментов."""
        return {
            "query": user_query,
            "answer": NO_EVIDENCE_ANSWER,
            "from_cache": False,
            "context_docs": [],
            "model": None,
            "mode": "API",
            "no_evidence": True,
            "evidence": assessment.as_dict(),
        }

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        # Текст вопроса в stdout (docker logs) не выводится: он остаётся только в ответе вызывающему.
        print(f"\n{'='*60}")
        print("Запрос получен (текст вопроса не выводится в лог)")
        print(f"{'='*60}")

        use_cache = use_cache and self.cache is not None
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)
            if cached_result:
                print("[+] Ответ найден в кеше")
                ctx = _normalize_cached_context(cached_result.get("context"))
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": ctx,
                    "cached_at": cached_result.get("created_at"),
                    "incomplete": False,
                }
            print("[-] Ответ не найден в кеше")

        print("[*] Поиск релевантных документов через API...")
        candidate_k = max(self.top_k * 3, self.top_k + 5)
        raw_docs = self.vector_store.search(user_query, top_k=candidate_k)
        # Без порога метрика не нужна: не читаем её и не предупреждаем о ней зря.
        metric = self._resolve_metric() if self.max_distance is not None else None
        assessment = assess_evidence(raw_docs, metric, self.max_distance)
        if not assessment.sufficient:
            print(
                f"[-] Недостаточно данных в корпусе ({assessment.reason}, "
                f"лучшее расстояние {assessment.best_distance}, порог {assessment.max_distance}); "
                "чат-модель не вызывается"
            )
            return self._no_evidence_result(user_query, assessment)
        context_docs = deduplicate_context_docs(raw_docs, max_docs=self.top_k)
        print(
            f"[+] Найдено {len(raw_docs)} кандидатов, "
            f"после дедупликации: {len(context_docs)} документов"
        )

        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)

        print(f"[*] Генерация ответа через OpenAI API ({self.model})...")
        answer, finish_reason = self._generate_answer(prompt)
        notice = INCOMPLETE_NOTICES.get(finish_reason or "")
        incomplete = notice is not None
        if notice:
            print(f"[!] Ответ неполный (finish_reason={finish_reason}): добавлено предупреждение, в кеш не сохраняется")
            answer = f"{answer}\n\n{notice}"
        print("[+] Ответ получен от API")

        if use_cache and not incomplete:
            print("[*] Сохранение в кеш...")
            context_for_cache = [
                {
                    "text": doc["text"],
                    "metadata": doc.get("metadata") or {},
                    "id": doc.get("id"),
                }
                for doc in context_docs
            ]
            self.cache.set(user_query, answer, context_for_cache)
            print("[+] Сохранено в кеш")

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API",
            "incomplete": incomplete,
            "finish_reason": finish_reason,
        }

    def _cache_stats(self) -> Dict[str, Any]:
        if self.cache is None:
            return {
                "enabled": False,
                "total_entries": 0,
                "oldest_entry": None,
                "newest_entry": None,
                "db_size_mb": 0.0,
            }
        return {"enabled": True, **self.cache.get_stats()}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self._cache_stats(),
            "model": self.model,
            "mode": "API",
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "max_distance": self.max_distance,
            "corpus_version": os.getenv("RAG_CORPUS_VERSION", "1"),
            "corpus_id": self.corpus_id,
        }


if __name__ == "__main__":
    import sys

    try:
        pipeline = RAGPipeline()

        test_queries = [
            "Когда независимая гарантия вступает в силу по ГК РФ?",
            "Какие основания для отказа заказчика в принятии независимой гарантии предусмотрены 44-ФЗ?",
            "Может ли гарант ссылаться на основное обязательство при отказе бенефициару?",
        ]

        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")

        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")

        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:\n{stats}")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
