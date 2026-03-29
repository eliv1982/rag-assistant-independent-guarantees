"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from cache import RAGCache
from openai_client import get_openai_client
from corpus_config import default_corpus_entries
from vector_store import VectorStore

_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    load_dotenv(_env)
else:
    load_dotenv()


LEGAL_SYSTEM_PROMPT = (
    "Ты — ассистент по вопросам независимых гарантий (банковских и иных). "
    "Отвечаешь строго на основе переданных фрагментов базы знаний. "
    "Не выдавай юридических заключений и не подменяй консультацию юриста; "
    "формулируй осторожно, если контекст неполный."
)


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
        collection_name: str = "rag_collection",
        cache_db_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        corpus_entries: Optional[List[Dict[str, Any]]] = None,
        data_file: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")

        self._base_dir = Path(__file__).resolve().parent
        self.model = model or os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "1500"))
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))

        self.openai_client = get_openai_client()

        if persist_directory is None:
            persist_directory = os.getenv("RAG_CHROMA_PATH", str(self._base_dir / "chroma_db"))
        if cache_db_path is None:
            cache_db_path = str(self._base_dir / "api_rag_cache.db")

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        if self.vector_store.collection.count() == 0:
            if corpus_entries is not None:
                print("Загрузка корпуса (несколько источников)...")
                self.vector_store.load_corpus(corpus_entries, base_dir=self._base_dir)
            elif data_file:
                print(f"Загрузка документов из {data_file}...")
                self.vector_store.load_documents(data_file, base_dir=self._base_dir)
            else:
                print("Загрузка корпуса по умолчанию (ГК РФ, URDG, обзор ВС)...")
                self.vector_store.load_corpus(default_corpus_entries(), base_dir=self._base_dir)

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)

        print("RAG Pipeline инициализирован (API mode)")

    def _format_context_block(self, doc: Dict[str, Any], index: int) -> str:
        meta = doc.get("metadata") or {}
        src = meta.get("source_display") or meta.get("source") or "источник"
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

        return f"""Ты помогаешь разбирать вопросы по независимым гарантиям с опорой на фрагменты норм (ГК РФ), правил URDG (если есть в контексте) и обзоров судебной практики ВС РФ.

Фрагменты базы знаний:
{context}

Вопрос пользователя: {query}

Инструкции:
- Отвечай только на основе приведённых фрагментов. Если данных недостаточно, прямо укажи, чего не хватает (например, нет статьи ГК или нет позиции суда).
- Для каждого существенного тезиса укажи источник: «ГК РФ», «URDG» или «обзор практики ВС» — по тому, из какого фрагмента он взят (ориентируйся на пометки в тексте фрагмента).
- Если во фрагментах есть разные уровни регулирования (закон vs договорная подчинённость URDG vs обобщение судебной практики), не смешивай их молча: раздели логику («по закону…», «по URDG…», «по позициям из обзора…»).
- Не придумывай номера статей, дел и цитат, которых нет во фрагментах.
- Ответ на русском языке; структурируй списком, если это улучшает ясность.

Ответ:"""

    def _generate_answer(self, prompt: str) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        print(f"{'='*60}")

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
                }
            print("[-] Ответ не найден в кеше")

        print("[*] Поиск релевантных документов через API...")
        context_docs = self.vector_store.search(user_query, top_k=self.top_k)
        print(f"[+] Найдено {len(context_docs)} релевантных документов")

        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)

        print(f"[*] Генерация ответа через OpenAI API ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")

        if use_cache:
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
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API",
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "corpus_version": os.getenv("RAG_CORPUS_VERSION", "1"),
        }


if __name__ == "__main__":
    import sys

    try:
        pipeline = RAGPipeline()

        test_queries = [
            "Когда независимая гарантия вступает в силу по ГК РФ?",
            "Что такое надлежащее представление по URDG?",
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
