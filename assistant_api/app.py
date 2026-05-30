"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from db_logger import DatabaseLogger
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
# Ищем .env в корне проекта (на уровень выше)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент (API Mode)                        ║
║  Retrieval-Augmented Generation через OpenAI API        ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("💾 Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"🌐 Источник: OpenAI API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n💬 Ответ:\n{result['answer']}")

    # Полный список фрагментов контекста (раньше показывались только 2 — см. RAG_CLI_CONTEXT_MAX_ITEMS)
    ctx = result.get("context_docs") or []
    if ctx:
        max_items = int(os.getenv("RAG_CLI_CONTEXT_MAX_ITEMS", "0"))
        preview_chars = int(os.getenv("RAG_CLI_CONTEXT_PREVIEW_CHARS", "220"))
        docs = ctx[:max_items] if max_items > 0 else ctx
        src = "кеша" if result["from_cache"] else "ретрива"
        note = f" (показано {len(docs)} из {len(ctx)})" if len(docs) < len(ctx) else ""
        print(f"\n📚 Контекст из {src} ({len(ctx)} фрагментов){note}:")
        for i, doc in enumerate(docs, 1):
            text = doc["text"] if isinstance(doc, dict) else str(doc)
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            label = meta.get("source_display", "")
            heading = (meta.get("section_heading") or "").strip()
            if len(text) > preview_chars:
                preview = text[:preview_chars].rstrip() + "…"
            else:
                preview = text
            parts = [f"   {i}."]
            if label:
                parts.append(f"[{label}]")
            if heading:
                parts.append(f"{heading[:120]}{'…' if len(heading) > 120 else ''}")
            print(" ".join(parts))
            print(f"      {preview}")
    
    print(f"{'─'*60}\n")


def _interaction_log_fields(
    result: Any,
    pipeline: RAGPipeline,
) -> Dict[str, Any]:
    """Извлечение полей для DatabaseLogger из результата pipeline."""
    if isinstance(result, dict):
        context_docs = result.get("context_docs") or []
        sources_count = len(context_docs) if isinstance(context_docs, list) else None
        return {
            "response": str(result.get("answer", "")),
            "from_cache": bool(result.get("from_cache", False)),
            "model": result.get("model") or pipeline.model,
            "top_k": pipeline.top_k,
            "sources_count": sources_count,
        }

    return {
        "response": str(result),
        "from_cache": False,
        "model": pipeline.model,
        "top_k": pipeline.top_k,
        "sources_count": None,
    }


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\n🤖 Модель: {stats['model']}")
    print(f"🔢 top_k: {stats.get('top_k', '—')}, max_tokens: {stats.get('max_tokens', '—')}")
    print(f"📌 Версия корпуса (кеш): {stats.get('corpus_version', '—')}")
    print(f"🌐 Режим: {stats['mode']}")
    print(f"{'═'*60}\n")


def main():
    """Главная функция приложения."""
    print_banner()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    try:
        # Инициализация RAG pipeline
        print("🚀 Инициализация системы...\n")
        _here = Path(__file__).resolve().parent
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path=str(_here / "api_rag_cache.db"),
            persist_directory=os.getenv("RAG_CHROMA_PATH", str(_here / "chroma_db")),
            model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
        )
        print("\n✅ Система готова к работе!\n")

        logger = DatabaseLogger()

    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("💭 Ваш вопрос: ").strip()
            
            # Обработка специальных команд
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("✅ Кеш очищен")
                continue
            
            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue

            start = time.perf_counter()
            try:
                result = pipeline.query(user_input)
                response_time_ms = int((time.perf_counter() - start) * 1000)

                fields = _interaction_log_fields(result, pipeline)
                logger.log_interaction(
                    query=user_input,
                    response=fields["response"],
                    from_cache=fields["from_cache"],
                    response_time_ms=response_time_ms,
                    model=fields["model"],
                    top_k=fields["top_k"],
                    sources_count=fields["sources_count"],
                    interface="cli",
                )

                print_response(result)

            except Exception as e:
                response_time_ms = int((time.perf_counter() - start) * 1000)
                logger.log_error(
                    query=user_input,
                    error_message=str(e),
                    response_time_ms=response_time_ms,
                    model=pipeline.model,
                    top_k=pipeline.top_k,
                    interface="cli",
                )
                print(f"\n❌ Ошибка: {e}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

