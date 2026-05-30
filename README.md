# RAG Assistant: Independent Guarantees

Портфолио **MVP RAG-ассистента** по независимым гарантиям. Проект включает **web-интерфейс на FastAPI**, **CLI-режим**, RAG-поиск по корпусу **ГК РФ / URDG / обзора судебной практики**, SQLite-кеш ответов, **SQLite-логирование взаимодействий**, **страницу метрик** и **Docker deployment**.

Ответы носят **информационный характер** и **не являются юридической консультацией**.

---

## Что умеет MVP

- отвечает на вопросы по независимым гарантиям на основе RAG-корпуса;
- показывает использованные **context docs / источники**;
- работает через **FastAPI web UI** и **CLI**;
- кеширует ответы в SQLite;
- логирует web- и CLI-взаимодействия;
- показывает статистику запросов на **`/stats`**;
- поддерживает **safe Markdown rendering** в web UI;
- использует **deduplication** для retrieved context docs;
- запускается локально и через **Docker**.

---

## База знаний

По умолчанию корпус включает:

- положения **ГК РФ** о независимой гарантии;
- **URDG 2010** (унифицированные правила);
- **обзор судебной практики ВС РФ** по спорам с участием независимой гарантии.

Исходные тексты: `assistant_api/data/*.txt`  
Конфигурация корпуса и метаданные источников: `assistant_api/corpus_config.py`

---

## Архитектура

| Компонент | Назначение |
|-----------|------------|
| `assistant_api/web_app.py` | FastAPI web UI: `/`, `/ask`, `/stats`, `/health` |
| `assistant_api/app.py` | CLI-интерфейс |
| `assistant_api/rag_pipeline.py` | RAG pipeline: кеш → Chroma → промпт → LLM |
| `assistant_api/vector_store.py` | ChromaDB, chunking, embeddings |
| `assistant_api/cache.py` | SQLite response cache |
| `assistant_api/db_logger.py` | SQLite interaction logger |
| `assistant_api/rag_helpers.py` | Helpers / factory для web-слоя |
| `assistant_api/retrieval_utils.py` | Deduplication retrieved context docs |
| `assistant_api/openai_client.py` | OpenAI-compatible client |
| `assistant_api/evaluate_ragas.py` | RAGAS evaluation |
| `assistant_api/templates/`, `assistant_api/static/` | Web UI templates и styles |
| `Dockerfile`, `docker-compose.yml` | Docker deployment |
| `runtime/` | Runtime data на хосте (в т.ч. `logs.db` в Docker) |

---

## Logging and metrics

`DatabaseLogger` пишет каждое взаимодействие в **SQLite**.

| Параметр | Значение |
|----------|----------|
| Путь к БД | `LOGS_DB_PATH` |
| Локально по умолчанию | `assistant_api/logs.db` |
| В Docker | `/app/runtime/logs.db` (volume `./runtime:/app/runtime`) |

**Логируются:** `query`, `response`, `interface` (`cli` / `web`), `status` (`success` / `error`), `response_time_ms`, `from_cache`, `model`, `top_k`, `sources_count`, `error_message`.

**Не логируются:** credentials, tokens, значения переменных окружения и прочие секреты.

**`/stats`** показывает: total requests, successful / failed, cache hits, cache hit rate, average response time, таблицу recent requests.

---

## Web UI routes

| Маршрут | Назначение |
|---------|------------|
| `GET /` | форма вопроса |
| `POST /ask` | обработка вопроса |
| `GET /stats` | статистика взаимодействий |
| `GET /health` | health check (`{"status": "ok"}`) |

---

## Screenshots

![Web interface](screenshots/01_web_home.png)

![Stats page](screenshots/02_stats_page.png)

---

## Local run

**Требования:** Python **3.11+**, зависимости из `requirements.txt`. Скопируйте `.env.example` → `.env` в корне репозитория и заполните переменные.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn assistant_api.web_app:app --reload --host 127.0.0.1 --port 8000
```

- Web UI: http://127.0.0.1:8000  
- Stats: http://127.0.0.1:8000/stats  
- Health: http://127.0.0.1:8000/health  

---

## Docker run

```bash
cp .env.example .env
docker compose up -d --build
```

- Web UI: http://127.0.0.1:8010  
- Порт на хосте **8010** → контейнер **8000**  
- Runtime logs DB на хосте: `./runtime/logs.db`

---

## CLI run

```bash
cd assistant_api
python app.py
```

Команды в сессии: `stats`, `clear` (очистка кеша), `exit`.

---

## Checks

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall assistant_api tests -q
python -c "import assistant_api.web_app; print('web_app import OK')"
```

---

## RAGAS evaluation

Дополнительная оценка качества retrieval и генерации:

```bash
cd assistant_api
python evaluate_ragas.py
```

Скрипт вызывает `RAGPipeline` **без кеша**, использует эталоны из `EVALUATION_GROUND_TRUTHS` в `evaluate_ragas.py` и считает метрики **Faithfulness**, **Context precision**, **Context utilization**. Для отчёта зафиксируйте версии `ragas` и `langchain-*` из окружения.

---

## LLM / corpus customization

Проект использует **OpenAI Python SDK** для чата и эмбеддингов; RAG-логика отделена от провайдера.

| Задача | Где настраивать |
|--------|-----------------|
| Модель чата / эмбеддингов | `.env`: `RAG_CHAT_MODEL`, `RAG_EMBEDDING_MODEL` |
| OpenAI-compatible endpoint | `.env`: `OPENAI_BASE_URL` (`assistant_api/openai_client.py`) |
| Состав корпуса и метаданные | `assistant_api/corpus_config.py` |
| Chunking, overlap | `.env`: `RAG_CHUNK_*`, `RAG_EMBEDDING_MODEL` |
| Промпт и структура ответа | `assistant_api/rag_pipeline.py` |
| Путь Chroma | `.env`: `RAG_CHROMA_PATH` или аргументы `RAGPipeline` |

После смены корпуса или параметров нарезки: удалите `assistant_api/chroma_db` (или задайте новый `RAG_CHROMA_PATH`), увеличьте **`RAG_CORPUS_VERSION`**, при необходимости очистите `assistant_api/api_rag_cache.db`, затем перезапустите — выполнится переиндексация.

Для провайдеров без OpenAI-compatible HTTP (например, GigaChat) потребуется точечная замена вызовов в `rag_pipeline.py` (`_generate_answer`) и `vector_store.py` (embeddings).

---

## Limitations

- портфолио MVP, не production-ready сервис;
- качество ответа зависит от корпуса, chunking и retrieval;
- ассистент может ошибаться или неполно извлекать нормы из контекста;
- ответы **не являются юридической консультацией**;
- **`/stats` в публичном деплое** следует закрывать авторизацией;
- для production нужны auth, rate limits, retention policy for logs, better observability.

---

## Future improvements

- reranking retrieved chunks;
- better legal chunking by article / paragraph;
- auth for `/stats` и admin endpoints;
- CSV export / admin page для логов;
- Docker + reverse proxy / domain / HTTPS;
- расширенный evaluation set и автоматизация RAGAS в CI.

---

## License and disclaimer

Ответы ассистента носят информационный характер и **не являются юридической консультацией**. Корпус и модели нужно верифицировать под вашу задачу.
