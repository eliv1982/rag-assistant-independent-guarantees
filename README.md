# RAG Assistant: Independent Guarantees

Портфолио **MVP RAG-ассистента** по независимым гарантиям (с упором на закупочное регулирование). Проект включает **web-интерфейс на FastAPI**, **CLI-режим**, RAG-поиск по корпусу **ГК РФ / 44-ФЗ / 223-ФЗ / постановлений Правительства РФ / обзоров практики ВС РФ**, SQLite-кеш ответов, **SQLite-логирование взаимодействий**, **страницу метрик**, **Docker deployment** и **CI**.

Ответы носят **информационный характер** и **не являются юридической консультацией**.

---

## Что умеет MVP

- отвечает на вопросы по независимым гарантиям на основе RAG-корпуса;
- показывает использованные **context docs / источники**;
- различает уровни регулирования (ГК РФ → 44-ФЗ / 223-ФЗ → постановления Правительства → практика ВС РФ);
- работает через **FastAPI web UI** и **CLI**;
- кеширует ответы в SQLite (кеш привязан к версии корпуса, модели и промпта);
- логирует web- и CLI-взаимодействия; сбой журнала не ломает ответ пользователю;
- показывает статистику запросов на **`/stats`**;
- поддерживает **safe Markdown rendering** в web UI;
- использует **deduplication** для retrieved context docs;
- запускается локально и через **Docker** (непривилегированный пользователь, healthcheck, постоянное хранилище).

---

## База знаний

Корпус — исходные тексты норм без пересказа, `assistant_api/data/*.txt`:

| Файл | Источник | Уровень |
|------|----------|---------|
| `GK_368_379.txt` | ГК РФ, ст. 368–379 (§ 6, независимая гарантия) | общие нормы |
| `44FZ_article_45.txt` | 44-ФЗ, ст. 45 | специальные нормы о закупках |
| `223FZ_article_3_4_guarantees.txt` | 223-ФЗ, ст. 3.4 (положения о гарантиях) | специальные нормы о закупках |
| `PP_1005.txt` | Постановление Правительства РФ от 08.11.2013 № 1005 | требования, реестры, формы (44-ФЗ) |
| `PP_1397.txt` | Постановление Правительства РФ от 09.08.2022 № 1397 | требования, формы, реестры (223-ФЗ, СМСП) |
| `VS_independent_guarantee_2019.txt` | Обзор практики ВС РФ от 05.06.2019 (все 17 позиций) | толкование и применение |
| `VS_contract_system_2017_guarantees.txt` | Обзор практики ВС РФ от 28.06.2017 (вводная часть и позиции 25, 30) | толкование и применение |

Политика корпуса и примечания: [assistant_api/data/README.md](assistant_api/data/README.md).
Состав, метаданные источников и тип нарезки задаются в [assistant_api/corpus_config.py](assistant_api/corpus_config.py).

Статья 369 ГК РФ утратила силу с 1 июня 2015 года (Федеральный закон от 08.03.2015 № 42-ФЗ); в корпусе она сохранена только одной строкой-статусом, без прежнего содержания. Для норм, которые во фрагментах помечены как утратившие силу, промпт требует не применять их как действующие.

### Идентификатор корпуса (`corpus_id`)

`corpus_id` — короткий отпечаток, который считается автоматически по содержимому файлов корпуса, метаданным источников, параметрам индексации (`RAG_CHUNK_*`, `RAG_EMBEDDING_MODEL`) и `RAG_CORPUS_VERSION`. Он определяет:

- имя коллекции Chroma: `guarantees_<corpus_id>`;
- ключ кеша ответов (вместе с моделью чата, `RAG_TOP_K` и версией промпта).

Поэтому после изменения корпуса приложение **само** строит новый индекс при первом запросе (с обращениями к API эмбеддингов), а ответы, посчитанные по старому корпусу или промпту, из кеша не отдаются. Старая коллекция остаётся на диске и больше не используется; чтобы освободить место, удалите каталог Chroma (см. ниже).

---

## Архитектура

| Компонент | Назначение |
|-----------|------------|
| `assistant_api/web_app.py` | FastAPI web UI: `/`, `/ask`, `/stats`, `/health` |
| `assistant_api/app.py` | CLI-интерфейс |
| `assistant_api/rag_pipeline.py` | RAG pipeline: кеш → Chroma → промпт → LLM |
| `assistant_api/corpus_config.py` | Состав корпуса, `corpus_id`, имя коллекции |
| `assistant_api/vector_store.py` | ChromaDB, chunking, embeddings |
| `assistant_api/cache.py` | SQLite response cache |
| `assistant_api/db_logger.py` | SQLite interaction logger |
| `assistant_api/rag_helpers.py` | Helpers / factory для web-слоя |
| `assistant_api/retrieval_utils.py` | Deduplication retrieved context docs |
| `assistant_api/openai_client.py` | OpenAI-compatible client |
| `assistant_api/evaluate_ragas.py` | RAGAS evaluation |
| `assistant_api/templates/`, `assistant_api/static/` | Web UI templates и styles |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Docker deployment |
| `scripts/docker_smoke.sh` | Smoke-тест Docker-образа (без обращений к OpenAI) |
| `.github/workflows/ci.yml` | CI: тесты и сборка Docker-образа |
| `requirements.txt`, `requirements-dev.txt`, `requirements-eval.txt` | Зависимости: runtime / тесты / RAGAS |

---

## Зависимости

| Файл | Для чего | Что внутри |
|------|----------|------------|
| `requirements.txt` | запуск (web, CLI) и Docker-образ | openai, chromadb, fastapi, uvicorn, jinja2, python-multipart, markdown, bleach, pysbd, python-dotenv |
| `requirements-dev.txt` | тесты | `-r requirements.txt` + `httpx` (нужен `fastapi.testclient.TestClient`) |
| `requirements-eval.txt` | RAGAS (`evaluate_ragas.py`) | `-r requirements.txt` + ragas, langchain-*, datasets, pandas, pydantic<3 |

В Docker-образ попадает только `requirements.txt`: тесты и RAGAS его не раздувают.

---

## Logging and metrics

`DatabaseLogger` пишет каждое взаимодействие в **SQLite**.

| Параметр | Значение |
|----------|----------|
| Путь к БД | `LOGS_DB_PATH` |
| Локально | не задавайте: по умолчанию `assistant_api/logs.db` |
| В Docker | `/app/runtime/logs.db` — задаёт `docker-compose.yml` (и `Dockerfile`), а не `.env` |

Не копируйте значения вида `/app/runtime/...` в локальный `.env`: такого каталога на хосте нет. `.env.example` держит `LOGS_DB_PATH`, `RAG_CHROMA_PATH` и `RAG_CACHE_DB_PATH` закомментированными. В контейнере значения из `environment` в compose приоритетнее `env_file`, поэтому локальные пути из `.env` в контейнер не попадают.

**Логируются:** `query`, `response`, `interface` (`cli` / `web`), `status` (`success` / `error`), `response_time_ms`, `from_cache`, `model`, `top_k`, `sources_count`, `error_message`.

**Не логируются:** credentials, tokens, значения переменных окружения и прочие секреты. Текст ошибки перед записью очищается от API-ключей (значение `OPENAI_API_KEY` и токены вида `sk-…`) и ограничивается по длине.

**Устойчивость:** недостающие каталоги для БД создаются; соединения всегда закрываются; сбой записи в журнал не превращает успешный ответ в ошибку и не скрывает исходную ошибку запроса; при недоступном журнале `/stats` отвечает страницей с пояснением (HTTP 503), а не падает.

**`/stats`** показывает: total requests, successful / failed, cache hits, cache hit rate, average response time, таблицу recent requests.

---

## Web UI routes

| Маршрут | Назначение |
|---------|------------|
| `GET /` | форма вопроса |
| `POST /ask` | обработка вопроса: `200` ответ; `400` пустой вопрос; `503` ассистент не настроен (нет `OPENAI_API_KEY`); `500` ошибка обработки |
| `GET /stats` | статистика взаимодействий (`503`, если журнал недоступен) |
| `GET /health` | health check (`{"status": "ok"}`), без обращения к OpenAI |

Пользователю показываются обобщённые сообщения об ошибках; подробности (тип и текст исключения) пишутся в серверный лог и журнал взаимодействий.

---

## Local run

**Требования:** Python **3.11+**. Скопируйте `.env.example` → `.env` в корне репозитория и укажите `OPENAI_API_KEY` (пути хранилища локально не задавайте).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn assistant_api.web_app:app --reload --host 127.0.0.1 --port 8000
```

- Web UI: http://127.0.0.1:8000  
- Stats: http://127.0.0.1:8000/stats  
- Health: http://127.0.0.1:8000/health  

При первом вопросе индекс корпуса строится через API эмбеддингов (платные вызовы); дальше он переиспользуется, пока не изменится `corpus_id`.

---

## Docker run

```bash
cp .env.example .env     # замените заглушку OPENAI_API_KEY на настоящий ключ
docker compose up -d --build
```

Пока в `OPENAI_API_KEY` пусто или осталась заглушка из `.env.example`, `/health` и `/stats` работают, а `/ask` отвечает `503` без обращений к OpenAI.

- Web UI: http://127.0.0.1:8010 — порт публикуется **только на localhost** (`127.0.0.1:8010 → контейнер 8000`).
- Контейнер работает **не от root** (uid 10001); для записи доступен только `/app/runtime`.
- В образе настроен `HEALTHCHECK` по `/health`: `docker ps` покажет `healthy`.
- В образ не попадают `.env`, локальные `chroma_db/`, кеш и логи, тесты и RAGAS-зависимости.

### Постоянное хранилище

Журнал (`logs.db`), индекс Chroma (`chroma_db/`) и кеш ответов (`api_rag_cache.db`) лежат в `/app/runtime`, это именованный том `rag_runtime`. Данные переживают пересборку образа и пересоздание контейнера, а корпус не переиндексируется заново (экономия на эмбеддингах).

```bash
# выгрузить журнал на хост
docker compose cp rag-guarantees:/app/runtime/logs.db ./logs.db

# сбросить индекс и кеш, сохранив журнал
docker compose exec rag-guarantees sh -c 'rm -rf /app/runtime/chroma_db /app/runtime/api_rag_cache.db'

# удалить ВСЕ данные (журнал, индекс, кеш)
docker compose down -v
```

Именованный том (а не папка `./runtime`) выбран, чтобы у непривилегированного пользователя всегда были права на запись, в том числе на Linux-хосте.

### Переход с прежней схемы (`./runtime` + порт на всех интерфейсах)

Раньше `compose` публиковал порт `8010` на всех интерфейсах, а журнал лежал в папке `./runtime`. Теперь порт доступен только с самого хоста, а данные хранятся в томе `rag_runtime`. Что сделать на сервере, где сервис уже запущен:

1. Доступ из сети перестанет работать — настройте reverse proxy или SSH-туннель (раздел ниже).
2. Прежний журнал можно перенести в том. После `docker compose up -d --build`:

```bash
docker compose cp ./runtime/logs.db rag-guarantees:/app/runtime/logs.db
docker compose exec -u root rag-guarantees chown app:app /app/runtime/logs.db
```

`chown` обязателен: `docker cp` создаёт файл от root, и непривилегированный пользователь не сможет в него писать (SQLite: «attempt to write a readonly database»).

Индекс Chroma и кеш прежде жили внутри контейнера и не переносятся: корпус изменился, индекс в любом случае строится заново.

### Доступ из сети

Порт намеренно не открыт наружу: на `/stats` видны тексты вопросов и ответов, а у приложения нет авторизации. Чтобы опубликовать сервис, поставьте перед ним reverse proxy (Caddy, nginx, Traefik) с HTTPS и авторизацией и проксируйте на `127.0.0.1:8010`. Для разового доступа с другой машины подойдёт SSH-туннель: `ssh -L 8010:127.0.0.1:8010 user@server`.

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
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py"
python -m compileall assistant_api tests -q
python -c "import assistant_api.web_app; print('web_app import OK')"
```

Тесты не обращаются к OpenAI: pipeline, хранилище и клиент подменяются заглушками.

Smoke-тест Docker-образа (проверяет `/health`, `HEALTHCHECK`, отсутствие root, запись в `/app/runtime`, журнал и его сохранность после пересоздания контейнера; ключ OpenAI в контейнер не передаётся):

```bash
docker build -t rag-guarantees:ci .
bash scripts/docker_smoke.sh rag-guarantees:ci
```

### CI

`.github/workflows/ci.yml` на каждый push в `main` и pull request запускает: тесты (Python 3.11 и 3.12), проверку `docker-compose.yml` (порт только на localhost), сборку Docker-образа и smoke-тест.

---

## RAGAS evaluation

Дополнительная оценка качества retrieval и генерации (платные вызовы OpenAI):

```bash
python -m pip install -r requirements-eval.txt
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
| Chunking, overlap | `.env`: `RAG_CHUNK_*` |
| Промпт и структура ответа | `assistant_api/rag_pipeline.py` |
| Пути Chroma и кеша | `.env`: `RAG_CHROMA_PATH`, `RAG_CACHE_DB_PATH` (локально обычно не нужны) |

После изменения файлов корпуса, параметров нарезки, модели эмбеддингов или промпта ничего удалять не нужно: изменится `corpus_id` (и/или версия промпта), приложение создаст новую коллекцию и не будет отдавать старые ответы из кеша. `RAG_CORPUS_VERSION` нужен только чтобы принудительно переиндексировать корпус при неизменных файлах. Старые коллекции остаются на диске; чтобы очистить место, удалите `assistant_api/chroma_db` (локально) или каталог `chroma_db` в томе (Docker, см. выше).

Для провайдеров без OpenAI-compatible HTTP (например, GigaChat) потребуется точечная замена вызовов в `rag_pipeline.py` (`_generate_answer`) и `vector_store.py` (embeddings).

---

## Limitations

- портфолио MVP, не production-ready сервис;
- качество ответа зависит от корпуса, chunking и retrieval;
- ассистент может ошибаться или неполно извлекать нормы из контекста;
- корпус охватывает ограниченный набор норм (см. «База знаний») и не отслеживает последующие изменения законодательства;
- ответы **не являются юридической консультацией**;
- **`/stats` в публичном деплое** следует закрывать авторизацией;
- для production нужны auth, rate limits, retention policy for logs, better observability.

---

## Future improvements

- reranking retrieved chunks;
- better legal chunking by article / paragraph / point;
- auth for `/stats` и admin endpoints;
- CSV export / admin page для логов;
- Docker + reverse proxy / domain / HTTPS;
- расширенный evaluation set и автоматизация RAGAS в CI.

---

## License and disclaimer

Ответы ассистента носят информационный характер и **не являются юридической консультацией**. Корпус и модели нужно верифицировать под вашу задачу.
