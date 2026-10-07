FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Приложение работает без root; единственный каталог для записи — /app/runtime.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Только runtime-зависимости (без тестов и RAGAS)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY assistant_api ./assistant_api

# Данные, которые должны переживать пересоздание контейнера: журнал, индекс Chroma, кеш ответов.
# Каталог принадлежит app, поэтому именованный том (docker-compose.yml) наследует эти права.
RUN mkdir -p /app/runtime && chown app:app /app/runtime
ENV LOGS_DB_PATH=/app/runtime/logs.db \
    RAG_CHROMA_PATH=/app/runtime/chroma_db \
    RAG_CACHE_DB_PATH=/app/runtime/api_rag_cache.db

USER app

EXPOSE 8000

# python:slim не содержит curl, поэтому проверка делается самим Python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "assistant_api.web_app:app", "--host", "0.0.0.0", "--port", "8000"]
