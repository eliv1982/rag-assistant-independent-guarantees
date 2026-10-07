#!/usr/bin/env bash
# Smoke-тест Docker-образа. Не обращается к OpenAI: ключ в контейнер не передаётся.
#
# Использование: scripts/docker_smoke.sh [образ]   (по умолчанию rag-guarantees:ci)
# Образ должен быть уже собран: docker build -t rag-guarantees:ci .
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash на Windows не должен переписывать пути вида /app/runtime

IMAGE="${1:-rag-guarantees:ci}"
PORT="${SMOKE_PORT:-18010}"
NAME="rag-smoke-$$"
VOLUME="${NAME}-runtime"
BASE="http://127.0.0.1:${PORT}"

fail() { echo "SMOKE FAIL: $*" >&2; docker logs --tail 40 "$NAME" >&2 || true; exit 1; }
ok() { echo "ok - $*"; }

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker volume rm "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

start_container() {
  # Так же, как в docker-compose.yml: именованный том на /app/runtime, порт только на localhost.
  docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" -v "${VOLUME}:/app/runtime" "$IMAGE" >/dev/null
}

wait_for() { # wait_for <описание> <секунды> <команда...>
  local what="$1" limit="$2"; shift 2
  for _ in $(seq 1 "$limit"); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  fail "не дождались: ${what}"
}

http_code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

docker volume create "$VOLUME" >/dev/null
start_container

wait_for "/health отвечает" 60 curl -fsS "${BASE}/health"
[ "$(curl -fsS "${BASE}/health")" = '{"status":"ok"}' ] || fail "неожиданный ответ /health"
ok "/health вернул {\"status\":\"ok\"}"

wait_for "HEALTHCHECK = healthy" 120 \
  sh -c "[ \"\$(docker inspect -f '{{.State.Health.Status}}' '$NAME')\" = healthy ]"
ok "Docker HEALTHCHECK: healthy"

uid="$(docker exec "$NAME" id -u)"
[ "$uid" != "0" ] || fail "контейнер работает от root"
ok "процесс не от root (uid ${uid})"

docker exec "$NAME" sh -c 'touch /app/runtime/.write_test && rm /app/runtime/.write_test' \
  || fail "/app/runtime недоступен для записи пользователю app"
ok "/app/runtime доступен для записи"

page="$(curl -fsS "${BASE}/")"
echo "$page" | grep -q "RAG Assistant" || fail "главная страница не открылась"
if echo "$page" | grep -qi "urdg"; then fail "в UI осталась ссылка на URDG"; fi
ok "главная страница без URDG"

# Ключа нет: ассистент не настроен -> 503 без обращения к OpenAI, ошибка попадает в журнал.
[ "$(http_code -X POST --data-urlencode 'question=smoke test' "${BASE}/ask")" = "503" ] \
  || fail "/ask без OPENAI_API_KEY должен вернуть 503"
ok "/ask без ключа -> 503"

[ "$(http_code -X POST --data-urlencode 'question=' "${BASE}/ask")" = "400" ] \
  || fail "/ask с пустым вопросом должен вернуть 400"
ok "/ask с пустым вопросом -> 400"

stats="$(curl -fsS "${BASE}/stats")"
echo "$stats" | grep -q "status-error" || fail "ошибка запроса не записана в журнал (/stats)"
docker exec "$NAME" test -s /app/runtime/logs.db || fail "logs.db не создан в /app/runtime"
ok "журнал пишется в /app/runtime/logs.db"

# Данные должны переживать пересоздание контейнера (том сохраняется).
docker rm -f "$NAME" >/dev/null
start_container
wait_for "/health после пересоздания" 60 curl -fsS "${BASE}/health"
curl -fsS "${BASE}/stats" | grep -q "status-error" || fail "журнал не сохранился после пересоздания контейнера"
ok "журнал сохранился после пересоздания контейнера"

echo "SMOKE PASSED"
