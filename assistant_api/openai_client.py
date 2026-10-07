"""
Единая настройка клиента OpenAI: таймаут, повторы на уровне SDK, опциональный base_url (прокси-шлюз).
Переменные окружения: OPENAI_TIMEOUT, OPENAI_MAX_RETRIES, OPENAI_BASE_URL.
"""

import os
from typing import Optional

from openai import OpenAI


# Значение из .env.example: после «cp .env.example .env» без правки ключа это не настоящий ключ.
_PLACEHOLDER_API_KEYS = frozenset({"your_openai_api_key_here"})


def api_key_configured() -> bool:
    """Задан ли настоящий OPENAI_API_KEY (пустое значение и заглушка из .env.example не считаются)."""
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return bool(key) and key not in _PLACEHOLDER_API_KEYS


def get_openai_client() -> OpenAI:
    if not api_key_configured():
        raise ValueError("OPENAI_API_KEY не установлен")
    api_key = os.environ["OPENAI_API_KEY"].strip()
    timeout = float(os.getenv("OPENAI_TIMEOUT", "180"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
    base_url: Optional[str] = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    kwargs = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
