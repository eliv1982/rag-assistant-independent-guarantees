"""
Единая настройка клиента OpenAI: таймаут, повторы на уровне SDK, опциональный base_url (прокси-шлюз).
Переменные окружения: OPENAI_TIMEOUT, OPENAI_MAX_RETRIES, OPENAI_BASE_URL.
"""

import os
from typing import Optional

from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не установлен")
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
