"""Shared LLM client. OpenRouter keys must use the OpenRouter base URL."""

import os
import openai


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def llm_client() -> openai.OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if key.startswith("sk-or-") or "openrouter.ai" in base_url:
        return openai.OpenAI(
            api_key=key,
            base_url=base_url or OPENROUTER_BASE,
        )
    return openai.OpenAI(api_key=key or None, base_url=base_url or None)


def llm_model() -> str:
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if (key.startswith("sk-or-") or "openrouter.ai" in base_url) and "/" not in model:
        return f"openai/{model}"
    return model
