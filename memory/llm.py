"""Shared LLM client. Groq is active. OpenAI / OpenRouter stay commented."""

import os
import openai


def llm_client() -> openai.OpenAI:
    # --- OpenAI ---
    # return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- OpenRouter ---
    # return openai.OpenAI(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    # )

    # --- Groq (active) ---
    return openai.OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


def llm_model() -> str:
    # OpenAI: return os.getenv("LLM_MODEL", "gpt-4o-mini")
    # OpenRouter: model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    # if "/" not in model:
    #     return f"openai/{model}"
    # return model
    return os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")


def llm_model_fast() -> str:
    """Chat, routing, and intent splits. Same Groq key, separate rate bucket."""
    return os.getenv("LLM_MODEL_FAST", "qwen/qwen3.6-27b")


def llm_no_thinking() -> dict:
    """Qwen 3.6 thinks by default. Hide that so chat is the answer only."""
    return {
        "extra_body": {
            "reasoning_effort": "none",
            "reasoning_format": "hidden",
        }
    }
