"""Provider-agnostic chat completion client.

LangChain is intentionally not a dependency of this module. Everything the
application asks of a model is a single system+user prompt with no tool use, no
retrieved-memory chaining and no agent loop, so a direct HTTP call is fewer
moving parts and one fewer large dependency to break at deploy time.

Every function here degrades to ``None`` rather than raising, so a provider
outage can never take down an endpoint that has a deterministic fallback.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 30.0


def provider_ready() -> tuple[str, str, str] | None:
    """Return ``(provider, model, api_key)`` when a provider is configured."""
    return settings.resolve_llm()


async def chat_completion(
    system: str,
    user: str,
    max_tokens: int = 400,
) -> tuple[str, str] | None:
    """Return ``(text, provider_name)``, or ``None`` if unavailable or failed."""
    resolved = provider_ready()
    if resolved is None:
        return None

    provider, model, api_key = resolved

    import httpx

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
    elif provider == "gemini":
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        headers = {}
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": max_tokens,
            },
        }
    else:
        base = (
            "https://openrouter.ai/api/v1"
            if provider == "openrouter"
            else "https://api.openai.com/v1"
        )
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "temperature": settings.llm_temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("LLM request failed (%s): %s", provider, exc)
        return None

    try:
        if provider == "anthropic":
            text = data["content"][0]["text"]
        elif provider == "gemini":
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("unexpected LLM response shape (%s): %s", provider, exc)
        return None

    text = (text or "").strip()
    return (text, provider) if text else None
