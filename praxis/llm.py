"""Thin wrapper around litellm for all agent LLM calls."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from litellm import completion as _default_completion

DEFAULT_MODEL = "groq/llama-3.1-8b-instant"


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("PRAXIS_MODEL") or DEFAULT_MODEL


class LLMClient:
    """Callable-completion wrapper; inject a fake for tests."""

    def __init__(
        self,
        completion: Callable[..., Any] | None = None,
        model: str | None = None,
    ) -> None:
        self._completion = completion or _default_completion
        self._model = _resolve_model(model)

    def call(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": _resolve_model(model) or self._model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
        }
        response = self._completion(**kwargs)
        return response["choices"][0]["message"]["content"]


_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Return the shared module-level client (lazily created)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def call_llm(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    completion: Callable[..., Any] | None = None,
) -> str:
    """Call an LLM, optionally injecting a completion function for tests."""
    client = LLMClient(completion=completion) if completion else get_client()
    return client.call(prompt, system=system, model=model)
