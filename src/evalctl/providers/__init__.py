"""Provider registry.

`build()` is the only constructor callers should use: it resolves the name,
checks the credential early, and hands over a shared HTTP client so one
connection pool serves the whole run instead of one per request.
"""

from __future__ import annotations

from typing import Callable, Mapping

import httpx

from ..errors import FatalError
from ..spec import Endpoint
from .anthropic import AnthropicProvider
from .base import ChatRequest, ChatResponse, Provider, Usage
from .mock import MockProvider
from .openai import OpenAICompatProvider, OpenAIProvider

_REGISTRY: dict[str, type[Provider]] = {
    AnthropicProvider.name: AnthropicProvider,
    OpenAIProvider.name: OpenAIProvider,
    OpenAICompatProvider.name: OpenAICompatProvider,
    MockProvider.name: MockProvider,
}

# Friendly aliases for endpoints that speak Chat Completions. Each is just
# openai-compat with a base_url filled in, so `provider: ollama` needs no
# further configuration to work against a local server.
_ALIASES: Mapping[str, tuple[str, str | None, str | None]] = {
    # alias -> (provider name, default base_url, default api key env)
    "ollama": ("openai-compat", "http://localhost:11434", None),
    "vllm": ("openai-compat", "http://localhost:8000", None),
    "together": ("openai-compat", "https://api.together.xyz", "TOGETHER_API_KEY"),
    "groq": ("openai-compat", "https://api.groq.com/openai", "GROQ_API_KEY"),
    "openrouter": ("openai-compat", "https://openrouter.ai/api", "OPENROUTER_API_KEY"),
}


def available_providers() -> list[str]:
    return sorted(set(_REGISTRY) | set(_ALIASES))


def register(name: str, factory: type[Provider]) -> None:
    """Add a provider at runtime. Kept public so a fork can bolt on an internal
    gateway without editing this file."""
    _REGISTRY[name] = factory


def resolve_endpoint(endpoint: Endpoint) -> tuple[type[Provider], Endpoint]:
    """Expand an alias into a concrete provider class plus a filled-in endpoint."""
    name = endpoint.provider.strip().lower()
    if name in _ALIASES:
        target, base_url, key_env = _ALIASES[name]
        from dataclasses import replace

        endpoint = replace(
            endpoint,
            provider=target,
            base_url=endpoint.base_url or base_url,
            api_key_env=endpoint.api_key_env or key_env,
        )
        name = target
    cls = _REGISTRY.get(name)
    if cls is None:
        raise FatalError(
            f"unknown provider '{endpoint.provider}' for model '{endpoint.id}'. "
            f"Available: {', '.join(available_providers())}"
        )
    return cls, endpoint


def build(endpoint: Endpoint, client: httpx.AsyncClient | None = None) -> Provider:
    cls, resolved = resolve_endpoint(endpoint)
    return cls(resolved, client)


def needs_http(endpoint: Endpoint) -> bool:
    cls, _ = resolve_endpoint(endpoint)
    return cls is not MockProvider


def preflight(endpoint: Endpoint) -> None:
    """Check credentials before a run starts.

    Discovering a missing API key on trial 400 of 500 is the kind of avoidable
    waste this harness exists to prevent, so every endpoint is checked up front.
    """
    cls, resolved = resolve_endpoint(endpoint)
    provider = cls(resolved, None)
    if provider.requires_api_key:
        provider.api_key()  # raises FatalError with the variable name


def make_client(limits_total: int = 64) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=limits_total, max_keepalive_connections=limits_total
        ),
        follow_redirects=True,
        headers={"user-agent": "evalctl"},
    )


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Provider",
    "Usage",
    "available_providers",
    "build",
    "make_client",
    "needs_http",
    "preflight",
    "register",
    "resolve_endpoint",
]
