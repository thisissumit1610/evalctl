"""OpenAI Chat Completions, and anything that speaks the same wire format.

`openai-compat` is the same implementation with a mandatory `base_url`, which
covers vLLM, Ollama, Together, Groq, OpenRouter and most self-hosted gateways.
That is the practical reason this file is worth more than one provider: the
Chat Completions shape is the lingua franca, so one client reaches most of the
ecosystem.
"""

from __future__ import annotations

import time
from typing import Any

from ..errors import FatalError, TransientError
from .base import ChatRequest, ChatResponse, Provider, Usage, post_json


class OpenAIProvider(Provider):
    name = "openai"
    default_api_key_env = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        common, extra = self.split_params(request.params)
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": role, "content": content} for role, content in request.messages)

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if "temperature" in common:
            payload["temperature"] = float(common["temperature"])
        if "top_p" in common:
            payload["top_p"] = float(common["top_p"])
        if "seed" in common:
            payload["seed"] = int(common["seed"])
        if "stop" in common:
            payload["stop"] = common["stop"]
        if "max_tokens" in common:
            # Reasoning models renamed this field. If a suite passes
            # max_completion_tokens through as a provider-specific extra, defer
            # to it rather than sending both and getting a 400.
            if "max_completion_tokens" in extra:
                pass
            else:
                payload["max_tokens"] = int(common["max_tokens"])
        payload.update(extra)

        headers = {
            "Authorization": f"Bearer {self.api_key()}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        data = await post_json(
            self.client,
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            payload=payload,
            provider=self.name,
            timeout=self.endpoint.limits.timeout_s if self.endpoint.limits else 120.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TransientError(f"{self.name} response had no choices: {str(data)[:400]}")
        first = choices[0] or {}
        message = first.get("message") or {}
        text = message.get("content")
        if text is None:
            # A content filter or a pure tool-call turn lands here. Treat it as
            # an empty answer rather than crashing: the scorer will mark it
            # wrong, which is the honest outcome, and the finish_reason is kept
            # so `evalctl show` can explain why.
            text = ""
        usage = data.get("usage") or {}
        return ChatResponse(
            text=str(text),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            finish_reason=first.get("finish_reason"),
            model=data.get("model"),
            response_id=data.get("id"),
            latency_ms=latency_ms,
        )


class OpenAICompatProvider(OpenAIProvider):
    """Same wire format, self-hosted or third-party endpoint."""

    name = "openai-compat"
    default_api_key_env = "OPENAI_COMPAT_API_KEY"
    default_base_url = None
    requires_api_key = False  # local servers usually take no key

    def __init__(self, endpoint, client=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(endpoint, client)
        if not self.base_url:
            raise FatalError(
                f"model '{endpoint.id}' uses provider 'openai-compat', which needs an explicit "
                "base_url (e.g. http://localhost:11434 for Ollama)"
            )

    def api_key(self) -> str:
        import os

        env_name = self.endpoint.api_key_env or self.default_api_key_env
        return os.environ.get(env_name, "").strip() if env_name else ""
