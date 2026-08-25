"""Anthropic Messages API."""

from __future__ import annotations

import time
from typing import Any

from ..errors import TransientError
from .base import ChatRequest, ChatResponse, Provider, Usage, join_text_blocks, post_json

ANTHROPIC_VERSION = "2023-06-01"
# The Messages API rejects a request without max_tokens, so the harness has to
# pick something when a suite does not. 1024 is generous for benchmark answers
# and small enough that a runaway generation cannot quietly cost real money.
DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(Provider):
    name = "anthropic"
    default_api_key_env = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        common, extra = self.split_params(request.params)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(common.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "messages": [{"role": role, "content": content} for role, content in request.messages],
        }
        if request.system:
            payload["system"] = request.system
        if "temperature" in common:
            payload["temperature"] = float(common["temperature"])
        if "top_p" in common:
            payload["top_p"] = float(common["top_p"])
        if "stop" in common:
            stop = common["stop"]
            payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        # `seed` is accepted in the spec vocabulary but Anthropic has no such
        # parameter; it still participates in the cache key, so dropping it here
        # changes nothing about reproducibility of a replay.
        payload.update(extra)

        headers = {
            "x-api-key": self.api_key(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        started = time.perf_counter()
        data = await post_json(
            self.client,
            f"{self.base_url}/v1/messages",
            headers=headers,
            payload=payload,
            provider=self.name,
            timeout=self.endpoint.limits.timeout_s if self.endpoint.limits else 120.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        content = data.get("content")
        if not isinstance(content, list):
            raise TransientError(f"anthropic response had no content list: {str(data)[:400]}")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=join_text_blocks(content),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            finish_reason=data.get("stop_reason"),
            model=data.get("model"),
            response_id=data.get("id"),
            latency_ms=latency_ms,
        )
