"""Provider interface: one normalised request/response shape for every API.

Everything above this layer -- runner, cache, scorers, stats -- speaks only
``ChatRequest``/``ChatResponse``. Adding a provider means implementing one
method and registering a name; nothing else in the codebase learns about it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import httpx

from .. import REQUEST_SCHEMA_VERSION
from ..errors import FatalError, RateLimited, TransientError
from ..spec import Endpoint
from ..util import estimate_tokens, sha256_of

# Sampling knobs this harness understands and maps per provider. Anything else
# in `params` is forwarded verbatim in the provider's own vocabulary, so a
# provider-specific flag stays usable without a code change here.
COMMON_PARAMS = ("temperature", "max_tokens", "top_p", "stop", "seed")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass(frozen=True)
class ChatRequest:
    """One model call, fully determined -- this is what gets hashed for the cache.

    ``sample_index`` is part of the identity on purpose. With ``repeats: 5`` at
    a non-zero temperature you want five *independent draws*, not one answer
    replayed five times; keying on the index gives each draw its own cache slot,
    so a re-run replays a stochastic experiment exactly.

    ``role`` separates candidate calls from judge calls so the two never collide
    in the cache even when the same model serves both.
    """

    model: str
    messages: tuple[tuple[str, str], ...]  # (role, content), oldest first
    system: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    sample_index: int = 0
    role: str = "candidate"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        head = f"{self.system}\n\n" if self.system else ""
        return head + "\n\n".join(f"{role}: {content}" for role, content in self.messages)

    def estimated_input_tokens(self) -> int:
        return estimate_tokens(self.prompt_text)

    def cache_key(self, provider: str, base_url: str | None) -> str:
        """Identity of the *call*, not of the scoring.

        Scorers are deliberately excluded: re-scoring a cached response is free,
        so changing a rubric or a tolerance must not force a re-generation.
        """
        return sha256_of(
            {
                "v": REQUEST_SCHEMA_VERSION,
                "provider": provider,
                "base_url": base_url,
                "model": self.model,
                "system": self.system,
                "messages": list(self.messages),
                "params": dict(self.params),
                "sample_index": self.sample_index,
                "role": self.role,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    model: str | None = None  # as echoed by the API: an alias may resolve elsewhere
    response_id: str | None = None
    latency_ms: float = 0.0
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage.as_dict(),
            "finish_reason": self.finish_reason,
            "model": self.model,
            "response_id": self.response_id,
            "latency_ms": self.latency_ms,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, cached: bool = False) -> "ChatResponse":
        usage = data.get("usage") or {}
        return ChatResponse(
            text=data.get("text", ""),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            finish_reason=data.get("finish_reason"),
            model=data.get("model"),
            response_id=data.get("response_id"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cached=cached,
        )


class Provider(abc.ABC):
    """Base class for model backends.

    Subclasses implement :meth:`complete` and are responsible for turning
    transport failures into the retry taxonomy in ``errors.py``. Getting that
    classification right is most of the value of this layer: the runner retries
    ``TransientError`` and ``RateLimited`` and gives up immediately on
    ``FatalError``, so a mis-classified 400 would burn five retries per case
    across the whole suite.
    """

    name: str = "base"
    requires_api_key: bool = True
    # Whether this provider reads ChatRequest.metadata. Only the offline mock
    # does; keeping it False elsewhere means an answer-key edit never
    # invalidates a cached call whose prompt did not contain the answer.
    uses_metadata: bool = False
    default_api_key_env: str | None = None
    default_base_url: str | None = None

    def __init__(self, endpoint: Endpoint, client: httpx.AsyncClient | None = None) -> None:
        self.endpoint = endpoint
        self.model = endpoint.model
        self.base_url = endpoint.base_url or self.default_base_url
        self._client = client

    # -- lifecycle ---------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                f"provider '{self.name}' has no HTTP client; construct it through "
                "providers.build() or pass one in"
            )
        return self._client

    async def aclose(self) -> None:
        return None

    # -- the one method a provider must implement --------------------------

    @abc.abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Issue one call. Raise RateLimited / TransientError / FatalError."""

    # -- helpers for subclasses -------------------------------------------

    def split_params(self, params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Separate the knobs we normalise from provider-specific passthrough."""
        common = {k: v for k, v in params.items() if k in COMMON_PARAMS}
        extra = {k: v for k, v in params.items() if k not in COMMON_PARAMS}
        return common, extra

    def api_key(self) -> str:
        import os

        env_name = self.endpoint.api_key_env or self.default_api_key_env
        if not self.requires_api_key:
            return ""
        if not env_name:
            raise FatalError(f"provider '{self.name}' needs an api_key_env to be configured")
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise FatalError(
                f"environment variable {env_name} is unset or empty -- needed for model "
                f"'{self.endpoint.id}' ({self.name}/{self.model})"
            )
        return key


# --------------------------------------------------------------------------
# shared HTTP error classification
# --------------------------------------------------------------------------


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Honour whichever cool-down header the provider actually sent.

    Providers disagree: `retry-after` is standard, but several send seconds in
    vendor-specific reset headers instead. Guessing an exponential backoff when
    the server told you the exact number is how a run ends up 10x slower than
    it needs to be -- or 10x too aggressive.
    """
    for header in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = headers.get(header)
        if not raw:
            continue
        text = raw.strip().lower()
        try:
            if text.endswith("ms"):
                return float(text[:-2]) / 1000.0
            if text.endswith("s"):
                return float(text[:-1])
            return float(text)
        except ValueError:
            continue  # HTTP-date form; fall back to the caller's backoff
    return None


def raise_for_status(response: httpx.Response, provider: str) -> None:
    """Map an HTTP status onto the retry taxonomy."""
    if response.status_code < 400:
        return
    body = response.text[:2000]
    status = response.status_code
    message = f"{provider} returned HTTP {status}: {body}"
    if status == 429:
        raise RateLimited(message, retry_after=parse_retry_after(response.headers), status=status, body=body)
    if status in (408, 409, 425, 500, 502, 503, 504, 529):
        raise TransientError(message, status=status, body=body)
    raise FatalError(message, status=status, body=body)


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    provider: str,
    timeout: float,
) -> dict[str, Any]:
    """POST with the transport exceptions folded into our taxonomy."""
    try:
        response = await client.post(url, headers=dict(headers), json=dict(payload), timeout=timeout)
    except httpx.TimeoutException as exc:
        raise TransientError(f"{provider} request timed out after {timeout}s") from exc
    except httpx.HTTPError as exc:
        # Connection resets, DNS blips, broken pipes: all worth one more try.
        raise TransientError(f"{provider} transport error: {exc}") from exc
    raise_for_status(response, provider)
    try:
        data = response.json()
    except ValueError as exc:
        raise TransientError(f"{provider} returned non-JSON body: {response.text[:500]}") from exc
    if not isinstance(data, dict):
        raise TransientError(f"{provider} returned a non-object JSON body: {type(data).__name__}")
    return data


def join_text_blocks(blocks: Sequence[Any]) -> str:
    """Flatten a content-block list (Anthropic style) into plain text."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
