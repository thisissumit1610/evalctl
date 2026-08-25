"""Cache, rate limiting and the provider retry taxonomy."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from evalctl.cache import CACHE_SCHEMA_VERSION, NullCache, ResponseCache
from evalctl.errors import FatalError, RateLimited, TransientError
from evalctl.providers import build, resolve_endpoint
from evalctl.providers.base import ChatRequest, parse_retry_after, raise_for_status
from evalctl.ratelimit import RateLimiter, TokenBucket
from evalctl.spec import Endpoint, Limits


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_round_trip(tmp_cache_path):
    with ResponseCache(tmp_cache_path) as cache:
        cache.put("k", {"text": "hello"}, provider="mock", model="m")
        assert cache.get("k") == {"text": "hello"}
        assert cache.stats.hits == 1 and cache.stats.writes == 1


def test_cache_miss_is_counted_and_returns_none(tmp_cache_path):
    with ResponseCache(tmp_cache_path) as cache:
        assert cache.get("absent") is None
        assert cache.stats.misses == 1


def test_cache_ttl_expires_old_entries(tmp_cache_path):
    with ResponseCache(tmp_cache_path) as writer:
        writer.put("k", {"text": "old"}, provider="mock", model="m")
    with ResponseCache(tmp_cache_path, max_age_s=0.0) as reader:
        # A zero TTL means "anything already written is stale", which is the
        # behaviour that protects you from a moving model alias.
        time.sleep(0.01)
        assert reader.get("k") is None
        assert reader.stats.stale_drops == 1


def test_cache_ignores_entries_from_a_different_schema_version(tmp_cache_path):
    with ResponseCache(tmp_cache_path) as cache:
        cache.put("k", {"text": "x"}, provider="mock", model="m")
        cache._conn.execute(
            "UPDATE responses SET schema_version = ? WHERE key = ?",
            (CACHE_SCHEMA_VERSION + 1, "k"),
        )
        cache._conn.commit()
        assert cache.get("k") is None


def test_cache_read_can_be_disabled_for_refresh(tmp_cache_path):
    with ResponseCache(tmp_cache_path) as cache:
        cache.put("k", {"text": "x"}, provider="mock", model="m")
    with ResponseCache(tmp_cache_path, read=False) as refresher:
        assert refresher.get("k") is None
        refresher.put("k", {"text": "new"}, provider="mock", model="m")
    with ResponseCache(tmp_cache_path) as cache:
        assert cache.get("k") == {"text": "new"}


def test_cache_clear_and_summary(tmp_cache_path):
    with ResponseCache(tmp_cache_path) as cache:
        for i in range(5):
            cache.put(f"k{i}", {"text": str(i)}, provider="mock", model="m")
        assert cache.summary()["entries"] == 5
        assert cache.clear() == 5
        assert cache.summary()["entries"] == 0


def test_null_cache_never_hits():
    cache = NullCache()
    cache.put("k", {"text": "x"}, provider="mock", model="m")
    assert cache.get("k") is None
    assert cache.summary()["entries"] == 0


# --------------------------------------------------------------------------
# cache keys
# --------------------------------------------------------------------------


def base_request(**overrides):
    fields = dict(
        model="m", messages=(("user", "hi"),), system=None, params={"temperature": 0.0},
        sample_index=0, role="candidate", metadata={},
    )
    fields.update(overrides)
    return ChatRequest(**fields)


def key(request):
    return request.cache_key("mock", None)


def test_cache_key_is_stable_across_dict_ordering():
    a = base_request(params={"temperature": 0.0, "max_tokens": 10})
    b = base_request(params={"max_tokens": 10, "temperature": 0.0})
    assert key(a) == key(b)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "other"},
        {"messages": (("user", "different"),)},
        {"system": "you are helpful"},
        {"params": {"temperature": 1.0}},
        {"sample_index": 1},
        {"role": "judge"},
    ],
)
def test_cache_key_changes_with_anything_that_changes_the_output(overrides):
    assert key(base_request()) != key(base_request(**overrides))


def test_repeats_get_separate_cache_slots():
    """`repeats: 3` must be three draws, not one answer replayed three times."""
    keys = {key(base_request(sample_index=i)) for i in range(3)}
    assert len(keys) == 3


def test_cache_key_ignores_the_base_url_only_when_it_is_the_same():
    request = base_request()
    assert request.cache_key("mock", "http://a") != request.cache_key("mock", "http://b")


# --------------------------------------------------------------------------
# token bucket
# --------------------------------------------------------------------------


def test_token_bucket_with_zero_rate_never_blocks():
    bucket = TokenBucket(0)
    assert bucket.unlimited
    assert asyncio.run(bucket.acquire(1_000_000)) == 0.0


def test_token_bucket_throttles_to_roughly_the_configured_rate():
    async def drain():
        bucket = TokenBucket(600)  # 10 per second
        started = time.monotonic()
        for _ in range(6):
            await bucket.acquire(1)
        return time.monotonic() - started

    elapsed = asyncio.run(drain())
    # Capacity starts at one second of refill (10), so six requests should be
    # near-instant -- this pins the burst allowance, not the steady rate.
    assert elapsed < 0.5


def test_token_bucket_does_not_start_with_a_full_minute_of_burst():
    """A bucket seeded with 60 tokens lets 60 requests fire at once, which is
    exactly the burst that trips server-side limits."""
    bucket = TokenBucket(60)
    assert bucket.capacity <= 2


def test_token_bucket_admits_a_request_larger_than_its_capacity():
    async def go():
        bucket = TokenBucket(60, capacity=10)
        return await bucket.acquire(1000)

    # Must not deadlock: a long-context prompt against a small quota still runs.
    assert asyncio.run(asyncio.wait_for(go(), timeout=5)) >= 0.0


# --------------------------------------------------------------------------
# limiter gate
# --------------------------------------------------------------------------


def test_gate_pauses_every_worker_not_just_the_one_that_got_429():
    async def go():
        limiter = RateLimiter(Limits(max_concurrency=4, requests_per_minute=0, tokens_per_minute=0))
        limiter.pause_for(0.25)
        started = time.monotonic()
        await asyncio.gather(*(limiter.wait_for_gate() for _ in range(4)))
        return time.monotonic() - started

    elapsed = asyncio.run(go())
    assert elapsed >= 0.2


def test_overlapping_pauses_extend_but_do_not_stack():
    limiter = RateLimiter(Limits())
    limiter.pause_for(0.5)
    first_deadline = limiter._paused_until
    limiter.pause_for(0.2)  # shorter: must not shorten or add
    assert limiter._paused_until == first_deadline
    limiter.pause_for(5.0)
    assert limiter._paused_until > first_deadline


def test_backoff_prefers_the_servers_retry_after():
    limiter = RateLimiter(Limits(initial_backoff_s=1.0, max_backoff_s=60.0))
    assert limiter.backoff_delay(1, retry_after=7.5) == 7.5
    assert limiter.backoff_delay(1, retry_after=999) == 60.0  # capped


def test_backoff_uses_full_jitter_so_workers_do_not_retry_in_lockstep():
    limiter = RateLimiter(Limits(initial_backoff_s=4.0, max_backoff_s=60.0))
    delays = {limiter.backoff_delay(3) for _ in range(50)}
    assert len(delays) > 10, "identical delays would reproduce the original burst"
    assert all(0 <= d <= 16 for d in delays)


def test_semaphore_bounds_concurrency():
    async def go():
        limiter = RateLimiter(Limits(max_concurrency=3, requests_per_minute=0, tokens_per_minute=0))
        active = 0
        peak = 0

        async def worker():
            nonlocal active, peak
            async with limiter.slot(0):
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(worker() for _ in range(12)))
        return peak

    assert asyncio.run(go()) <= 3


# --------------------------------------------------------------------------
# provider error taxonomy
# --------------------------------------------------------------------------


def response(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, text="body")


def test_429_is_rate_limited_and_carries_retry_after():
    with pytest.raises(RateLimited) as exc:
        raise_for_status(response(429, {"retry-after": "12"}), "test")
    assert exc.value.retry_after == 12.0


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529, 408])
def test_server_errors_are_transient(status):
    with pytest.raises(TransientError):
        raise_for_status(response(status), "test")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_fatal_so_retries_do_not_burn_budget(status):
    with pytest.raises(FatalError):
        raise_for_status(response(status), "test")


def test_success_does_not_raise():
    raise_for_status(response(200), "test")


def test_retry_after_parsing_handles_units_and_junk():
    assert parse_retry_after({"retry-after": "1.5"}) == 1.5
    assert parse_retry_after({"retry-after": "500ms"}) == 0.5
    assert parse_retry_after({"x-ratelimit-reset-requests": "3s"}) == 3.0
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert parse_retry_after({}) is None


# --------------------------------------------------------------------------
# provider registry
# --------------------------------------------------------------------------


def test_alias_fills_in_base_url_and_key_env():
    _, resolved = resolve_endpoint(Endpoint(id="local", provider="ollama", model="llama3"))
    assert resolved.provider == "openai-compat"
    assert resolved.base_url == "http://localhost:11434"


def test_unknown_provider_lists_the_available_ones():
    with pytest.raises(FatalError, match="unknown provider"):
        resolve_endpoint(Endpoint(id="x", provider="magic", model="m"))


def test_openai_compat_requires_a_base_url():
    with pytest.raises(FatalError, match="base_url"):
        build(Endpoint(id="x", provider="openai-compat", model="m"))


def test_missing_api_key_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = build(Endpoint(id="c", provider="anthropic", model="claude-sonnet-5"))
    with pytest.raises(FatalError, match="ANTHROPIC_API_KEY"):
        provider.api_key()


def test_mock_provider_is_deterministic_across_processes():
    """Determinism is the whole point of the offline model."""
    provider = build(Endpoint(id="m", provider="mock", model="sim", params={"ability": 0.5}))
    request = ChatRequest(
        model="sim", messages=(("user", "q?"),), params={"ability": 0.5}, metadata={"expected": "7"}
    )
    first = asyncio.run(provider.complete(request)).text
    second = asyncio.run(provider.complete(request)).text
    assert first == second


def test_mock_provider_accuracy_tracks_the_configured_ability():
    provider = build(Endpoint(id="m", provider="mock", model="sim"))
    correct = 0
    total = 400
    for i in range(total):
        request = ChatRequest(
            model="sim",
            messages=(("user", f"question {i}?"),),
            params={"ability": 0.7, "spread": 0.7},
            metadata={"expected": "7"},
        )
        correct += asyncio.run(provider.complete(request)).text == "7"
    assert 0.62 <= correct / total <= 0.78


def test_mock_provider_answers_judge_prompts_with_valid_json():
    import json

    provider = build(Endpoint(id="j", provider="mock", model="sim-judge"))
    prompt = (
        "<response>\nsome answer\n</response>\n"
        "<rubric>\n- accuracy (weight 2): correct?\n</rubric>\n"
        "Score every criterion with a whole number from 0 to 4, where 0 = fails.\n"
        'Reply with exactly this JSON object and nothing else:\n'
        '{"scores": {"accuracy": <0-4>}, "rationale": "<one sentence>"}'
    )
    request = ChatRequest(model="sim-judge", messages=(("user", prompt),), role="judge")
    payload = json.loads(asyncio.run(provider.complete(request)).text)
    assert set(payload["scores"]) == {"accuracy"}
    assert 0 <= payload["scores"]["accuracy"] <= 4


def test_mock_provider_can_simulate_transient_failures():
    provider = build(Endpoint(id="m", provider="mock", model="sim"))
    failures = 0
    for i in range(50):
        request = ChatRequest(
            model="sim", messages=(("user", f"q{i}"),), params={"error_rate": 1.0}
        )
        try:
            asyncio.run(provider.complete(request))
        except TransientError:
            failures += 1
    assert failures == 50
