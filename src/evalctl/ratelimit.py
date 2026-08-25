"""Client-side rate limiting: token buckets, a concurrency cap, and a shared
backoff gate.

Three mechanisms, because provider limits come in three shapes
--------------------------------------------------------------
* **requests/minute** -- a token bucket at request granularity.
* **tokens/minute** -- a second bucket charged with the *estimated* prompt size
  before the call. This is the limit that actually bites on long-context
  benchmarks, and the one most naive harnesses ignore until a run dies at 80%.
* **max concurrency** -- a semaphore. Buckets bound the average rate; they do
  not stop 200 coroutines from firing in the same millisecond after a refill.

The gate is the part people skip
--------------------------------
When a 429 comes back, backing off *only the coroutine that got it* is close to
useless: the other N-1 workers keep hammering the endpoint and keep getting
429s, so the backoff never converges and you look like an attacker. On a 429
this limiter pauses **every** worker until the same deadline, preferring the
server's own ``Retry-After`` over a guess. That single decision is the
difference between a run that degrades gracefully under load and one that
collapses.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from .spec import Limits


@dataclass
class LimiterStats:
    requests: int = 0
    throttle_wait_s: float = 0.0
    gate_wait_s: float = 0.0
    rate_limit_events: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "throttle_wait_s": round(self.throttle_wait_s, 3),
            "gate_wait_s": round(self.gate_wait_s, 3),
            "rate_limit_events": self.rate_limit_events,
        }


class TokenBucket:
    """Standard leaky bucket. A rate of 0 means "no limit" and never blocks.

    Capacity defaults to one second of refill (with a floor of 1) instead of a
    full minute: a bucket that starts full at 60/min lets the first 60 requests
    fire instantly, which is exactly the burst that trips server-side limits
    even though the *average* rate is legal.
    """

    def __init__(self, rate_per_minute: float, capacity: float | None = None) -> None:
        self.rate_per_second = max(0.0, rate_per_minute) / 60.0
        if capacity is not None:
            self.capacity = max(1.0, capacity)
        else:
            self.capacity = max(1.0, self.rate_per_second)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def unlimited(self) -> bool:
        return self.rate_per_second <= 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)

    async def acquire(self, amount: float = 1.0) -> float:
        """Block until `amount` tokens are available. Returns seconds waited."""
        if self.unlimited or amount <= 0:
            return 0.0
        # A single request larger than the bucket would deadlock, so let it
        # through after one full drain rather than waiting forever. This shows
        # up with long-context prompts against a low tokens/minute quota.
        amount = min(amount, self.capacity)
        waited = 0.0
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                deficit = amount - self._tokens
                delay = deficit / self.rate_per_second
            delay = min(delay, 60.0)
            await asyncio.sleep(delay)
            waited += delay


class RateLimiter:
    """Composite limiter for one endpoint."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.requests = TokenBucket(limits.requests_per_minute)
        self.tokens = TokenBucket(limits.tokens_per_minute)
        self.semaphore = asyncio.Semaphore(limits.max_concurrency)
        self.stats = LimiterStats()
        self._paused_until = 0.0
        self._pause_lock = asyncio.Lock()

    # -- shared backoff gate ----------------------------------------------

    def pause_for(self, seconds: float) -> None:
        """Hold every worker on this endpoint for at least `seconds`.

        Deadlines only ever extend, so overlapping 429s from several in-flight
        requests do not stack into a multi-minute stall.
        """
        if seconds <= 0:
            return
        deadline = time.monotonic() + seconds
        if deadline > self._paused_until:
            self._paused_until = deadline
            self.stats.rate_limit_events += 1

    async def wait_for_gate(self) -> float:
        waited = 0.0
        while True:
            remaining = self._paused_until - time.monotonic()
            if remaining <= 0:
                self.stats.gate_wait_s += waited
                return waited
            await asyncio.sleep(min(remaining, 5.0))
            waited += min(remaining, 5.0)

    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with full jitter, capped by the suite's limits.

        Full jitter (uniform in [0, computed]) rather than a fixed delay: N
        workers that back off by exactly the same amount retry in exactly the
        same instant and reproduce the burst that caused the 429.
        """
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.limits.max_backoff_s)
        base = self.limits.initial_backoff_s * (2 ** max(0, attempt - 1))
        return random.uniform(0, min(base, self.limits.max_backoff_s))

    # -- the acquire path --------------------------------------------------

    @contextlib.asynccontextmanager
    async def slot(self, estimated_tokens: int = 0) -> AsyncIterator[None]:
        """Reserve capacity for one request.

        Order matters: gate first (do not consume bucket capacity while the
        endpoint is telling us to stop), then the buckets, then the semaphore.
        Taking the semaphore last keeps a coroutine from occupying a
        concurrency slot for the whole time it sits waiting on a refill.
        """
        await self.wait_for_gate()
        waited = await self.requests.acquire(1.0)
        waited += await self.tokens.acquire(float(estimated_tokens))
        self.stats.throttle_wait_s += waited
        async with self.semaphore:
            await self.wait_for_gate()  # a 429 may have landed while we queued
            self.stats.requests += 1
            yield


@dataclass
class LimiterRegistry:
    """One limiter per endpoint fingerprint.

    Per-endpoint rather than global: two models on different providers do not
    share a quota, and throttling them together would halve throughput for no
    reason. Two entries for the same provider *and* model do share one limiter,
    which is what you want when a suite lists the same model twice with
    different sampling params.
    """

    default_limits: Limits
    _limiters: dict[str, RateLimiter] = field(default_factory=dict)

    def for_endpoint(self, key: str, limits: Limits | None = None) -> RateLimiter:
        limiter = self._limiters.get(key)
        if limiter is None:
            limiter = RateLimiter(limits or self.default_limits)
            self._limiters[key] = limiter
        return limiter

    def stats(self) -> dict[str, dict[str, float | int]]:
        return {key: limiter.stats.as_dict() for key, limiter in self._limiters.items()}
