"""Content-addressed response cache backed by SQLite.

What it buys you
----------------
Evaluation is re-run constantly, and almost always for a reason that has
nothing to do with the model: you fixed a scorer, added a normalizer, tightened
a rubric, or want the same numbers in a different format. Re-generating in those
cases costs money and, worse, *changes the answers* -- so a scorer fix and a
sampling shift land in the same diff and you cannot tell which moved the metric.

Caching on the request makes re-scoring free and, more importantly, exact. The
key covers everything that determines the model's output (see
``ChatRequest.cache_key``) and nothing about how the output is graded.

Why SQLite and not a directory of JSON files
--------------------------------------------
One file to copy or delete, atomic writes under concurrency, and a real index
so ``cache stats`` does not stat 200k files. WAL mode lets a second process
read the cache while a run is writing to it.

Staleness
---------
``max_age_s`` exists because model aliases move. ``claude-sonnet-5`` or
``gpt-4o`` point at different weights over time while the cache key stays
identical, so an unbounded cache will happily serve you last quarter's model
and call it today's result. Pin exact versioned model ids when you can, and set
a TTL when you cannot.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CACHE_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key            TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    role           TEXT NOT NULL,
    payload        TEXT NOT NULL,
    created_at     REAL NOT NULL,
    hits           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_responses_created ON responses (created_at);
CREATE INDEX IF NOT EXISTS idx_responses_model   ON responses (provider, model);
"""


@dataclass
class CacheStats:
    entries: int = 0
    hits: int = 0
    misses: int = 0
    writes: int = 0
    stale_drops: int = 0

    @property
    def hit_rate(self) -> float:
        looked_up = self.hits + self.misses
        return self.hits / looked_up if looked_up else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "stale_drops": self.stale_drops,
            "hit_rate": round(self.hit_rate, 4),
        }


class ResponseCache:
    """Thread-safe key/value store for provider responses.

    Calls are short (sub-millisecond for a keyed lookup), so they run inline on
    the event loop under a lock rather than in a thread pool -- a pool would
    add more latency than the query costs.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        read: bool = True,
        write: bool = True,
        max_age_s: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.read_enabled = read
        self.write_enabled = write
        self.max_age_s = max_age_s
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lookup ------------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.read_enabled:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, created_at, schema_version FROM responses WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.stats.misses += 1
                return None
            payload, created_at, schema_version = row
            if schema_version != CACHE_SCHEMA_VERSION:
                self.stats.misses += 1
                self.stats.stale_drops += 1
                return None
            if self.max_age_s is not None and (time.time() - created_at) > self.max_age_s:
                self.stats.misses += 1
                self.stats.stale_drops += 1
                return None
            self._conn.execute("UPDATE responses SET hits = hits + 1 WHERE key = ?", (key,))
            self._conn.commit()
            self.stats.hits += 1
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            # A corrupt row should degrade to a cache miss, never kill the run.
            self.stats.stale_drops += 1
            return None
        return decoded if isinstance(decoded, dict) else None

    def put(self, key: str, value: Mapping[str, Any], *, provider: str, model: str,
            role: str = "candidate") -> None:
        if not self.write_enabled:
            return
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, schema_version, provider, model, role, payload, created_at, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT hits FROM responses WHERE key = ?), 0))",
                (key, CACHE_SCHEMA_VERSION, provider, model, role, payload, time.time(), key),
            )
            self._conn.commit()
            self.stats.writes += 1

    # -- maintenance -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), MIN(created_at), MAX(created_at), COALESCE(SUM(hits), 0) "
                "FROM responses"
            ).fetchone()
            by_model = self._conn.execute(
                "SELECT provider, model, COUNT(*) FROM responses "
                "GROUP BY provider, model ORDER BY COUNT(*) DESC LIMIT 20"
            ).fetchall()
        count, oldest, newest, total_hits = row
        size_bytes = self.path.stat().st_size if self.path.exists() else 0
        return {
            "path": str(self.path),
            "entries": int(count or 0),
            "size_bytes": size_bytes,
            "oldest": oldest,
            "newest": newest,
            "lifetime_hits": int(total_hits or 0),
            "by_model": [
                {"provider": p, "model": m, "entries": c} for p, m, c in by_model
            ],
        }

    def clear(self, *, older_than_s: float | None = None) -> int:
        with self._lock:
            if older_than_s is None:
                cursor = self._conn.execute("DELETE FROM responses")
            else:
                cutoff = time.time() - older_than_s
                cursor = self._conn.execute("DELETE FROM responses WHERE created_at < ?", (cutoff,))
            self._conn.commit()
            removed = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
            self._conn.execute("VACUUM")
            self._conn.commit()
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NullCache(ResponseCache):
    """Drop-in no-op for ``--no-cache``.

    A separate type rather than an ``if cache is not None`` at every call site:
    the runner should not have to remember which mode it is in.
    """

    def __init__(self) -> None:  # noqa: D107 - deliberately does not open a file
        self.path = Path(":memory:")
        self.read_enabled = False
        self.write_enabled = False
        self.max_age_s = None
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._conn = None  # type: ignore[assignment]

    def get(self, key: str) -> dict[str, Any] | None:
        self.stats.misses += 1
        return None

    def put(self, key: str, value: Mapping[str, Any], **kwargs: Any) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {"path": "(disabled)", "entries": 0, "size_bytes": 0, "by_model": []}

    def clear(self, *, older_than_s: float | None = None) -> int:
        return 0

    def close(self) -> None:
        return None


def default_cache_path(root: str | Path = ".") -> Path:
    return Path(root) / ".evalctl" / "cache.sqlite"
