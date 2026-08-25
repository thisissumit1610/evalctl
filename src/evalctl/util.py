"""Small shared helpers: canonical hashing, ids, timing, safe truncation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace, non-ASCII preserved.

    Every hash in this project goes through here so that a dict built in a
    different order still produces the same cache key.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, length: int = 12) -> str:
    return sha256_of(obj)[:length]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_id_for(name: str, when: float | None = None) -> str:
    """Human-sortable run id: 20260825-174233-demo-a1b2c3."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime(when if when is not None else time.time()))
    slug = slugify(name) or "run"
    salt = hashlib.sha256(f"{ts}{slug}{os.getpid()}{time.time_ns()}".encode()).hexdigest()[:6]
    return f"{ts}-{slug}-{salt}"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 32) -> str:
    out = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return out[:max_length].strip("-")


def truncate(text: str, limit: int = 400, marker: str = " ...[+{n} chars]") -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + marker.format(n=len(text) - limit)


def git_revision(cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Best-effort git SHA of the working tree, recorded in the run manifest.

    Returns None outside a repo. Never raises -- provenance is nice to have,
    not a reason to fail a run.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return None
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{sha}-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return sha


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def estimate_tokens(text: str) -> int:
    """Cheap pre-flight token estimate for the rate limiter's token bucket.

    Deliberately not a real tokenizer: importing one per provider would cost
    more than the accuracy is worth here. ~4 chars/token overestimates slightly
    on English prose, which is the safe direction for a budget check -- we
    reserve a little more quota than we spend. Actual usage from the API
    response is what gets recorded and billed against.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
