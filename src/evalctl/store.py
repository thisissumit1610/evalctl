"""Run artifacts on disk.

Layout::

    runs/20260825-174233-demo-a1b2c3/
      manifest.json   provenance: suite fingerprint, models, limits, git sha
      records.jsonl   one line per trial, appended as it completes
      summary.json    aggregates, written at the end

Why JSONL, appended and flushed per trial
-----------------------------------------
Long runs get interrupted -- a laptop sleeps, a token expires, someone hits
Ctrl-C after watching the cost tick up. Appending one self-describing line per
trial means an interrupted run is still a valid, readable dataset, and
``--resume`` is just "read the keys already present and skip them". The
alternative -- accumulating in memory and writing at the end -- turns every
interruption into a total loss of a run you already paid for.

Why the manifest is separate from the records
----------------------------------------------
The manifest answers "can I trust a comparison between these two runs?" -- the
suite fingerprint, the exact model params, the harness version, the git sha.
``evalctl diff`` reads it and refuses to be quiet when the two runs measured
different things.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import __version__
from .errors import RunNotFound
from .spec import Suite
from .util import git_revision, utc_now_iso

MANIFEST_NAME = "manifest.json"
RECORDS_NAME = "records.jsonl"
SUMMARY_NAME = "summary.json"
DEFAULT_RUNS_DIR = "runs"

TrialKey = tuple[str, str, str, int]


@dataclass
class TrialRecord:
    """One (model, task, case, sample) outcome."""

    run_id: str
    model_id: str
    task_id: str
    case_id: str
    sample_index: int
    status: str = "ok"  # ok | error
    response_text: str = ""
    score: float | None = None
    passed: bool | None = None
    components: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    attempts: int = 1
    finish_reason: str | None = None
    cache_key: str = ""
    reported_model: str | None = None
    tags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now_iso)

    @property
    def key(self) -> TrialKey:
        return (self.model_id, self.task_id, self.case_id, self.sample_index)

    @property
    def unit(self) -> str:
        """Case identity, stable across runs -- what a paired diff joins on."""
        return f"{self.task_id}/{self.case_id}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TrialRecord":
        known = {f for f in TrialRecord.__dataclass_fields__}  # type: ignore[attr-defined]
        # Ignore unknown fields rather than crashing: a record written by a
        # newer version should still be readable by an older report.
        return TrialRecord(**{k: v for k, v in data.items() if k in known})


class RunStore:
    """Append-only writer / reader for one run directory."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.run_id = self.dir.name
        self._handle = None

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, root: str | Path, run_id: str) -> "RunStore":
        directory = Path(root) / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory)

    @classmethod
    def open(cls, selector: str, root: str | Path = DEFAULT_RUNS_DIR) -> "RunStore":
        """Resolve a run by id, unique prefix, path, or the word ``latest``."""
        as_path = Path(selector)
        if (as_path / MANIFEST_NAME).exists() or (as_path / RECORDS_NAME).exists():
            return cls(as_path)

        root_path = Path(root)
        candidates = list_runs(root_path)
        if not candidates:
            raise RunNotFound(
                f"no runs found under '{root_path}'. Run `evalctl run <suite.yaml>` first."
            )
        if selector in {"latest", "last"}:
            return cls(candidates[0])

        exact = [c for c in candidates if c.name == selector]
        if exact:
            return cls(exact[0])
        prefixed = [c for c in candidates if c.name.startswith(selector)]
        if len(prefixed) == 1:
            return cls(prefixed[0])
        if len(prefixed) > 1:
            names = ", ".join(c.name for c in prefixed[:6])
            raise RunNotFound(f"'{selector}' matches {len(prefixed)} runs: {names} ...")
        recent = ", ".join(c.name for c in candidates[:4])
        raise RunNotFound(f"no run matching '{selector}'. Recent runs: {recent}")

    # -- writing -----------------------------------------------------------

    def append(self, record: TrialRecord) -> None:
        """Write one record and flush it.

        Flushing per trial costs a syscall and buys crash safety on work that
        cost real money to produce. That trade is not close.
        """
        if self._handle is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._handle = (self.dir / RECORDS_NAME).open("a", encoding="utf-8")
        self._handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / name
        # Write-then-rename so a reader never sees a half-written manifest.
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temp, path)
        return path

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        return self.write_json(MANIFEST_NAME, manifest)

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        return self.write_json(SUMMARY_NAME, summary)

    # -- reading -----------------------------------------------------------

    @property
    def records_path(self) -> Path:
        return self.dir / RECORDS_NAME

    def records(self) -> Iterator[TrialRecord]:
        path = self.records_path
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a hard kill. Everything before it
                    # is still good, so skip and keep going.
                    continue
                if isinstance(data, dict):
                    yield TrialRecord.from_dict(data)

    def all_records(self) -> list[TrialRecord]:
        return list(self.records())

    def completed_keys(self) -> set[TrialKey]:
        """Trial keys already on disk -- the basis for ``--resume``.

        Errored trials are deliberately *not* counted as complete, so a resume
        retries them. That is almost always what you want: the usual reason a
        run has errors is a transient outage that has since cleared.
        """
        return {r.key for r in self.records() if r.status == "ok"}

    def manifest(self) -> dict[str, Any]:
        path = self.dir / MANIFEST_NAME
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def summary(self) -> dict[str, Any]:
        path = self.dir / SUMMARY_NAME
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def exists(self) -> bool:
        return self.records_path.exists() or (self.dir / MANIFEST_NAME).exists()


def list_runs(root: str | Path = DEFAULT_RUNS_DIR) -> list[Path]:
    """Run directories, newest first (ids sort chronologically by construction)."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    runs = [
        child
        for child in root_path.iterdir()
        if child.is_dir() and ((child / RECORDS_NAME).exists() or (child / MANIFEST_NAME).exists())
    ]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def build_manifest(
    suite: Suite,
    run_id: str,
    *,
    command: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to judge whether two runs are comparable."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "evalctl_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": git_revision(),
        "command": list(command) if command else [],
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "source": suite.source,
            "fingerprint": suite.fingerprint(),
            "task_patterns": list(suite.task_patterns),
            "repeats": suite.repeats,
            "seed": suite.seed,
            "errors": suite.errors,
            "pass_threshold": suite.pass_threshold,
            "params": dict(suite.params),
        },
        "models": [
            {
                "id": endpoint.id,
                "provider": endpoint.provider,
                "model": endpoint.model,
                "params": suite.params_for(suite.tasks[0], endpoint) if suite.tasks else dict(endpoint.params),
                "base_url": endpoint.base_url,
                "fingerprint": endpoint.fingerprint(),
                "pricing": {
                    "input_per_mtok": endpoint.pricing.input_per_mtok,
                    "output_per_mtok": endpoint.pricing.output_per_mtok,
                },
            }
            for endpoint in suite.models
        ],
        "judge": (
            {
                "id": suite.judge.endpoint.id,
                "provider": suite.judge.endpoint.provider,
                "model": suite.judge.endpoint.model,
                "params": dict(suite.judge.endpoint.params),
                "samples": suite.judge.samples,
                "fingerprint": suite.judge.endpoint.fingerprint(),
            }
            if suite.judge
            else None
        ),
        "limits": {
            "max_concurrency": suite.limits.max_concurrency,
            "requests_per_minute": suite.limits.requests_per_minute,
            "tokens_per_minute": suite.limits.tokens_per_minute,
            "max_retries": suite.limits.max_retries,
            "timeout_s": suite.limits.timeout_s,
        },
        "tasks": [
            {
                "id": task.id,
                "fingerprint": task.fingerprint(),
                "cases": len(task.cases),
                "tags": list(task.tags),
                "scorers": [s.label for s in task.scoring],
                "source": task.source,
            }
            for task in suite.tasks
        ],
        "totals": {
            "models": len(suite.models),
            "tasks": len(suite.tasks),
            "cases": sum(len(t.cases) for t in suite.tasks),
            "planned_trials": suite.total_trials,
        },
    }
    if extra:
        manifest.update(extra)
    return manifest
