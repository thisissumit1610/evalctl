"""Turning trial records into numbers you can defend.

The one decision that changes every headline number
----------------------------------------------------
A trial can fail for reasons that have nothing to do with the model's ability:
a 500 from the provider, a timeout, a content filter. There are two defensible
ways to handle those, and they disagree:

``zero``     an error counts as a wrong answer. Right when the failure is a
             property of the *case* -- a prompt the provider refuses will fail
             every time, and hiding it flatters the model.
``exclude``  the case is dropped. Right when the failure is a property of the
             *run* -- a rate-limit storm should not be scored as ignorance.

Neither is universally correct, so the harness refuses to pick silently: the
policy comes from the suite, is recorded in the manifest, is printed on every
report, and the error count is always shown next to the score. A benchmark
number whose error policy you cannot see is not a number you can compare.

Aggregation order
-----------------
Repeats are averaged **within a case** before anything else, so a case with
``repeats: 5`` contributes exactly as much to the total as one with a single
sample. The alternative -- pooling every sample -- silently weights cases by how
many times they happened to be run.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from .stats import DEFAULT_LEVEL, DEFAULT_RESAMPLES, Interval, Observation, bootstrap_ci
from .store import TrialRecord

Metric = Literal["score", "pass"]
ClusterBy = Literal["task", "case"]
ErrorPolicy = Literal["zero", "exclude"]


@dataclass(frozen=True)
class CaseOutcome:
    """One case for one model, with its repeats already collapsed."""

    task_id: str
    case_id: str
    model_id: str
    score: float
    passed: float  # a fraction once repeats are averaged, not a bool
    samples: int
    errors: int
    tags: tuple[str, ...] = ()

    @property
    def unit(self) -> str:
        return f"{self.task_id}/{self.case_id}"


def collapse_cases(
    records: Iterable[TrialRecord],
    *,
    model_id: str | None = None,
    error_policy: ErrorPolicy = "zero",
) -> list[CaseOutcome]:
    """Average repeats within each (model, task, case)."""
    buckets: dict[tuple[str, str, str], list[TrialRecord]] = {}
    for record in records:
        if model_id is not None and record.model_id != model_id:
            continue
        buckets.setdefault((record.model_id, record.task_id, record.case_id), []).append(record)

    outcomes: list[CaseOutcome] = []
    for (mid, task_id, case_id), group in buckets.items():
        errors = sum(1 for r in group if r.status != "ok")
        if error_policy == "exclude":
            usable = [r for r in group if r.status == "ok"]
            if not usable:
                # Every repeat failed and the policy says not to score errors,
                # so this case contributes nothing at all.
                continue
            scores = [float(r.score or 0.0) for r in usable]
            passes = [1.0 if r.passed else 0.0 for r in usable]
            samples = len(usable)
        else:
            scores = [float(r.score or 0.0) if r.status == "ok" else 0.0 for r in group]
            passes = [1.0 if (r.status == "ok" and r.passed) else 0.0 for r in group]
            samples = len(group)
        tags: set[str] = set()
        for record in group:
            tags.update(record.tags or ())
        outcomes.append(
            CaseOutcome(
                task_id=task_id,
                case_id=case_id,
                model_id=mid,
                score=sum(scores) / len(scores),
                passed=sum(passes) / len(passes),
                samples=samples,
                errors=errors,
                tags=tuple(sorted(tags)),
            )
        )
    outcomes.sort(key=lambda o: (o.model_id, o.task_id, o.case_id))
    return outcomes


def observations(
    outcomes: Sequence[CaseOutcome],
    metric: Metric = "score",
    cluster_by: ClusterBy = "task",
) -> list[Observation]:
    """Case outcomes as bootstrap observations.

    ``cluster_by`` decides what the interval generalises over, and it is the
    single most consequential knob in the whole package:

    ``task``  (default) resample whole tasks. The claim becomes "on another
              benchmark built like this one", which is what a suite of
              templated cases can actually support. Wider, and honest.
    ``case``  resample individual cases, i.e. assume they are independent.
              Narrower, and what most harnesses do implicitly. Correct only if
              your cases really are unrelated draws.

    Both are offered rather than one being hidden, because the gap between them
    is information: if the task-clustered interval is much wider, your cases are
    correlated and your effective sample size is smaller than your case count.
    """
    return [
        Observation(
            unit=o.unit,
            cluster=o.task_id if cluster_by == "task" else o.unit,
            value=o.score if metric == "score" else o.passed,
        )
        for o in outcomes
    ]


@dataclass(frozen=True)
class TaskStats:
    task_id: str
    cases: int
    score: float
    pass_rate: float
    errors: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cases": self.cases,
            "score": round(self.score, 6),
            "pass_rate": round(self.pass_rate, 6),
            "errors": self.errors,
        }


@dataclass(frozen=True)
class ModelStats:
    model_id: str
    trials: int
    cases: int
    errors: int
    score: Interval
    pass_rate: Interval
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    retries: int = 0
    tasks: tuple[TaskStats, ...] = ()
    by_tag: Mapping[str, float] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.trials if self.trials else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.trials if self.trials else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trials": self.trials,
            "cases": self.cases,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 6),
            "score": self.score.as_dict(),
            "pass_rate": self.pass_rate.as_dict(),
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 6),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "retries": self.retries,
            "tasks": [t.as_dict() for t in self.tasks],
            "by_tag": {k: round(v, 6) for k, v in self.by_tag.items()},
        }


@dataclass(frozen=True)
class RunAnalysis:
    run_id: str
    models: tuple[ModelStats, ...]
    error_policy: str
    level: float
    resamples: int
    seed: int
    cluster_by: str = "task"
    total_trials: int = 0
    total_errors: int = 0
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    notes: tuple[str, ...] = ()

    def model(self, model_id: str) -> ModelStats | None:
        return next((m for m in self.models if m.model_id == model_id), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "error_policy": self.error_policy,
            "confidence_level": self.level,
            "bootstrap_resamples": self.resamples,
            "seed": self.seed,
            "cluster_by": self.cluster_by,
            "total_trials": self.total_trials,
            "total_errors": self.total_errors,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "duration_s": round(self.duration_s, 3),
            "models": [m.as_dict() for m in self.models],
            "notes": list(self.notes),
        }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def analyze_model(
    records: Sequence[TrialRecord],
    model_id: str,
    *,
    error_policy: ErrorPolicy = "zero",
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    cluster_by: ClusterBy = "task",
) -> ModelStats:
    mine = [r for r in records if r.model_id == model_id]
    outcomes = collapse_cases(mine, model_id=model_id, error_policy=error_policy)

    score_ci = bootstrap_ci(
        observations(outcomes, "score", cluster_by), level=level, resamples=resamples, seed=seed
    )
    pass_ci = bootstrap_ci(
        observations(outcomes, "pass", cluster_by), level=level, resamples=resamples, seed=seed + 1
    )

    # Latency is reported over calls that actually went to the network. Mixing
    # in replayed cache hits would report the speed of a hash lookup.
    latencies = [r.latency_ms for r in mine if r.status == "ok" and not r.cached]

    by_task: dict[str, list[CaseOutcome]] = {}
    for outcome in outcomes:
        by_task.setdefault(outcome.task_id, []).append(outcome)
    errors_by_task: dict[str, int] = {}
    for record in mine:
        if record.status != "ok":
            errors_by_task[record.task_id] = errors_by_task.get(record.task_id, 0) + 1

    tasks = tuple(
        TaskStats(
            task_id=task_id,
            cases=len(group),
            score=statistics.fmean(o.score for o in group),
            pass_rate=statistics.fmean(o.passed for o in group),
            errors=errors_by_task.get(task_id, 0),
        )
        for task_id, group in sorted(by_task.items())
    )

    by_tag: dict[str, list[float]] = {}
    for outcome in outcomes:
        for tag in outcome.tags:
            by_tag.setdefault(tag, []).append(outcome.score)

    return ModelStats(
        model_id=model_id,
        trials=len(mine),
        cases=len(outcomes),
        errors=sum(1 for r in mine if r.status != "ok"),
        score=score_ci,
        pass_rate=pass_ci,
        cost_usd=sum(r.cost_usd for r in mine),
        input_tokens=sum(r.input_tokens for r in mine),
        output_tokens=sum(r.output_tokens for r in mine),
        cache_hits=sum(1 for r in mine if r.cached),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        retries=sum(max(0, r.attempts - 1) for r in mine),
        tasks=tasks,
        by_tag={tag: statistics.fmean(values) for tag, values in sorted(by_tag.items())},
    )


def analyze_run(
    records: Sequence[TrialRecord],
    *,
    run_id: str = "",
    error_policy: ErrorPolicy = "zero",
    level: float = DEFAULT_LEVEL,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    cluster_by: ClusterBy = "task",
    duration_s: float = 0.0,
) -> RunAnalysis:
    model_ids = sorted({r.model_id for r in records})
    models = tuple(
        analyze_model(
            records,
            model_id,
            error_policy=error_policy,
            level=level,
            resamples=resamples,
            seed=seed,
            cluster_by=cluster_by,
        )
        for model_id in model_ids
    )
    total_errors = sum(m.errors for m in models)
    notes: list[str] = []
    if total_errors:
        share = total_errors / max(1, len(records))
        notes.append(
            f"{total_errors} trial(s) failed ({share:.1%}); error policy '{error_policy}' "
            f"{'counts them as 0' if error_policy == 'zero' else 'excludes them'}"
        )
        if share > 0.05:
            notes.append(
                "error rate above 5% -- treat these scores as provisional and re-run the "
                "failures with `--resume` before comparing"
            )
    degenerate = [m.model_id for m in models if m.score.method in {"empty", "degenerate"}]
    if degenerate:
        notes.append(
            f"no interval for {', '.join(degenerate)}: a confidence interval needs at least "
            "two tasks to resample between"
        )
    return RunAnalysis(
        run_id=run_id,
        models=models,
        error_policy=error_policy,
        level=level,
        resamples=resamples,
        seed=seed,
        cluster_by=cluster_by,
        total_trials=len(records),
        total_errors=total_errors,
        total_cost_usd=sum(m.cost_usd for m in models),
        duration_s=duration_s,
        notes=tuple(notes),
    )
