"""Run orchestration: fan out trials, respect limits, score, persist.

Shape of a run
--------------
The unit of work is a **trial**: one (model, task, case, sample) tuple. Trials
are independent, so the runner is a bounded worker pool over a queue rather
than anything cleverer. Concurrency is bounded per *endpoint* by the rate
limiter, which is where the real constraint lives -- two models on different
providers should not throttle each other.

Failure policy
--------------
A trial that fails after its retries is recorded with ``status="error"`` and the
run continues. Aborting the whole run on one bad case would throw away every
result already paid for, and errors are usually a property of a case (a prompt
that trips a filter) rather than of the run.

How errors count toward the score is *not* decided here. That is
``suite.errors`` (``zero`` or ``exclude``), applied at analysis time -- because
the choice materially changes the headline number and belongs next to the
number, not buried in the execution path.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import providers as provider_registry
from .cache import NullCache, ResponseCache
from .errors import EvalctlError, FatalError, ProviderError, RateLimited, ScoringError, TransientError
from .providers.base import ChatRequest, ChatResponse, Provider
from .ratelimit import LimiterRegistry, RateLimiter
from .scorers import ScoringContext, aggregate, build_for_case
from .scorers.base import ScoredComponent
from .spec import Case, Endpoint, JudgeConfig, Suite, Task
from .store import RunStore, TrialRecord
from .templating import render
from .util import truncate

ProgressFn = Callable[["RunProgress"], None]


@dataclass(frozen=True)
class Trial:
    model: Endpoint
    task: Task
    case: Case
    sample_index: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.model.id, self.task.id, self.case.id, self.sample_index)


@dataclass
class RunProgress:
    total: int
    done: int = 0
    ok: int = 0
    errors: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    last: TrialRecord | None = None

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def rate_per_s(self) -> float:
        return self.done / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def eta_s(self) -> float:
        remaining = self.total - self.done
        return remaining / self.rate_per_s if self.rate_per_s > 0 else 0.0


@dataclass
class RunResult:
    run_id: str
    store: RunStore
    records: list[TrialRecord]
    progress: RunProgress
    limiter_stats: Mapping[str, Mapping[str, float | int]] = field(default_factory=dict)
    cache_stats: Mapping[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


# --------------------------------------------------------------------------
# suite filtering (used by CLI flags)
# --------------------------------------------------------------------------


def filter_suite(
    suite: Suite,
    *,
    tasks: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    limit: int | None = None,
    repeats: int | None = None,
) -> Suite:
    """Narrow a suite for a partial run.

    ``limit`` truncates cases *per task* rather than globally, so a smoke run
    keeps coverage across every task instead of exhausting the first one.
    """
    selected_tasks = list(suite.tasks)
    if tasks:
        selected_tasks = [
            t for t in selected_tasks if any(fnmatch.fnmatch(t.id, pattern) for pattern in tasks)
        ]
        if not selected_tasks:
            raise FatalError(
                f"--task {', '.join(tasks)} matched no tasks. Available: "
                f"{', '.join(t.id for t in suite.tasks)}"
            )
    if tags:
        wanted = set(tags)
        selected_tasks = [
            t
            for t in selected_tasks
            if wanted & (set(t.tags) | {tag for c in t.cases for tag in c.tags})
        ]
        if not selected_tasks:
            raise FatalError(f"--tag {', '.join(tags)} matched no tasks")
    if limit is not None:
        selected_tasks = [replace(t, cases=t.cases[:limit]) for t in selected_tasks]

    selected_models = list(suite.models)
    if models:
        selected_models = [
            m for m in selected_models if any(fnmatch.fnmatch(m.id, pattern) for pattern in models)
        ]
        if not selected_models:
            raise FatalError(
                f"--model {', '.join(models)} matched nothing. Available: "
                f"{', '.join(m.id for m in suite.models)}"
            )

    out = suite.with_tasks(selected_tasks).with_models(selected_models)
    if repeats is not None:
        out = replace(out, repeats=repeats)
    return out


def build_trials(suite: Suite) -> list[Trial]:
    return [
        Trial(model=model, task=task, case=case, sample_index=index)
        for model in suite.models
        for task in suite.tasks
        for case in task.cases
        for index in range(suite.repeats)
    ]


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------


def render_request(
    suite: Suite,
    task: Task,
    case: Case,
    endpoint: Endpoint,
    sample_index: int,
    *,
    include_metadata: bool = False,
) -> ChatRequest:
    where = f"{task.id}/{case.id}"
    system = render(task.system, case.vars, where=f"{where}.system") if task.system else None
    messages = tuple(
        (m.role, render(m.content, case.vars, where=f"{where}.messages[{i}]"))
        for i, m in enumerate(task.messages)
    )
    metadata: dict[str, Any] = {}
    if include_metadata:
        # Only the offline mock reads this. It is kept out of every real
        # provider's request so that editing an answer key cannot invalidate a
        # cache entry for a call whose prompt never contained the answer.
        expected = expected_answer(task, case)
        if expected is not None:
            metadata["expected"] = expected
    return ChatRequest(
        model=endpoint.model,
        messages=messages,
        system=system,
        params=suite.params_for(task, endpoint),
        sample_index=sample_index,
        role="candidate",
        metadata=metadata,
    )


def expected_answer(task: Task, case: Case) -> str | None:
    """Best-effort answer key, read from the first scorer that declares one."""
    for spec in task.scoring_for(case):
        for field_name in ("target", "reference"):
            raw = spec.config.get(field_name)
            if isinstance(raw, str):
                try:
                    return render(raw, case.vars, where=f"{task.id}/{case.id}.{field_name}")
                except EvalctlError:
                    return raw
            if isinstance(raw, (int, float)):
                return str(raw)
    return None


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------


class JudgeService:
    """Runs judge calls for scorers, through the same cache and limiter.

    The judge gets no special treatment: it is rate limited, cached and costed
    exactly like a model under test. An uncached judge is the usual reason a
    "cheap" eval suite turns out to cost more than the generations did.
    """

    def __init__(
        self,
        config: JudgeConfig,
        provider: Provider,
        limiter: RateLimiter,
        cache: ResponseCache,
        recorder: "CostRecorder",
    ) -> None:
        self.config = config
        self.samples = config.samples
        self.provider = provider
        self.limiter = limiter
        self.cache = cache
        self.recorder = recorder

    async def __call__(
        self, *, system: str | None, user: str, sample_index: int, cache_tag: str
    ) -> str:
        request = ChatRequest(
            model=self.config.endpoint.model,
            messages=(("user", user),),
            system=system,
            params=dict(self.config.endpoint.params),
            sample_index=sample_index,
            role="judge",
        )
        response, _ = await call_with_retries(
            self.provider, request, self.limiter, self.cache, self.config.endpoint
        )
        self.recorder.record(self.config.endpoint, response)
        return response.text


@dataclass
class CostRecorder:
    """Running totals for calls that are not tied to a single trial (the judge)."""

    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    def record(self, endpoint: Endpoint, response: ChatResponse) -> None:
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if response.cached:
            self.cache_hits += 1
        else:
            self.cost_usd += endpoint.pricing.cost(
                response.usage.input_tokens, response.usage.output_tokens
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


# --------------------------------------------------------------------------
# the call path: cache -> limiter -> provider, with retries
# --------------------------------------------------------------------------


async def call_with_retries(
    provider: Provider,
    request: ChatRequest,
    limiter: RateLimiter,
    cache: ResponseCache,
    endpoint: Endpoint,
) -> tuple[ChatResponse, int]:
    """Return (response, attempts). Raises the last error if all attempts fail."""
    key = request.cache_key(provider.name, provider.base_url)
    hit = cache.get(key)
    if hit is not None:
        return ChatResponse.from_dict(hit, cached=True), 0

    limits = endpoint.limits or limiter.limits
    estimated = request.estimated_input_tokens()
    last_error: Exception | None = None

    for attempt in range(1, limits.max_retries + 2):
        try:
            async with limiter.slot(estimated):
                response = await provider.complete(request)
        except RateLimited as exc:
            last_error = exc
            # Pause every worker on this endpoint, not just this one -- see
            # ratelimit.RateLimiter.pause_for.
            limiter.pause_for(limiter.backoff_delay(attempt, exc.retry_after))
            continue
        except TransientError as exc:
            last_error = exc
            if attempt > limits.max_retries:
                break
            await asyncio.sleep(limiter.backoff_delay(attempt))
            continue
        except FatalError:
            raise

        cache.put(
            key,
            response.as_dict(),
            provider=provider.name,
            model=endpoint.model,
            role=request.role,
        )
        return response, attempt

    assert last_error is not None
    raise last_error


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        suite: Suite,
        store: RunStore,
        *,
        cache: ResponseCache | None = None,
        resume: bool = False,
        progress: ProgressFn | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.suite = suite
        self.store = store
        self.cache = cache if cache is not None else NullCache()
        self.resume = resume
        self.progress_fn = progress
        self.max_workers = max_workers
        self.limiters = LimiterRegistry(default_limits=suite.limits)
        self.judge_costs = CostRecorder()
        self._providers: dict[str, Provider] = {}
        self._client = None

    # -- setup -------------------------------------------------------------

    def _limiter_for(self, endpoint: Endpoint) -> RateLimiter:
        # Keyed by provider+model, not by endpoint id: two entries for the same
        # model with different sampling params share one server-side quota, so
        # they must share one limiter.
        key = f"{endpoint.provider}:{endpoint.model}:{endpoint.base_url or ''}"
        return self.limiters.for_endpoint(key, endpoint.limits or self.suite.limits)

    def _provider_for(self, endpoint: Endpoint) -> Provider:
        if endpoint.id not in self._providers:
            self._providers[endpoint.id] = provider_registry.build(endpoint, self._client)
        return self._providers[endpoint.id]

    async def run(self) -> RunResult:
        started = time.monotonic()
        trials = build_trials(self.suite)
        skipped: set[tuple[str, str, str, int]] = set()
        if self.resume:
            skipped = self.store.completed_keys()
            trials = [t for t in trials if t.key not in skipped]

        progress = RunProgress(total=len(trials))
        records: list[TrialRecord] = []

        if not trials:
            return RunResult(
                run_id=self.store.run_id,
                store=self.store,
                records=records,
                progress=progress,
                cache_stats=self.cache.stats.as_dict(),
                duration_s=time.monotonic() - started,
            )

        endpoints = list(self.suite.models)
        if self.suite.judge:
            endpoints.append(self.suite.judge.endpoint)
        for endpoint in endpoints:
            provider_registry.preflight(endpoint)

        needs_http = any(provider_registry.needs_http(e) for e in endpoints)
        self._client = provider_registry.make_client() if needs_http else None

        judge: JudgeService | None = None
        if self.suite.judge:
            judge = JudgeService(
                self.suite.judge,
                self._provider_for(self.suite.judge.endpoint),
                self._limiter_for(self.suite.judge.endpoint),
                self.cache,
                self.judge_costs,
            )

        queue: asyncio.Queue[Trial | None] = asyncio.Queue()
        for trial in trials:
            queue.put_nowait(trial)

        workers = self.max_workers or min(
            256, sum((m.limits or self.suite.limits).max_concurrency for m in self.suite.models)
        )
        workers = max(1, min(workers, len(trials)))
        write_lock = asyncio.Lock()

        async def worker() -> None:
            while True:
                try:
                    trial = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await self._run_trial(trial, judge)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # last-resort net: never lose the pool
                    record = self._error_record(trial, exc, attempts=0)
                async with write_lock:
                    self.store.append(record)
                    records.append(record)
                    progress.done += 1
                    progress.last = record
                    if record.status == "ok":
                        progress.ok += 1
                    else:
                        progress.errors += 1
                    if record.cached:
                        progress.cache_hits += 1
                    progress.cost_usd += record.cost_usd
                    if self.progress_fn is not None:
                        self.progress_fn(progress)
                queue.task_done()

        try:
            await asyncio.gather(*(worker() for _ in range(workers)))
        finally:
            await self._shutdown()

        return RunResult(
            run_id=self.store.run_id,
            store=self.store,
            records=records,
            progress=progress,
            limiter_stats=self.limiters.stats(),
            cache_stats=self.cache.stats.as_dict(),
            duration_s=time.monotonic() - started,
        )

    async def _shutdown(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- one trial ---------------------------------------------------------

    async def _run_trial(self, trial: Trial, judge: JudgeService | None) -> TrialRecord:
        provider = self._provider_for(trial.model)
        limiter = self._limiter_for(trial.model)
        request = render_request(
            self.suite,
            trial.task,
            trial.case,
            trial.model,
            trial.sample_index,
            include_metadata=getattr(provider, "uses_metadata", False),
        )

        try:
            response, attempts = await call_with_retries(
                provider, request, limiter, self.cache, trial.model
            )
        except ProviderError as exc:
            return self._error_record(trial, exc, attempts=self.suite.limits.max_retries + 1)

        try:
            verdict_components = await self._score(trial, request, response, judge)
        except ScoringError as exc:
            record = self._error_record(trial, exc, attempts=1, error_type="scoring")
            record.response_text = response.text
            record.cached = response.cached
            record.input_tokens = response.usage.input_tokens
            record.output_tokens = response.usage.output_tokens
            return record

        verdict = aggregate(verdict_components, self.suite.threshold_for(trial.task))
        cost = 0.0 if response.cached else trial.model.pricing.cost(
            response.usage.input_tokens, response.usage.output_tokens
        )
        return TrialRecord(
            run_id=self.store.run_id,
            model_id=trial.model.id,
            task_id=trial.task.id,
            case_id=trial.case.id,
            sample_index=trial.sample_index,
            status="ok",
            response_text=response.text,
            score=verdict.score,
            passed=verdict.passed,
            components=[c.as_dict() for c in verdict.components],
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            latency_ms=response.latency_ms,
            cached=response.cached,
            attempts=attempts,
            finish_reason=response.finish_reason,
            cache_key=request.cache_key(provider.name, provider.base_url),
            reported_model=response.model,
            tags=sorted(set(trial.task.tags) | set(trial.case.tags)),
        )

    async def _score(
        self,
        trial: Trial,
        request: ChatRequest,
        response: ChatResponse,
        judge: JudgeService | None,
    ) -> list[ScoredComponent]:
        scorers = build_for_case(trial.task, trial.case)
        context = ScoringContext(
            task=trial.task,
            case=trial.case,
            endpoint=trial.model,
            sample_index=trial.sample_index,
            prompt_text=request.prompt_text,
            vars=trial.case.vars,
            judge=judge,
        )
        components: list[ScoredComponent] = []
        for scorer in scorers:
            score = await scorer.score(response.text, context)
            components.append(
                ScoredComponent(
                    name=scorer.spec.label,
                    type=scorer.type,
                    value=score.value,
                    passed=score.passed,
                    weight=scorer.spec.weight,
                    required=scorer.spec.required,
                    detail=score.detail,
                )
            )
        return components

    def _error_record(
        self, trial: Trial, exc: Exception, *, attempts: int, error_type: str | None = None
    ) -> TrialRecord:
        return TrialRecord(
            run_id=self.store.run_id,
            model_id=trial.model.id,
            task_id=trial.task.id,
            case_id=trial.case.id,
            sample_index=trial.sample_index,
            status="error",
            error=truncate(str(exc), 800),
            error_type=error_type or type(exc).__name__,
            attempts=attempts,
            tags=sorted(set(trial.task.tags) | set(trial.case.tags)),
        )


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------


@dataclass
class DryRunEstimate:
    model_id: str
    trials: int
    input_tokens: int
    projected_output_tokens: int
    cost_usd: float
    cached_trials: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trials": self.trials,
            "input_tokens": self.input_tokens,
            "projected_output_tokens": self.projected_output_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "cached_trials": self.cached_trials,
        }


def dry_run(
    suite: Suite, *, cache: ResponseCache | None = None, assumed_output_tokens: int | None = None
) -> tuple[list[DryRunEstimate], list[tuple[Trial, ChatRequest]]]:
    """Render every prompt and project the bill without calling anything.

    Output length is the one thing that genuinely cannot be known in advance, so
    it is assumed to be ``max_tokens`` unless overridden -- a deliberate
    over-estimate, since a budget surprise should only ever go downward. Cache
    hits are subtracted, which is what makes the estimate useful on a re-run.
    """
    estimates: dict[str, DryRunEstimate] = {}
    rendered: list[tuple[Trial, ChatRequest]] = []
    for trial in build_trials(suite):
        _, resolved = provider_registry.resolve_endpoint(trial.model)
        request = render_request(
            suite, trial.task, trial.case, trial.model, trial.sample_index, include_metadata=False
        )
        rendered.append((trial, request))
        estimate = estimates.setdefault(trial.model.id, DryRunEstimate(trial.model.id, 0, 0, 0, 0.0))
        estimate.trials += 1

        cached = False
        if cache is not None:
            provider_cls, _ = provider_registry.resolve_endpoint(trial.model)
            key = request.cache_key(provider_cls.name, resolved.base_url or provider_cls.default_base_url)
            cached = cache.get(key) is not None
        if cached:
            estimate.cached_trials += 1
            continue

        params = suite.params_for(trial.task, trial.model)
        out_tokens = assumed_output_tokens or int(params.get("max_tokens", 512))
        in_tokens = request.estimated_input_tokens()
        estimate.input_tokens += in_tokens
        estimate.projected_output_tokens += out_tokens
        estimate.cost_usd += trial.model.pricing.cost(in_tokens, out_tokens)
    return list(estimates.values()), rendered


async def run_suite(
    suite: Suite,
    store: RunStore,
    *,
    cache: ResponseCache | None = None,
    resume: bool = False,
    progress: ProgressFn | None = None,
) -> RunResult:
    runner = Runner(suite, store, cache=cache, resume=resume, progress=progress)
    return await runner.run()


def iter_case_units(suite: Suite) -> Iterable[str]:
    for task in suite.tasks:
        for case in task.cases:
            yield f"{task.id}/{case.id}"
