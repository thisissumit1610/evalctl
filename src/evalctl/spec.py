"""YAML spec loading and validation.

Two document kinds:

  * a **task file** -- a prompt template, a scoring stack, and a list of cases;
  * a **suite file** -- which tasks to run, against which models, under which
    limits, with which judge.

Design notes
------------
*Unknown keys are errors.* A spec loader that ignores ``temperture: 0`` will
happily run your whole benchmark at temperature 1.0 and report the number with
a straight face. Every mapping here is checked against a closed key set, and the
error names the file and the dotted path.

*Everything is frozen.* Specs are hashed into cache keys and run manifests; a
mutable spec is a spec that can drift between the hash and the request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import SpecError
from .templating import variables_in, variables_in_deep
from .util import sha256_of

# --------------------------------------------------------------------------
# typed accessors -- every one reports file + dotted path on failure
# --------------------------------------------------------------------------


def _fail(msg: str, path: str, source: str) -> SpecError:
    return SpecError(msg, path=path, source=source)


def _check_keys(data: Mapping[str, Any], allowed: Iterable[str], path: str, source: str) -> None:
    allowed_set = set(allowed)
    unknown = [k for k in data if k not in allowed_set]
    if unknown:
        raise _fail(
            f"unknown key(s): {', '.join(sorted(unknown))}. "
            f"Allowed here: {', '.join(sorted(allowed_set))}",
            path,
            source,
        )


def _as_mapping(value: Any, path: str, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _fail(f"expected a mapping, got {type(value).__name__}", path, source)
    for key in value:
        if not isinstance(key, str):
            raise _fail(f"mapping keys must be strings, got {key!r}", path, source)
    return dict(value)


def _as_list(value: Any, path: str, source: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise _fail(f"expected a list, got {type(value).__name__}", path, source)
    return list(value)


def _as_str(value: Any, path: str, source: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(f"expected a string, got {type(value).__name__}", path, source)
    if not allow_empty and not value.strip():
        raise _fail("must not be empty", path, source)
    return value


def _as_float(
    value: Any, path: str, source: str, *, lo: float | None = None, hi: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"expected a number, got {type(value).__name__}", path, source)
    out = float(value)
    if lo is not None and out < lo:
        raise _fail(f"must be >= {lo}, got {out}", path, source)
    if hi is not None and out > hi:
        raise _fail(f"must be <= {hi}, got {out}", path, source)
    return out


def _as_int(
    value: Any, path: str, source: str, *, lo: int | None = None, hi: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"expected an integer, got {type(value).__name__}", path, source)
    if lo is not None and value < lo:
        raise _fail(f"must be >= {lo}, got {value}", path, source)
    if hi is not None and value > hi:
        raise _fail(f"must be <= {hi}, got {value}", path, source)
    return value


def _as_bool(value: Any, path: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(f"expected true/false, got {type(value).__name__}", path, source)
    return value


def _as_tags(value: Any, path: str, source: str) -> tuple[str, ...]:
    items = _as_list(value, path, source)
    return tuple(_as_str(t, f"{path}[{i}]", source) for i, t in enumerate(items))


def _load_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read file: {exc}", source=str(path)) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid YAML: {exc}", source=str(path)) from exc


# --------------------------------------------------------------------------
# task specs
# --------------------------------------------------------------------------

_SCORER_META_KEYS = {"type", "weight", "required", "name"}


@dataclass(frozen=True)
class ScorerSpec:
    """One entry in a task's ``scoring:`` list.

    ``weight`` sets its share of the task's aggregate score. ``required``
    decides whether it can veto a pass: a rubric judge is often scored but not
    required, so a low style score lowers the number without flipping the
    pass/fail bit that a regression gate keys on.
    """

    type: str
    weight: float = 1.0
    required: bool = True
    name: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.type

    @staticmethod
    def parse(data: Any, path: str, source: str) -> "ScorerSpec":
        if isinstance(data, str):  # shorthand:  - exact_match
            return ScorerSpec(type=data)
        mapping = _as_mapping(data, path, source)
        if "type" not in mapping:
            raise _fail("scorer needs a 'type'", path, source)
        scorer_type = _as_str(mapping["type"], f"{path}.type", source)
        name = mapping.get("name")
        if name is not None:
            name = _as_str(name, f"{path}.name", source)
        return ScorerSpec(
            type=scorer_type,
            weight=_as_float(mapping.get("weight", 1.0), f"{path}.weight", source, lo=0.0),
            required=_as_bool(mapping.get("required", True), f"{path}.required", source),
            name=name,
            config={k: v for k, v in mapping.items() if k not in _SCORER_META_KEYS},
        )


@dataclass(frozen=True)
class Case:
    """One concrete instance of a task: the variables that fill the template."""

    id: str
    vars: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    scoring: tuple[ScorerSpec, ...] | None = None  # overrides the task's stack

    @staticmethod
    def parse(data: Any, index: int, path: str, source: str) -> "Case":
        mapping = _as_mapping(data, path, source)
        _check_keys(mapping, {"id", "vars", "tags", "scoring"}, path, source)
        scoring: tuple[ScorerSpec, ...] | None = None
        if "scoring" in mapping:
            entries = _as_list(mapping["scoring"], f"{path}.scoring", source)
            scoring = tuple(
                ScorerSpec.parse(s, f"{path}.scoring[{i}]", source) for i, s in enumerate(entries)
            )
        return Case(
            id=_as_str(mapping.get("id", f"case-{index:03d}"), f"{path}.id", source),
            vars=_as_mapping(mapping.get("vars"), f"{path}.vars", source),
            tags=_as_tags(mapping.get("tags"), f"{path}.tags", source),
            scoring=scoring,
        )


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Task:
    """A prompt template plus a scoring stack, instantiated over ``cases``."""

    id: str
    cases: tuple[Case, ...]
    scoring: tuple[ScorerSpec, ...]
    system: str | None = None
    messages: tuple[Message, ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    pass_threshold: float | None = None
    source: str = ""

    def fingerprint(self) -> str:
        """Content hash of everything that defines what this task asks.

        Two runs whose task fingerprints differ are not comparable, and
        ``evalctl diff`` says so rather than quietly averaging apples with
        oranges.
        """
        return sha256_of(
            {
                "id": self.id,
                "system": self.system,
                "messages": [(m.role, m.content) for m in self.messages],
                "params": dict(self.params),
                "pass_threshold": self.pass_threshold,
                "scoring": [
                    {
                        "type": s.type,
                        "weight": s.weight,
                        "required": s.required,
                        "name": s.name,
                        "config": dict(s.config),
                    }
                    for s in self.scoring
                ],
                "cases": [
                    {"id": c.id, "vars": dict(c.vars), "tags": list(c.tags)} for c in self.cases
                ],
            }
        )

    def scoring_for(self, case: Case) -> tuple[ScorerSpec, ...]:
        return case.scoring if case.scoring is not None else self.scoring


_TASK_KEYS = {"id", "description", "tags", "prompt", "params", "scoring", "cases", "pass_threshold"}
_PROMPT_KEYS = {"system", "user", "messages"}


def _parse_prompt(data: Any, path: str, source: str) -> tuple[str | None, tuple[Message, ...]]:
    mapping = _as_mapping(data, path, source)
    _check_keys(mapping, _PROMPT_KEYS, path, source)
    system = mapping.get("system")
    if system is not None:
        system = _as_str(system, f"{path}.system", source)
    has_user, has_messages = "user" in mapping, "messages" in mapping
    if has_user and has_messages:
        raise _fail("set either 'user' or 'messages', not both", path, source)
    if not has_user and not has_messages:
        raise _fail("needs a 'user' template or a 'messages' list", path, source)
    if has_user:
        return system, (Message("user", _as_str(mapping["user"], f"{path}.user", source)),)

    raw = _as_list(mapping["messages"], f"{path}.messages", source)
    if not raw:
        raise _fail("'messages' must not be empty", f"{path}.messages", source)
    messages: list[Message] = []
    for i, item in enumerate(raw):
        item_path = f"{path}.messages[{i}]"
        item_map = _as_mapping(item, item_path, source)
        _check_keys(item_map, {"role", "content"}, item_path, source)
        role = _as_str(item_map.get("role", ""), f"{item_path}.role", source)
        if role not in {"user", "assistant"}:
            raise _fail(
                "role must be 'user' or 'assistant' (the system prompt goes in prompt.system), "
                f"got '{role}'",
                f"{item_path}.role",
                source,
            )
        content = _as_str(item_map.get("content", ""), f"{item_path}.content", source)
        messages.append(Message(role, content))
    if messages[-1].role != "user":
        raise _fail("the last message must have role 'user'", f"{path}.messages", source)
    return system, tuple(messages)


def parse_task(data: Any, source: str, path: str = "") -> Task:
    prefix = f"{path}." if path else ""
    mapping = _as_mapping(data, path or "<root>", source)
    _check_keys(mapping, _TASK_KEYS, path or "<root>", source)
    if "id" not in mapping:
        raise _fail("task needs an 'id'", path or "<root>", source)
    if "prompt" not in mapping:
        raise _fail("task needs a 'prompt'", path or "<root>", source)

    system, messages = _parse_prompt(mapping["prompt"], f"{prefix}prompt", source)
    scoring_entries = _as_list(mapping.get("scoring"), f"{prefix}scoring", source)
    scoring = tuple(
        ScorerSpec.parse(s, f"{prefix}scoring[{i}]", source) for i, s in enumerate(scoring_entries)
    )

    raw_cases = _as_list(mapping.get("cases"), f"{prefix}cases", source)
    if not raw_cases:
        raise _fail("task needs at least one entry under 'cases'", f"{prefix}cases", source)
    cases = tuple(Case.parse(c, i, f"{prefix}cases[{i}]", source) for i, c in enumerate(raw_cases))
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise _fail(f"duplicate case id '{case.id}'", f"{prefix}cases", source)
        seen.add(case.id)

    threshold = mapping.get("pass_threshold")
    if threshold is not None:
        threshold = _as_float(threshold, f"{prefix}pass_threshold", source, lo=0.0, hi=1.0)

    task = Task(
        id=_as_str(mapping["id"], f"{prefix}id", source),
        cases=cases,
        scoring=scoring,
        system=system,
        messages=messages,
        description=_as_str(
            mapping.get("description", ""), f"{prefix}description", source, allow_empty=True
        ),
        tags=_as_tags(mapping.get("tags"), f"{prefix}tags", source),
        params=_as_mapping(mapping.get("params"), f"{prefix}params", source),
        pass_threshold=threshold,
        source=source,
    )
    _check_task_templates(task, source)
    return task


def _check_task_templates(task: Task, source: str) -> None:
    """Fail fast on template variables no case supplies.

    The highest-value validation in this file: without it a typo in
    ``{{ quesiton }}`` surfaces only after you have paid for the whole run.
    """
    needed: set[str] = set()
    if task.system:
        needed |= variables_in(task.system)
    for message in task.messages:
        needed |= variables_in(message.content)
    for case in task.cases:
        case_needed = set(needed)
        for scorer in task.scoring_for(case):
            case_needed |= variables_in_deep(scorer.config)
        missing = sorted(case_needed - set(case.vars))
        if missing:
            available = ", ".join(sorted(map(str, case.vars))) or "(none)"
            raise _fail(
                f"case '{case.id}' is missing var(s) {', '.join(missing)} used by the task; "
                f"it defines: {available}",
                f"task '{task.id}'",
                source,
            )


def load_tasks_from_file(path: Path) -> list[Task]:
    """A task file holds one task mapping, a list of them, or ``{tasks: [...]}``."""
    data = _load_yaml(path)
    source = str(path)
    if data is None:
        raise SpecError("file is empty", source=source)
    if isinstance(data, Mapping):
        if "tasks" in data and "id" not in data:
            _check_keys(dict(data), {"tasks"}, "<root>", source)
            entries = _as_list(data["tasks"], "tasks", source)
            return [parse_task(t, source, f"tasks[{i}]") for i, t in enumerate(entries)]
        return [parse_task(data, source)]
    entries = _as_list(data, "<root>", source)
    return [parse_task(t, source, f"[{i}]") for i, t in enumerate(entries)]


# --------------------------------------------------------------------------
# suite specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens; drives cost reporting and ``--dry-run`` estimates."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000

    @staticmethod
    def parse(data: Any, path: str, source: str) -> "Pricing":
        mapping = _as_mapping(data, path, source)
        _check_keys(mapping, {"input_per_mtok", "output_per_mtok"}, path, source)
        return Pricing(
            input_per_mtok=_as_float(
                mapping.get("input_per_mtok", 0.0), f"{path}.input_per_mtok", source, lo=0.0
            ),
            output_per_mtok=_as_float(
                mapping.get("output_per_mtok", 0.0), f"{path}.output_per_mtok", source, lo=0.0
            ),
        )


@dataclass(frozen=True)
class Limits:
    """Client-side throttling.

    Defaults are conservative on purpose: the cost of being too slow is
    minutes, the cost of being too fast is a 429 storm and a half-finished run.
    """

    max_concurrency: int = 8
    requests_per_minute: float = 60.0
    tokens_per_minute: float = 60_000.0
    max_retries: int = 5
    timeout_s: float = 120.0
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0

    @staticmethod
    def parse(data: Any, path: str, source: str, base: "Limits | None" = None) -> "Limits":
        base = base or Limits()
        mapping = _as_mapping(data, path, source)
        _check_keys(
            mapping,
            {
                "max_concurrency",
                "requests_per_minute",
                "tokens_per_minute",
                "max_retries",
                "timeout_s",
                "initial_backoff_s",
                "max_backoff_s",
            },
            path,
            source,
        )
        get = mapping.get
        return Limits(
            max_concurrency=_as_int(
                get("max_concurrency", base.max_concurrency), f"{path}.max_concurrency", source, lo=1
            ),
            requests_per_minute=_as_float(
                get("requests_per_minute", base.requests_per_minute),
                f"{path}.requests_per_minute",
                source,
                lo=0.0,
            ),
            tokens_per_minute=_as_float(
                get("tokens_per_minute", base.tokens_per_minute),
                f"{path}.tokens_per_minute",
                source,
                lo=0.0,
            ),
            max_retries=_as_int(get("max_retries", base.max_retries), f"{path}.max_retries", source, lo=0),
            timeout_s=_as_float(get("timeout_s", base.timeout_s), f"{path}.timeout_s", source, lo=1.0),
            initial_backoff_s=_as_float(
                get("initial_backoff_s", base.initial_backoff_s),
                f"{path}.initial_backoff_s",
                source,
                lo=0.0,
            ),
            max_backoff_s=_as_float(
                get("max_backoff_s", base.max_backoff_s), f"{path}.max_backoff_s", source, lo=0.0
            ),
        )


_ENDPOINT_KEYS = {"id", "provider", "model", "params", "base_url", "api_key_env", "pricing", "limits"}


@dataclass(frozen=True)
class Endpoint:
    """A callable model configuration.

    Used both for models under test and for the judge, because they need
    exactly the same knobs -- and because the judge deserves the same rate
    limiting, caching and cost accounting as anything else.
    """

    id: str
    provider: str
    model: str
    params: Mapping[str, Any] = field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    pricing: Pricing = field(default_factory=Pricing)
    limits: Limits | None = None

    def fingerprint(self) -> str:
        return sha256_of(
            {
                "provider": self.provider,
                "model": self.model,
                "params": dict(self.params),
                "base_url": self.base_url,
            }
        )

    @staticmethod
    def parse(
        data: Any,
        path: str,
        source: str,
        *,
        default_id: str | None = None,
        base_limits: "Limits | None" = None,
    ) -> "Endpoint":
        mapping = _as_mapping(data, path, source)
        _check_keys(mapping, _ENDPOINT_KEYS, path, source)
        for required in ("provider", "model"):
            if required not in mapping:
                raise _fail(f"needs '{required}'", path, source)
        model = _as_str(mapping["model"], f"{path}.model", source)
        base_url = mapping.get("base_url")
        if base_url is not None:
            base_url = _as_str(base_url, f"{path}.base_url", source).rstrip("/")
        api_key_env = mapping.get("api_key_env")
        if api_key_env is not None:
            api_key_env = _as_str(api_key_env, f"{path}.api_key_env", source)
        limits = None
        if "limits" in mapping:
            limits = Limits.parse(mapping["limits"], f"{path}.limits", source, base=base_limits)
        return Endpoint(
            id=_as_str(mapping.get("id", default_id or model), f"{path}.id", source),
            provider=_as_str(mapping["provider"], f"{path}.provider", source),
            model=model,
            params=_as_mapping(mapping.get("params"), f"{path}.params", source),
            base_url=base_url,
            api_key_env=api_key_env,
            pricing=Pricing.parse(mapping.get("pricing"), f"{path}.pricing", source),
            limits=limits,
        )


@dataclass(frozen=True)
class JudgeConfig:
    """The judge is an endpoint plus how many times to poll it.

    ``samples > 1`` runs the rubric several times and takes the median per
    criterion -- the cheapest available defence against a judge that is noisy
    near its own decision boundary. See docs/design.md, "The judge".
    """

    endpoint: Endpoint
    samples: int = 1

    @staticmethod
    def parse(data: Any, path: str, source: str, base_limits: Limits | None = None) -> "JudgeConfig":
        mapping = _as_mapping(data, path, source)
        _check_keys(mapping, _ENDPOINT_KEYS | {"samples"}, path, source)
        samples = _as_int(mapping.get("samples", 1), f"{path}.samples", source, lo=1, hi=15)
        if samples % 2 == 0:
            raise _fail(
                "judge.samples should be odd so the median is a single observed score, "
                f"got {samples}",
                f"{path}.samples",
                source,
            )
        endpoint = Endpoint.parse(
            {k: v for k, v in mapping.items() if k != "samples"},
            path,
            source,
            default_id="judge",
            base_limits=base_limits,
        )
        return JudgeConfig(endpoint=endpoint, samples=samples)


_SUITE_KEYS = {
    "name",
    "description",
    "tasks",
    "models",
    "repeats",
    "seed",
    "limits",
    "judge",
    "errors",
    "cache",
    "params",
    "pass_threshold",
}
ERROR_POLICIES = ("zero", "exclude")


@dataclass(frozen=True)
class Suite:
    name: str
    models: tuple[Endpoint, ...]
    tasks: tuple[Task, ...]
    description: str = ""
    repeats: int = 1
    seed: int = 0
    limits: Limits = field(default_factory=Limits)
    judge: JudgeConfig | None = None
    errors: str = "zero"
    cache: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)
    pass_threshold: float | None = None
    source: str = ""
    task_patterns: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        """Identity of *what was measured* -- tasks and cases only.

        Deliberately excludes models, limits and repeats: swapping the model is
        the entire point of a diff, and re-running with more repeats or higher
        concurrency does not change what the benchmark asks.
        """
        return sha256_of(sorted(t.fingerprint() for t in self.tasks))

    def params_for(self, task: Task, endpoint: Endpoint) -> dict[str, Any]:
        """Sampling params, most specific wins: suite < task < model."""
        merged: dict[str, Any] = dict(self.params)
        merged.update(task.params)
        merged.update(endpoint.params)
        return merged

    def threshold_for(self, task: Task) -> float | None:
        return task.pass_threshold if task.pass_threshold is not None else self.pass_threshold

    def with_tasks(self, tasks: Sequence[Task]) -> "Suite":
        return replace(self, tasks=tuple(tasks))

    def with_models(self, models: Sequence[Endpoint]) -> "Suite":
        return replace(self, models=tuple(models))

    @property
    def total_trials(self) -> int:
        cases = sum(len(t.cases) for t in self.tasks)
        return cases * self.repeats * len(self.models)


def _resolve_task_patterns(patterns: Sequence[str], base_dir: Path, source: str) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for i, pattern in enumerate(patterns):
        candidate = Path(pattern)
        if not candidate.is_absolute():
            candidate = base_dir / pattern
        if candidate.is_file():
            matches = [candidate]
        elif candidate.is_dir():
            matches = sorted(p for p in candidate.rglob("*") if p.suffix in {".yaml", ".yml"})
        else:
            # Path.glob wants a pattern relative to a root, so split at the
            # first magic component: `examples/tasks/*.yaml` then works from
            # any cwd, resolved relative to the suite file.
            parts = Path(pattern).parts
            magic = next((j for j, p in enumerate(parts) if any(c in p for c in "*?[")), None)
            if magic is None:
                raise _fail(f"no task file at '{pattern}'", f"tasks[{i}]", source)
            root = base_dir.joinpath(*parts[:magic]) if magic else base_dir
            matches = sorted(root.glob(str(Path(*parts[magic:]))))
        if not matches:
            raise _fail(f"pattern '{pattern}' matched no files", f"tasks[{i}]", source)
        for match in matches:
            key = match.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(match)
    return resolved


def load_suite(path: str | os.PathLike[str]) -> Suite:
    suite_path = Path(path)
    source = str(suite_path)
    data = _load_yaml(suite_path)
    if not isinstance(data, Mapping):
        raise SpecError("a suite file must be a mapping at the top level", source=source)
    mapping = dict(data)
    _check_keys(mapping, _SUITE_KEYS, "<root>", source)

    limits = Limits.parse(mapping.get("limits"), "limits", source)

    raw_models = _as_list(mapping.get("models"), "models", source)
    if not raw_models:
        raise _fail("suite needs at least one entry under 'models'", "models", source)
    models = tuple(
        Endpoint.parse(m, f"models[{i}]", source, base_limits=limits)
        for i, m in enumerate(raw_models)
    )
    model_ids = [m.id for m in models]
    duplicates = {mid for mid in model_ids if model_ids.count(mid) > 1}
    if duplicates:
        raise _fail(
            f"duplicate model id(s): {', '.join(sorted(duplicates))}. Model ids label every row "
            "in a report and both sides of a diff, so they have to be unique.",
            "models",
            source,
        )

    patterns = tuple(
        _as_str(p, f"tasks[{i}]", source)
        for i, p in enumerate(_as_list(mapping.get("tasks"), "tasks", source))
    )
    if not patterns:
        raise _fail("suite needs at least one entry under 'tasks'", "tasks", source)
    tasks: list[Task] = []
    for task_file in _resolve_task_patterns(patterns, suite_path.parent, source):
        tasks.extend(load_tasks_from_file(task_file))
    if not tasks:
        raise _fail("the task patterns resolved to zero tasks", "tasks", source)
    task_ids = [t.id for t in tasks]
    dupe_tasks = {tid for tid in task_ids if task_ids.count(tid) > 1}
    if dupe_tasks:
        raise _fail(
            f"duplicate task id(s) across files: {', '.join(sorted(dupe_tasks))}", "tasks", source
        )

    errors = _as_str(mapping.get("errors", "zero"), "errors", source)
    if errors not in ERROR_POLICIES:
        raise _fail(f"must be one of {ERROR_POLICIES}, got '{errors}'", "errors", source)

    judge = None
    if mapping.get("judge") is not None:
        judge = JudgeConfig.parse(mapping["judge"], "judge", source, base_limits=limits)

    threshold = mapping.get("pass_threshold")
    if threshold is not None:
        threshold = _as_float(threshold, "pass_threshold", source, lo=0.0, hi=1.0)

    suite = Suite(
        name=_as_str(mapping.get("name", suite_path.stem), "name", source),
        description=_as_str(mapping.get("description", ""), "description", source, allow_empty=True),
        models=models,
        tasks=tuple(tasks),
        repeats=_as_int(mapping.get("repeats", 1), "repeats", source, lo=1, hi=100),
        seed=_as_int(mapping.get("seed", 0), "seed", source, lo=0),
        limits=limits,
        judge=judge,
        errors=errors,
        cache=_as_bool(mapping.get("cache", True), "cache", source),
        params=_as_mapping(mapping.get("params"), "params", source),
        pass_threshold=threshold,
        source=source,
        task_patterns=patterns,
    )
    _check_judge_configured(suite)
    return suite


def _check_judge_configured(suite: Suite) -> None:
    if suite.judge is not None:
        return
    for task in suite.tasks:
        for case in task.cases:
            for scorer in task.scoring_for(case):
                if scorer.type == "llm_judge":
                    raise _fail(
                        f"task '{task.id}' uses the llm_judge scorer but the suite defines no "
                        "'judge:' endpoint",
                        "judge",
                        suite.source,
                    )
