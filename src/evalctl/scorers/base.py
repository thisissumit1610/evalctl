"""Scorer interface and score aggregation.

Contract
--------
Every scorer returns a value in ``[0, 1]`` **and** an independent ``passed``
bool. Keeping them separate matters: a rubric judge can give 0.72 without that
meaning "pass", and an exact match gives 1.0 or 0.0 where the bool is the only
honest summary. Collapsing the two into a threshold at the scorer level would
throw away the distinction before anything can use it.

Aggregation is stated in one place (:func:`aggregate`) so a task's headline
number is never the private convention of whichever scorer ran last.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ..errors import ScoringError
from ..spec import Case, Endpoint, ScorerSpec, Task


@dataclass(frozen=True)
class Score:
    """What a single scorer decided."""

    value: float
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ScoringError(f"scorer returned {self.value}, which is outside [0, 1]")


class JudgeRunner(Protocol):
    """What a scorer may ask of the judge model.

    Narrow on purpose: a scorer gets to send a prompt and read text back. It
    cannot reach the cache, the limiter or the run store, so adding a scorer
    can never quietly change how the harness spends money.
    """

    samples: int

    async def __call__(
        self, *, system: str | None, user: str, sample_index: int, cache_tag: str
    ) -> str: ...


@dataclass(frozen=True)
class ScoringContext:
    """Everything a scorer is allowed to see about the trial it is grading."""

    task: Task
    case: Case
    endpoint: Endpoint
    sample_index: int
    prompt_text: str
    vars: Mapping[str, Any] = field(default_factory=dict)
    judge: JudgeRunner | None = None

    @property
    def trial_tag(self) -> str:
        return f"{self.task.id}/{self.case.id}#{self.sample_index}"


@dataclass(frozen=True)
class ScoredComponent:
    """One scorer's contribution to a trial, kept in the record for `evalctl show`."""

    name: str
    type: str
    value: float
    passed: bool
    weight: float
    required: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "value": round(self.value, 6),
            "passed": self.passed,
            "weight": self.weight,
            "required": self.required,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Verdict:
    score: float
    passed: bool
    components: tuple[ScoredComponent, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "passed": self.passed,
            "components": [c.as_dict() for c in self.components],
        }


def aggregate(components: Sequence[ScoredComponent], threshold: float | None = None) -> Verdict:
    """Combine scorers into a task-level verdict.

    * score  -- weighted mean of the component values.
    * passed -- every ``required`` component passed, and (if the task sets a
      ``pass_threshold``) the aggregate score clears it.

    "All required scorers must pass" rather than "the weighted score clears a
    bar" is the conservative reading, and it is what makes ``required: false``
    meaningful: a style rubric can drag the number down without ever flipping
    the bit a regression gate watches.
    """
    if not components:
        raise ScoringError("cannot aggregate an empty scorer list")
    total_weight = sum(c.weight for c in components)
    if total_weight > 0:
        score = sum(c.value * c.weight for c in components) / total_weight
    else:
        # Every weight was set to 0: fall back to an unweighted mean rather
        # than dividing by zero or silently reporting 0.
        score = sum(c.value for c in components) / len(components)
    passed = all(c.passed for c in components if c.required)
    if threshold is not None:
        passed = passed and score >= threshold
    return Verdict(score=min(1.0, max(0.0, score)), passed=passed, components=tuple(components))


class Scorer(abc.ABC):
    """Base class for scorers.

    ``config`` arrives already rendered against the case variables, so a spec
    can write ``target: "{{ answer }}"`` and the scorer just sees the value.
    """

    type: str = "base"
    needs_judge: bool = False

    def __init__(self, spec: ScorerSpec, config: Mapping[str, Any]) -> None:
        self.spec = spec
        self.config = dict(config)
        self.where = f"scoring[{spec.label}]"
        self.validate()

    def validate(self) -> None:
        """Check config at construction time, before any model is called."""
        return None

    @abc.abstractmethod
    async def score(self, response_text: str, ctx: ScoringContext) -> Score: ...

    # -- typed config access, with errors that name the scorer -------------

    def _missing(self, key: str) -> ScoringError:
        return ScoringError(
            f"{self.where}: scorer '{self.type}' requires '{key}' "
            f"(got keys: {', '.join(sorted(self.config)) or 'none'})"
        )

    def require(self, key: str) -> Any:
        if key not in self.config:
            raise self._missing(key)
        return self.config[key]

    def opt_str(self, key: str, default: str | None = None) -> str | None:
        value = self.config.get(key, default)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ScoringError(f"{self.where}: '{key}' must be a string, got {type(value).__name__}")
        return value

    def req_str(self, key: str) -> str:
        value = self.require(key)
        if not isinstance(value, str):
            raise ScoringError(f"{self.where}: '{key}' must be a string, got {type(value).__name__}")
        return value

    def as_text(self, value: Any, key: str) -> str:
        """Coerce a scalar config value to text.

        Case variables come from YAML, so ``answer: 55`` is an int and
        ``answer: "55"`` is a string. A target that only differs by which one
        the spec author happened to write should behave identically, so numbers
        are stringified here rather than rejected.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value) if isinstance(value, float) and not value.is_integer() else (
                str(int(value)) if isinstance(value, float) else str(value)
            )
        raise ScoringError(
            f"{self.where}: '{key}' must be text or a number, got {type(value).__name__}"
        )

    def opt_float(self, key: str, default: float | None = None) -> float | None:
        value = self.config.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoringError(f"{self.where}: '{key}' must be a number, got {type(value).__name__}")
        return float(value)

    def opt_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if not isinstance(value, bool):
            raise ScoringError(f"{self.where}: '{key}' must be true/false, got {type(value).__name__}")
        return value

    def opt_list(self, key: str, default: Sequence[Any] | None = None) -> list[Any] | None:
        value = self.config.get(key, default)
        if value is None:
            return None
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ScoringError(f"{self.where}: '{key}' must be a list, got {type(value).__name__}")
        return list(value)

    def str_list(self, key: str, default: Sequence[str] | None = None) -> list[str]:
        values = self.opt_list(key, default) or []
        return [self.as_text(item, f"{key}[{i}]") for i, item in enumerate(values)]

    def reject_unknown(self, allowed: Sequence[str]) -> None:
        """Same closed-key-set discipline as the spec loader.

        A scorer that ignores ``tolerence: 0.01`` grades every near-miss as
        wrong and never tells you why.
        """
        unknown = sorted(set(self.config) - set(allowed))
        if unknown:
            raise ScoringError(
                f"{self.where}: unknown option(s) {', '.join(unknown)} for scorer "
                f"'{self.type}'. Allowed: {', '.join(sorted(allowed))}"
            )


ScorerFactory = Callable[[ScorerSpec, Mapping[str, Any]], Scorer]
ScoreFn = Callable[[str, ScoringContext], Awaitable[Score]]
