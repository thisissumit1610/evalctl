"""Scorer registry.

`build()` renders a scorer's config against the case variables and constructs
it. Construction validates, so a bad rubric or an unknown option fails before
the run starts rather than on the trial that happens to use it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import ScoringError, SpecError
from ..spec import Case, ScorerSpec, Task
from ..templating import render_deep
from .base import Score, ScoredComponent, Scorer, ScoringContext, Verdict, aggregate
from .builtin import (
    ChoiceScorer,
    ContainsScorer,
    ExactMatchScorer,
    JSONMatchScorer,
    NotContainsScorer,
    NumericScorer,
    RegexScorer,
)
from .llm_judge import LLMJudgeScorer, judge_disagreement

_REGISTRY: dict[str, type[Scorer]] = {
    cls.type: cls
    for cls in (
        ExactMatchScorer,
        ContainsScorer,
        NotContainsScorer,
        RegexScorer,
        NumericScorer,
        ChoiceScorer,
        JSONMatchScorer,
        LLMJudgeScorer,
    )
}

# Aliases people reach for out of habit from other harnesses.
_ALIASES = {
    "exact": "exact_match",
    "match": "exact_match",
    "includes": "contains",
    "substring": "contains",
    "number": "numeric",
    "mcq": "choice",
    "multiple_choice": "choice",
    "json": "json_match",
    "judge": "llm_judge",
    "rubric": "llm_judge",
}


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


def register(scorer_cls: type[Scorer]) -> None:
    _REGISTRY[scorer_cls.type] = scorer_cls


def resolve(type_name: str) -> type[Scorer]:
    name = _ALIASES.get(type_name.strip().lower(), type_name.strip().lower())
    cls = _REGISTRY.get(name)
    if cls is None:
        raise SpecError(
            f"unknown scorer type '{type_name}'. Available: {', '.join(available_scorers())}"
        )
    return cls


def build(spec: ScorerSpec, variables: Mapping[str, Any], *, where: str = "scoring") -> Scorer:
    cls = resolve(spec.type)
    rendered = render_deep(dict(spec.config), variables, where=where)
    return cls(spec, rendered)


def build_for_case(task: Task, case: Case) -> list[Scorer]:
    """Construct every scorer for one case, with template vars already applied."""
    scorers: list[Scorer] = []
    for i, spec in enumerate(task.scoring_for(case)):
        where = f"{task.id}/{case.id} scoring[{i}]"
        try:
            scorers.append(build(spec, case.vars, where=where))
        except (ScoringError, SpecError) as exc:
            raise ScoringError(f"{where}: {exc}") from exc
    return scorers


def uses_judge(task: Task) -> bool:
    for case in task.cases:
        for spec in task.scoring_for(case):
            try:
                if resolve(spec.type).needs_judge:
                    return True
            except SpecError:
                continue
    return False


__all__ = [
    "Score",
    "ScoredComponent",
    "Scorer",
    "ScoringContext",
    "Verdict",
    "aggregate",
    "available_scorers",
    "build",
    "build_for_case",
    "judge_disagreement",
    "register",
    "resolve",
    "uses_judge",
]
